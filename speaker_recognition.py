from dataclasses import dataclass, replace
from pathlib import Path
from sklearn.metrics import roc_curve
import torch.nn.functional as F

import torch
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

class SpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path, use_cache: bool = True):
        super().__init__()

        self.wav_files = list(dataset_dir.rglob("*.wav"))
        self.labels = []
        self.cached_data: dict[int, tuple] = {}
        self.use_cache = use_cache

        class2idx = {}
        last_class_idx = -1
        for path in self.wav_files:
            class_name = path.parent.stem

            if class_name not in class2idx:
                last_class_idx += 1
                class2idx[class_name] = last_class_idx
            self.labels.append(class2idx[class_name])

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        if self.use_cache:
            if idx not in self.cached_data:
                wav, _ = torchaudio.load(self.wav_files[idx])
                self.cached_data[idx] = (idx, wav[0], self.labels[idx])
            return self.cached_data[idx]
        else:
            wav, _ = torchaudio.load(self.wav_files[idx])
            return (idx, wav[0], self.labels[idx])

class StackingSubsampling(torch.nn.Module):
    def __init__(self, stride, feat_in, feat_out):
        super().__init__()
        self.stride = stride
        self.out = torch.nn.Linear(stride * feat_in, feat_out)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = features.size()
        pad_size = (self.stride - (t % self.stride)) % self.stride
        features = torch.nn.functional.pad(features, (0, 0, 0, pad_size))
        _, t, _ = features.size()
        features = torch.reshape(features, (b, t // self.stride, d * self.stride))
        out_features = self.out(features)
        out_length = torch.div(
            features_length + pad_size, self.stride, rounding_mode="floor"
        )
        return out_features, out_length

class StatisticsPooling(torch.nn.Module):
    @staticmethod
    def get_length_mask(length):
        """
        length: B
        """
        max_len = length.max().long().item()

        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        return mask.to(length.dtype)

    def forward(self, encoded, encoded_len):
        """
        encoded: B x T x D
        encoded_len: B
        return: B x 2D
        """

        mask = self.get_length_mask(encoded_len).unsqueeze(2)  # B x T x 1

        total = encoded_len.unsqueeze(1)

        avg = (encoded * mask).sum(dim=1) / total

        std = torch.sqrt(
            (mask * (encoded - avg.unsqueeze(dim=1)) ** 2).sum(dim=1) / total
        )

        return torch.cat((avg, std), dim=1)

class AngularMarginSoftmax(torch.nn.Module):
    """
    Angular Margin Softmax Loss (AAM-Softmax / ArcFace)
    https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self, embedding_dim: int, num_classes: int, margin: float, scale: float
    ):
        super().__init__()
        self.margin = margin 
        self.scale = scale
        self.num_classes = num_classes
        
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: B x D - эмбеддинги из модели
        labels: B - метки классов
        return: scalar tensor - loss value
        """
        device = embeddings.device
        labels = labels.to(device)
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        weight_norm = torch.nn.functional.normalize(self.weight, p=2, dim=1)
        
        cosine = torch.nn.functional.linear(embeddings_norm, weight_norm)
        
        cosine = cosine.clamp(-1 + 1e-5, 1 - 1e-5)
        theta = torch.acos(cosine)

        one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        theta_m = theta + one_hot * self.margin

        cosine_m = torch.cos(theta_m)

        scaled_cosine_m = self.scale * cosine_m
        
        loss = self.criterion(scaled_cosine_m, labels)
        
        return loss

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: B x D
        return: B - predicted class indices
        """
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        weight_norm = torch.nn.functional.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings_norm, weight_norm)
        
        predictions = torch.argmax(cosine, dim=1)
        
        return predictions


def evaluate(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str
) -> float:
    
    model.eval()
    
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            _, wavs, labels = batch
            wavs = wavs.to(device)
            
            emb, _ = model.forward(wavs)
            
            embeddings_list.append(emb.cpu())
            labels_list.append(labels)
    
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    
    labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    mask = ~torch.eye(len(labels), dtype=torch.bool)
    
    genuine_scores = similarity_matrix[mask & labels_matrix].numpy()
    impostor_scores = similarity_matrix[mask & ~labels_matrix].numpy()
    
    eer = compute_eer(genuine_scores, impostor_scores)
    
    model.train()
    
    return eer


def compute_eer(genuine_scores, impostor_scores):
    """
    Вычисляет Equal Error Rate из genuine и impostor scores.
    
    Args:
        genuine_scores: scores для пар одного и того же спикера
        impostor_scores: scores для пар разных спикеров
    
    Returns:
        EER value
    """
    import numpy as np
    
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    
    sorted_indices = np.argsort(scores)
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]
    
    n_genuine = len(genuine_scores)
    n_impostor = len(impostor_scores)
    
    far = []
    frr = []
    
    for threshold in scores:
        false_accept = np.sum((scores[labels == 0]) >= threshold)
        far.append(false_accept / n_impostor if n_impostor > 0 else 0)
        
        false_reject = np.sum((scores[labels == 1]) < threshold)
        frr.append(false_reject / n_genuine if n_genuine > 0 else 0)
    
    far = np.array(far)
    frr = np.array(frr)
    
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)
    
    eer = (far[min_index] + frr[min_index]) / 2
    
    return eer

class SpecScaler(torch.nn.Module):
    def forward(self, spectrogram):
        return torch.log(spectrogram.clamp_(1e-9, 1e9))

class Conformer(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=conf.sample_rate,
                n_fft=conf.n_fft,
                win_length=conf.win_length,
                hop_length=conf.hop_length,
                n_mels=conf.n_mels,
            ),
            SpecScaler(),
        )
        self.subsampling = StackingSubsampling(conf.stride, conf.feat_in, conf.d_model)
        self.backbone = torchaudio.models.Conformer(
            input_dim=conf.d_model,
            num_heads=conf.n_heads,
            ffn_dim=conf.d_model * conf.ff_exp_factor,
            num_layers=conf.n_layers,
            depthwise_conv_kernel_size=conf.kernel_size,
            dropout=conf.dropout,
        )
        self.pooler = StatisticsPooling()
        self.extractor = torch.nn.Sequential(
            torch.nn.Linear(2 * conf.d_model, conf.d_model),
            torch.nn.ELU(),
            torch.nn.Linear(conf.d_model, conf.emb_size),
            torch.nn.ELU(),
        )
        self.proj = torch.nn.Sequential(torch.nn.Linear(conf.emb_size, conf.n_classes))

    def forward(self, wavs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        features = self.transform(wavs)

        features_length = (
            torch.ones(features.shape[0], device=features.device) * features.shape[2]
        ).to(torch.long)

        features = features.transpose(1, 2)
        features, features_length = self.subsampling(features, features_length)
        encoded, encoded_len = self.backbone(features, features_length)
        emb = self.pooler(encoded, encoded_len)
        emb = self.extractor(emb)
        scores = self.proj(emb)
        return emb, scores

@dataclass
class ModelParams:
    stride: int = 8
    feat_in: int = 64
    d_model: int = 32
    n_heads: int = 4
    ff_exp_factor: int = 2
    n_layers: int = 2
    kernel_size: int = 5
    dropout: float = 0.0
    emb_size: int = 16
    n_classes: int = 377
    sample_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64

@dataclass
class ModuleParams:
    dataset_dir: Path
    checkpoints_dir: Path
    log_dir: Path
    model_params: ModelParams
    angular_margin: float | None = None
    angular_scale: float | None = None
    use_cache: bool = True
    device: str = "cuda"
    n_epochs: int = 100
    batch_size: int = 64
    num_workers: int = 127
    learning_rate: float = 1e-2
    loss_function: str = "cross_entropy"  # "cross_entropy" or "angular_margin"
    validation_dir: Path | None = None
    validation_frequency: int = 5

def compute_eer(labels, scores):
    """
    Compute Equal Error Rate (EER)
    labels: array of true labels (0 or 1)
    scores: array of similarity scores
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return eer * 100


def evaluate(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str
) -> float:
    """
    Evaluate the model using Equal Error Rate (EER)
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            _, wavs, labels = batch
            embeddings, _ = model(wavs.to(device))
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    
    similarity_matrix = torch.mm(all_embeddings, all_embeddings.t())
    
    n = len(all_labels)
    pair_labels = []
    pair_scores = []
    
    num_samples = min(n, 1000)
    indices = np.random.choice(n, num_samples, replace=False)
    
    for i in indices:
        for j in range(i + 1, n):
            pair_labels.append(1 if all_labels[i] == all_labels[j] else 0)
            pair_scores.append(similarity_matrix[i, j].item())
    
    eer = compute_eer(np.array(pair_labels), np.array(pair_scores))
    
    model.train()
    return eer

def main(conf: ModuleParams) -> None:

    conf.log_dir.mkdir(exist_ok=True, parents=True)
    conf.checkpoints_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=conf.log_dir)

    dataset = SpeakerDataset(dataset_dir=conf.dataset_dir, use_cache=conf.use_cache)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=True
    )

    val_dataloader = None
    if conf.validation_dir and conf.validation_dir.exists():
        val_dataset = SpeakerDataset(
            dataset_dir=conf.validation_dir, use_cache=conf.use_cache
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=conf.batch_size,
            num_workers=conf.num_workers,
            shuffle=False,
        )

    n_classes = len(set(dataset.labels))

    model_params = conf.model_params
    model_params = replace(model_params, n_classes=n_classes)

    model = Conformer(model_params).to(conf.device)

    if conf.loss_function == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
        optim_params = model.parameters()
    elif conf.loss_function == "angular_margin":
        criterion = AngularMarginSoftmax(
            embedding_dim=model_params.emb_size,
            num_classes=n_classes,
            margin=conf.angular_margin,
            scale=conf.angular_scale,
        ).to(conf.device)
        optim_params = list(model.parameters()) + list(criterion.parameters())
    else:
        raise ValueError(f"Invalid loss function: {conf.loss_function}")

    optim = torch.optim.Adam(params=optim_params, lr=conf.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=0.7,     
        patience=2,      
        min_lr=1e-5,
    )


    pbar = tqdm(range(conf.n_epochs), position=0, leave=True)

    global_step = 0
    
    best_val_metric = None

    for epoch in pbar:
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for batch in train_dataloader:
            _, wavs, labels = batch
            wavs = wavs.to(conf.device)
            labels = labels.to(conf.device)

            emb, scores = model.forward(wavs.to(conf.device))

            optim.zero_grad()
#             loss = None
            if conf.loss_function == "cross_entropy":
                loss = criterion(scores, labels)
                predictions = torch.argmax(scores, dim=1)
            else:
                loss = criterion(emb, labels)
                predictions = criterion.predict(emb)

#             loss = criterion(scores, labels.to(conf.device))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

#             predictions = torch.argmax(scores, dim=1)

            correct = (predictions == labels.to(conf.device)).sum().item()
            epoch_correct += correct
            epoch_total += labels.size(0)

            epoch_losses.append(loss.item())

            writer.add_scalar("Loss/Batch", loss.item(), global_step)
            writer.add_scalar("Accuracy/Batch", correct / labels.size(0), global_step)
            writer.add_scalar("Learning_Rate", optim.param_groups[0]["lr"], global_step)

            global_step += 1

            pbar.set_postfix({"batch_loss": f"{loss.item():.2f}"})

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_accuracy = epoch_correct / epoch_total

        writer.add_scalar("Loss/Epoch", avg_epoch_loss, epoch)
        writer.add_scalar("Accuracy/Epoch", epoch_accuracy, epoch)
        
        val_metric_for_scheduler = avg_epoch_loss

        if val_dataloader and (epoch + 1) % conf.validation_frequency == 0:
            print(f"\nRunning validation evaluation at epoch {epoch + 1}...")
            try:
                eer = evaluate(model, val_dataloader, conf.device)
            except NotImplementedError:
                eer = -1
            writer.add_scalar("Validation/EER", eer, epoch)
            
            if eer >= 0:
                val_metric_for_scheduler = eer

            if best_val_metric is None or val_metric_for_scheduler < best_val_metric:
                best_val_metric = val_metric_for_scheduler
            
        scheduler.step(val_metric_for_scheduler)

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": avg_epoch_loss,
                "accuracy": epoch_accuracy,
                "loss_function": conf.loss_function,
                "angular_margin": (
                    conf.angular_margin
                    if conf.loss_function == "angular_margin"
                    else None
                ),
                "angular_scale": (
                    conf.angular_scale
                    if conf.loss_function == "angular_margin"
                    else None
                ),
            },
            conf.checkpoints_dir / f"epoch_{epoch + 1}.ckpt",
        )

    writer.close()

if __name__ == "__main__":

    params = ModuleParams(
        dataset_dir=Path("./data/train"),
        use_cache=True,
        checkpoints_dir=Path("./checkpoints/aam_optimal"),
        model_params=ModelParams(emb_size=32, dropout=0.05),
        device="cuda",
        angular_margin=0.2,
        angular_scale=20,
        num_workers=127,
        n_epochs=60,
        log_dir=Path("./logs/aam_optimal"),
        loss_function="angular_margin",
        learning_rate=1e-3,
        validation_dir=Path("./data/dev"),
        validation_frequency=1,
    )
    main(params)
















