"""实验2：VAE 振动信号数据扩充。

本脚本复用实验0保存的窗口数据，训练 1D-CNN-VAE（1维卷积变分自编码器），
并输出两类 VAE 生成结果：
1. VAE先验采样：直接从标准正态潜在空间采样并解码；
2. VAE残差扩充：保留真实窗口主体形态，只叠加潜在空间扰动带来的小残差。

第二种方式更适合当前“短周期健康振动信号扩充”的数据条件。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import rfft, rfftfreq
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.experiment0_preprocess_eval import (
    PROJECT_ROOT,
    build_feature_summary,
    compute_features,
    load_config,
    setup_chinese_font,
)
from src.experiment1_traditional_augmentation import compare_features, plot_pca_comparison, save_windows_csv


@dataclass(frozen=True)
class VAEConfig:
    latent_dim: int = 32
    beta: float = 0.0005
    epochs: int = 120
    batch_size: int = 64
    learning_rate: float = 0.001
    val_ratio: float = 0.15
    generated_count: int | None = None
    residual_latent_noise: float = 0.80
    residual_scale: float = 1.00
    clamp_percentile_low: float = 0.2
    clamp_percentile_high: float = 99.8


class Conv1dVAE(nn.Module):
    """用于 2048 点振动窗口的 1D-CNN-VAE。"""

    def __init__(self, window_length: int, latent_dim: int) -> None:
        super().__init__()
        if window_length != 2048:
            raise ValueError("当前模型结构按 2048 点窗口设计，如需其他长度请同步调整卷积结构。")

        self.window_length = window_length
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.feature_shape = (128, window_length // 16)
        flat_dim = self.feature_shape[0] * self.feature_shape[1]
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(16, 1, kernel_size=8, stride=2, padding=3),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.fc_decode(z).view(z.shape[0], *self.feature_shape)
        return self.decoder(decoded)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def read_raw_config(config_path: Path) -> dict[str, object]:
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        return {}


def load_vae_config(config_path: Path, args: argparse.Namespace) -> VAEConfig:
    raw = read_raw_config(config_path)
    exp_cfg = raw.get("experiment2", {}) if isinstance(raw, dict) else {}
    cfg = VAEConfig(
        latent_dim=int(exp_cfg.get("latent_dim", 32)),
        beta=float(exp_cfg.get("beta", 0.0005)),
        epochs=int(exp_cfg.get("epochs", 120)),
        batch_size=int(exp_cfg.get("batch_size", 64)),
        learning_rate=float(exp_cfg.get("learning_rate", 0.001)),
        val_ratio=float(exp_cfg.get("val_ratio", 0.15)),
        generated_count=exp_cfg.get("generated_count"),
        residual_latent_noise=float(exp_cfg.get("residual_latent_noise", 0.80)),
        residual_scale=float(exp_cfg.get("residual_scale", 1.00)),
        clamp_percentile_low=float(exp_cfg.get("clamp_percentile_low", 0.2)),
        clamp_percentile_high=float(exp_cfg.get("clamp_percentile_high", 99.8)),
    )
    if args.epochs is not None:
        cfg = dataclass_replace(cfg, epochs=args.epochs)
    if args.batch_size is not None:
        cfg = dataclass_replace(cfg, batch_size=args.batch_size)
    if args.generated_count is not None:
        cfg = dataclass_replace(cfg, generated_count=args.generated_count)
    return cfg


def dataclass_replace(cfg: VAEConfig, **kwargs: object) -> VAEConfig:
    data = asdict(cfg)
    data.update(kwargs)
    return VAEConfig(**data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "model": output_dir / "01_模型与训练记录",
        "generated": output_dir / "02_生成数据",
        "features": output_dir / "03_特征表",
        "figures": output_dir / "04_评价图表",
        "report": output_dir / "05_实验说明",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_standardize_stats(path: Path) -> tuple[float, float]:
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    return float(stats["全局均值"]), float(stats["全局标准差"])


def restore_to_original_scale(standardized: np.ndarray, mean: float, std: float, remove_mean: bool) -> np.ndarray:
    restored = standardized * std + mean
    if remove_mean:
        restored = restored - restored.mean(axis=1, keepdims=True)
    return restored.astype(np.float32)


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss


def train_vae(
    windows_standardized: np.ndarray,
    cfg,
    vae_cfg: VAEConfig,
    dirs: dict[str, Path],
    device: torch.device,
) -> tuple[Conv1dVAE, pd.DataFrame]:
    tensor = torch.from_numpy(windows_standardized).float().unsqueeze(1)
    dataset = TensorDataset(tensor)
    val_count = max(1, int(round(len(dataset) * vae_cfg.val_ratio)))
    train_count = len(dataset) - val_count
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [train_count, val_count], generator=generator)
    train_loader = DataLoader(train_set, batch_size=vae_cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=vae_cfg.batch_size, shuffle=False, drop_last=False)

    model = Conv1dVAE(cfg.window_length, vae_cfg.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=vae_cfg.learning_rate)
    best_val = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, vae_cfg.epochs + 1):
        model.train()
        train_loss = train_recon = train_kl = 0.0
        train_seen = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, vae_cfg.beta)
            loss.backward()
            optimizer.step()

            batch_size = batch.shape[0]
            train_seen += batch_size
            train_loss += float(loss.detach().cpu()) * batch_size
            train_recon += float(recon_loss.detach().cpu()) * batch_size
            train_kl += float(kl_loss.detach().cpu()) * batch_size

        model.eval()
        val_loss = val_recon = val_kl = 0.0
        val_seen = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, vae_cfg.beta)
                batch_size = batch.shape[0]
                val_seen += batch_size
                val_loss += float(loss.cpu()) * batch_size
                val_recon += float(recon_loss.cpu()) * batch_size
                val_kl += float(kl_loss.cpu()) * batch_size

        row = {
            "轮次": epoch,
            "训练总损失": train_loss / train_seen,
            "训练重构误差": train_recon / train_seen,
            "训练KL散度": train_kl / train_seen,
            "验证总损失": val_loss / val_seen,
            "验证重构误差": val_recon / val_seen,
            "验证KL散度": val_kl / val_seen,
        }
        history.append(row)

        if row["验证总损失"] < best_val:
            best_val = float(row["验证总损失"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vae_config": asdict(vae_cfg),
                    "window_length": cfg.window_length,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                dirs["model"] / "vae_x2_best.pt",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == vae_cfg.epochs:
            print(
                f"轮次 {epoch:03d}/{vae_cfg.epochs} | "
                f"训练损失 {row['训练总损失']:.6f} | 验证损失 {row['验证总损失']:.6f}"
            )

    checkpoint = torch.load(dirs["model"] / "vae_x2_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    history_df = pd.DataFrame(history)
    history_df.to_csv(dirs["model"] / "VAE训练历史.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vae_config": asdict(vae_cfg),
            "window_length": cfg.window_length,
            "history": history,
        },
        dirs["model"] / "vae_x2_final.pt",
    )
    return model, history_df


def generate_prior(model: Conv1dVAE, count: int, latent_dim: int, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, count, batch_size):
            current = min(batch_size, count - start)
            z = torch.randn(current, latent_dim, device=device)
            decoded = model.decode(z).squeeze(1).cpu().numpy()
            outputs.append(decoded)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def generate_residual(
    model: Conv1dVAE,
    windows_standardized: np.ndarray,
    count: int,
    vae_cfg: VAEConfig,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rng = np.random.default_rng(seed)
    indices = rng.choice(windows_standardized.shape[0], size=count, replace=True)
    base = torch.from_numpy(windows_standardized[indices]).float().unsqueeze(1).to(device)

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, count, vae_cfg.batch_size):
            batch = base[start : start + vae_cfg.batch_size]
            mu, logvar = model.encode(batch)
            std = torch.exp(0.5 * logvar)
            z_base = mu
            z_perturbed = mu + vae_cfg.residual_latent_noise * torch.randn_like(std) * std
            decoded_base = model.decode(z_base)
            decoded_perturbed = model.decode(z_perturbed)
            residual = decoded_perturbed - decoded_base
            generated = batch + vae_cfg.residual_scale * residual
            outputs.append(generated.squeeze(1).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def clamp_generated(generated: np.ndarray, reference: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    low = float(np.percentile(reference, low_percentile))
    high = float(np.percentile(reference, high_percentile))
    return np.clip(generated, low, high).astype(np.float32)


def plot_training_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["轮次"], history["训练总损失"], label="训练总损失", linewidth=1.2)
    axes[0].plot(history["轮次"], history["验证总损失"], label="验证总损失", linewidth=1.2)
    axes[0].set_title("VAE 总损失变化")
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("损失")
    axes[0].grid(alpha=0.22)
    axes[0].legend()

    axes[1].plot(history["轮次"], history["训练重构误差"], label="训练重构误差", linewidth=1.2)
    axes[1].plot(history["轮次"], history["验证重构误差"], label="验证重构误差", linewidth=1.2)
    axes[1].set_title("VAE 重构误差变化")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("均方误差")
    axes[1].grid(alpha=0.22)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_waveform_comparison(real_windows: np.ndarray, generated_by_method: dict[str, np.ndarray], sampling_frequency: int, path: Path) -> None:
    time_ms = np.arange(real_windows.shape[1]) / sampling_frequency * 1000
    fig, axes = plt.subplots(len(generated_by_method), 1, figsize=(11, 5.8), sharex=True)
    if len(generated_by_method) == 1:
        axes = [axes]
    for ax, (method_name, generated) in zip(axes, generated_by_method.items()):
        ax.plot(time_ms, real_windows[0], color="#4C78A8", linewidth=0.8, label="真实样本")
        ax.plot(time_ms, generated[0], color="#F58518", linewidth=0.8, alpha=0.85, label=method_name)
        ax.set_title(method_name)
        ax.set_ylabel("幅值")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("时间 / ms")
    fig.suptitle("VAE 生成样本与真实样本波形对比")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mean_spectrum_comparison(real_windows: np.ndarray, generated_by_method: dict[str, np.ndarray], sampling_frequency: int, path: Path) -> None:
    freqs = rfftfreq(real_windows.shape[1], d=1.0 / sampling_frequency)
    real_amp = np.abs(rfft(real_windows - real_windows.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(freqs, real_amp, color="#222222", linewidth=1.3, label="真实样本")
    colors = ["#F58518", "#54A24B"]
    for color, (method_name, generated) in zip(colors, generated_by_method.items()):
        gen_amp = np.abs(rfft(generated - generated.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
        ax.plot(freqs, gen_amp, color=color, linewidth=0.95, alpha=0.9, label=method_name)
    ax.set_title("VAE 生成样本与真实样本平均频谱对比")
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("平均幅值")
    ax.set_xlim(0, sampling_frequency / 2)
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_metric_summary(summary: pd.DataFrame, path: Path) -> None:
    metric_cols = ["平均均值相对误差_%", "平均JS散度", "MMD距离"]
    titles = ["平均均值相对误差 / %", "平均 JS 散度", "MMD 距离"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    colors = ["#F58518", "#54A24B"]
    for ax, col, title in zip(axes, metric_cols, titles):
        ax.bar(summary["方法"], summary[col], color=colors[: len(summary)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("VAE 扩充方法指标汇总")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    output_dir: Path,
    cfg,
    vae_cfg: VAEConfig,
    device: torch.device,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    real_count: int,
) -> None:
    best_epoch = history.sort_values("验证总损失").iloc[0]
    best_mmd = summary.sort_values("MMD距离").iloc[0]
    table_cols = ["方法", "生成样本数量", "平均均值相对误差_%", "平均JS散度", "MMD距离"]
    lines = [
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join(["---"] * len(table_cols)) + " |",
    ]
    for _, row in summary[table_cols].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["方法"]),
                    str(int(row["生成样本数量"])),
                    f"{row['平均均值相对误差_%']:.6g}",
                    f"{row['平均JS散度']:.6g}",
                    f"{row['MMD距离']:.6g}",
                ]
            )
            + " |"
        )
    table_text = "\n".join(lines)

    report = f"""# 实验2：VAE 振动信号数据扩充

## 实验目的

本实验训练 VAE（变分自编码器）学习真实主轴振动窗口的潜在分布，并生成新的振动窗口样本。VAE 的编码器把 2048 点振动窗口压缩为低维潜在变量，解码器再从潜在变量恢复振动窗口。

## 数据与模型参数

- 输入数据：实验0保存的 `{cfg.channel}` 通道标准化窗口；
- 真实窗口数量：{real_count}；
- 单个窗口长度：{cfg.window_length} 个采样点；
- 采样频率：{cfg.sampling_frequency} Hz；
- 主轴转速：{cfg.spindle_speed_rpm} rpm；
- 潜在维度：{vae_cfg.latent_dim}；
- beta 系数：{vae_cfg.beta}；
- 训练轮数：{vae_cfg.epochs}；
- 批大小：{vae_cfg.batch_size}；
- 学习率：{vae_cfg.learning_rate}；
- 训练设备：{device}。

## 生成方式

- VAE先验采样：从标准正态分布直接采样潜在变量并解码，能体现模型独立生成能力，但在小样本健康振动数据上可能更容易产生过平滑波形；
- VAE残差扩充：先选取真实窗口作为基础，再在潜在空间施加扰动，只把解码差异作为残差叠加回真实窗口，通常更容易保持原始振动形态。

## 训练结果

- 最优验证总损失出现在第 {int(best_epoch["轮次"])} 轮；
- 最优验证总损失：{best_epoch["验证总损失"]:.6g}；
- 对应验证重构误差：{best_epoch["验证重构误差"]:.6g}；
- 对应验证 KL 散度：{best_epoch["验证KL散度"]:.6g}。

## 指标汇总

{table_text}

从本次结果看，按 MMD 距离（最大均值差异，越小越接近真实样本整体分布）排序，较优方式是 **{best_mmd["方法"]}**。需要注意，这只是本次训练得到的实验结果；论文正式结果建议固定随机种子、完整训练并与后续 WGAN-GP 系列实验一起统一比较。

## 输出文件说明

- `01_模型与训练记录/`：VAE 最优模型、最终模型和训练历史；
- `02_生成数据/`：VAE 生成窗口数据，包含 `.npy` 和 `.csv`；
- `03_特征表/`：生成样本特征、单特征误差明细和方法指标汇总；
- `04_评价图表/`：训练损失、波形对比、平均频谱、PCA 分布和指标汇总图。
"""
    (output_dir / "05_实验说明" / "实验2说明.md").write_text(report, encoding="utf-8")


def evaluate_generated(
    real_windows: np.ndarray,
    generated_by_method: dict[str, np.ndarray],
    cfg,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    real_features = compute_features(real_windows, cfg.sampling_frequency, cfg.spindle_speed_rpm)
    all_detail: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for offset, (method_name, generated_windows) in enumerate(generated_by_method.items()):
        generated_features = compute_features(generated_windows, cfg.sampling_frequency, cfg.spindle_speed_rpm)
        generated_features.insert(0, "样本编号", np.arange(generated_features.shape[0]))
        generated_features.to_csv(dirs["features"] / f"{method_name}_公共评价特征.csv", index=False, encoding="utf-8-sig")
        build_feature_summary(generated_features.drop(columns=["样本编号"])).to_csv(
            dirs["features"] / f"{method_name}_特征统计摘要.csv",
            encoding="utf-8-sig",
        )

        detail, metrics = compare_features(
            real_features,
            generated_features.drop(columns=["样本编号"]),
            method_name,
            cfg.seed + 100 + offset,
        )
        all_detail.append(detail)
        summary_rows.append({"方法": method_name, "生成样本数量": generated_windows.shape[0], **metrics})

        plot_pca_comparison(
            real_features,
            generated_features.drop(columns=["样本编号"]),
            method_name,
            dirs["figures"] / f"PCA特征分布_{method_name}.png",
        )

    detail_table = pd.concat(all_detail, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    detail_table.to_csv(dirs["features"] / "VAE_单特征误差明细.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(dirs["features"] / "VAE_方法指标汇总.csv", index=False, encoding="utf-8-sig")
    return summary


def run(config_path: Path, args: argparse.Namespace) -> None:
    cfg = load_config(config_path)
    vae_cfg = load_vae_config(config_path, args)
    set_seed(cfg.seed)
    setup_chinese_font()

    experiment0_dir = cfg.output_dir
    processed_dir = experiment0_dir / "01_预处理数据"
    real_path = processed_dir / "X-2_窗口数据_原始.npy"
    standardized_path = processed_dir / "X-2_窗口数据_标准化.npy"
    stats_path = processed_dir / "标准化参数.json"
    if not real_path.exists() or not standardized_path.exists() or not stats_path.exists():
        raise FileNotFoundError("没有找到实验0的预处理结果，请先运行实验0。")

    output_dir = PROJECT_ROOT / "实验2_VAE"
    dirs = ensure_dirs(output_dir)

    real_windows = np.load(real_path).astype(np.float32)
    windows_standardized = np.load(standardized_path).astype(np.float32)
    mean, std = load_standardize_stats(stats_path)
    generated_count = vae_cfg.generated_count or real_windows.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"训练样本形状：{windows_standardized.shape}")
    print(f"计划生成样本数量：{generated_count}")

    model, history = train_vae(windows_standardized, cfg, vae_cfg, dirs, device)
    plot_training_history(history, dirs["figures"] / "VAE训练损失曲线.png")

    prior_std = generate_prior(model, generated_count, vae_cfg.latent_dim, device, vae_cfg.batch_size)
    residual_std = generate_residual(model, windows_standardized, generated_count, vae_cfg, cfg.seed + 202, device)
    prior_std = clamp_generated(prior_std, windows_standardized, vae_cfg.clamp_percentile_low, vae_cfg.clamp_percentile_high)
    residual_std = clamp_generated(residual_std, windows_standardized, vae_cfg.clamp_percentile_low, vae_cfg.clamp_percentile_high)

    generated_by_method = {
        "VAE先验采样": restore_to_original_scale(prior_std, mean, std, cfg.remove_window_mean),
        "VAE残差扩充": restore_to_original_scale(residual_std, mean, std, cfg.remove_window_mean),
    }

    for method_name, generated_windows in generated_by_method.items():
        np.save(dirs["generated"] / f"{method_name}_窗口数据.npy", generated_windows)
        save_windows_csv(dirs["generated"] / f"{method_name}_窗口数据.csv", generated_windows)

    summary = evaluate_generated(real_windows, generated_by_method, cfg, dirs)
    plot_waveform_comparison(real_windows, generated_by_method, cfg.sampling_frequency, dirs["figures"] / "VAE_波形对比.png")
    plot_mean_spectrum_comparison(real_windows, generated_by_method, cfg.sampling_frequency, dirs["figures"] / "VAE_平均频谱对比.png")
    plot_metric_summary(summary, dirs["figures"] / "VAE_指标汇总.png")

    run_info = {
        "实验名称": "实验2：VAE",
        "输入真实窗口": str(real_path),
        "输入标准化窗口": str(standardized_path),
        "输出目录": str(output_dir),
        "真实窗口数量": int(real_windows.shape[0]),
        "生成窗口数量": int(generated_count),
        "窗口长度": int(real_windows.shape[1]),
        "设备": str(device),
        "随机种子": int(cfg.seed),
        "VAE参数": asdict(vae_cfg),
    }
    with (dirs["report"] / "实验2运行信息.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    write_report(output_dir, cfg, vae_cfg, device, history, summary, real_windows.shape[0])

    print("实验2完成")
    print(f"输出目录：{output_dir}")
    print("VAE 方法指标汇总：")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验2：VAE 振动信号数据扩充")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="配置文件路径，默认读取项目根目录 config.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖配置中的批大小")
    parser.add_argument("--generated-count", type=int, default=None, help="覆盖生成样本数量")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args.config.resolve(), cli_args)
