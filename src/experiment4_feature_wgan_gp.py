"""实验4：特征约束 WGAN-GP 振动信号数据扩充。

本脚本在普通 WGAN-GP 基础上加入振动特征约束：
1. 生成器输入随机噪声和目标特征条件；
2. 判别器同时判断“振动窗口 + 特征条件”是否来自真实数据；
3. 训练生成器时加入特征损失，使生成窗口的 RMS、峰值、频带能量等特征接近目标条件。

这里的 WGAN-GP 是“带梯度惩罚的 Wasserstein 生成对抗网络”，
特征约束的作用是让生成结果不仅波形看起来像真实样本，而且常用振动统计特征也更接近真实分布。
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
from torch.utils.data import DataLoader, TensorDataset

from src.experiment0_preprocess_eval import (
    PROJECT_ROOT,
    build_feature_summary,
    compute_features,
    load_config,
    setup_chinese_font,
)
from src.experiment1_traditional_augmentation import compare_features, plot_pca_comparison, save_windows_csv
from src.experiment2_vae import load_standardize_stats, restore_to_original_scale
from src.experiment3_wgan_gp import clamp_generated


FEATURE_COLUMNS = [
    "均方根_RMS",
    "峰值_Peak",
    "峰峰值_PeakToPeak",
    "偏度_Skewness",
    "峭度_Kurtosis",
    "峰值因子_CrestFactor",
    "频率重心_Hz",
    "0到1kHz能量占比",
    "1到3kHz能量占比",
    "3到5kHz能量占比",
    "一倍频能量占比",
    "二倍频能量占比",
    "三四倍频能量占比",
]


@dataclass(frozen=True)
class FeatureWGANConfig:
    latent_dim: int = 128
    epochs: int = 120
    batch_size: int = 64
    generator_learning_rate: float = 0.0001
    discriminator_learning_rate: float = 0.0001
    critic_steps: int = 3
    gradient_penalty_weight: float = 10.0
    feature_loss_weight: float = 8.0
    generated_count: int | None = None
    clamp_percentile_low: float = 0.2
    clamp_percentile_high: float = 99.8


class FeatureConditionedGenerator(nn.Module):
    """特征约束 WGAN-GP 生成器：随机噪声 + 目标特征 -> 2048 点振动窗口。"""

    def __init__(self, latent_dim: int, condition_dim: int, window_length: int) -> None:
        super().__init__()
        if window_length != 2048:
            raise ValueError("当前网络按 2048 点窗口设计，如需其他长度请同步调整网络结构。")
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.window_length = window_length
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256 * 128),
            nn.BatchNorm1d(256 * 128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(32, 1, kernel_size=8, stride=2, padding=3),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, condition], dim=1)
        x = self.fc(x).view(z.shape[0], 256, 128)
        return self.net(x)


class FeatureConditionedCritic(nn.Module):
    """特征约束 WGAN-GP 判别器：同时输入振动窗口和条件特征。"""

    def __init__(self, condition_dim: int, window_length: int) -> None:
        super().__init__()
        if window_length != 2048:
            raise ValueError("当前网络按 2048 点窗口设计，如需其他长度请同步调整网络结构。")
        self.signal_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.condition_net = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * (window_length // 16) + 64, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        signal_features = self.signal_net(x).flatten(start_dim=1)
        condition_features = self.condition_net(condition)
        return self.fc(torch.cat([signal_features, condition_features], dim=1)).view(-1)


def read_raw_config(config_path: Path) -> dict[str, object]:
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        return {}


def replace_config(cfg: FeatureWGANConfig, **kwargs: object) -> FeatureWGANConfig:
    data = asdict(cfg)
    data.update(kwargs)
    return FeatureWGANConfig(**data)


def load_feature_wgan_config(config_path: Path, args: argparse.Namespace) -> FeatureWGANConfig:
    raw = read_raw_config(config_path)
    exp_cfg = raw.get("experiment4", {}) if isinstance(raw, dict) else {}
    fallback_cfg = raw.get("experiment3", {}) if isinstance(raw, dict) else {}
    merged = {**fallback_cfg, **exp_cfg}
    cfg = FeatureWGANConfig(
        latent_dim=int(merged.get("latent_dim", 128)),
        epochs=int(merged.get("epochs", 120)),
        batch_size=int(merged.get("batch_size", 64)),
        generator_learning_rate=float(merged.get("generator_learning_rate", 0.0001)),
        discriminator_learning_rate=float(merged.get("discriminator_learning_rate", 0.0001)),
        critic_steps=int(merged.get("critic_steps", 3)),
        gradient_penalty_weight=float(merged.get("gradient_penalty_weight", 10.0)),
        feature_loss_weight=float(merged.get("feature_loss_weight", 8.0)),
        generated_count=merged.get("generated_count"),
        clamp_percentile_low=float(merged.get("clamp_percentile_low", 0.2)),
        clamp_percentile_high=float(merged.get("clamp_percentile_high", 99.8)),
    )
    if args.epochs is not None:
        cfg = replace_config(cfg, epochs=args.epochs)
    if args.batch_size is not None:
        cfg = replace_config(cfg, batch_size=args.batch_size)
    if args.generated_count is not None:
        cfg = replace_config(cfg, generated_count=args.generated_count)
    if args.critic_steps is not None:
        cfg = replace_config(cfg, critic_steps=args.critic_steps)
    if args.feature_loss_weight is not None:
        cfg = replace_config(cfg, feature_loss_weight=args.feature_loss_weight)
    return cfg


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


def build_condition_table(
    windows_standardized: np.ndarray,
    sampling_frequency: int,
    spindle_speed_rpm: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    features = compute_features(windows_standardized, sampling_frequency, spindle_speed_rpm)
    raw = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = ((raw - mean) / std).astype(np.float32)
    stats = {
        "特征名称": FEATURE_COLUMNS,
        "条件特征均值": mean.ravel().astype(float).tolist(),
        "条件特征标准差": std.ravel().astype(float).tolist(),
    }
    return raw, normalized, stats


def make_frequency_tensors(window_length: int, sampling_frequency: int, spindle_speed_rpm: int, device: torch.device) -> dict[str, torch.Tensor]:
    freqs = torch.fft.rfftfreq(window_length, d=1.0 / sampling_frequency).to(device)
    base_freq = spindle_speed_rpm / 60.0
    masks = {
        "freqs": freqs,
        "0到1kHz能量占比": ((freqs >= 0) & (freqs < 1000)).float(),
        "1到3kHz能量占比": ((freqs >= 1000) & (freqs < 3000)).float(),
        "3到5kHz能量占比": ((freqs >= 3000) & (freqs < 5000)).float(),
        "一倍频能量占比": ((freqs >= base_freq - 5.0) & (freqs <= base_freq + 5.0)).float(),
        "二倍频能量占比": ((freqs >= 2 * base_freq - 5.0) & (freqs <= 2 * base_freq + 5.0)).float(),
        "三四倍频能量占比": ((freqs >= 3.5 * base_freq - base_freq) & (freqs <= 3.5 * base_freq + base_freq)).float(),
    }
    return masks


def torch_condition_features(
    windows: torch.Tensor,
    frequency_tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    x = windows.squeeze(1)
    centered = x - x.mean(dim=1, keepdim=True)
    eps = 1e-12
    rms = torch.sqrt(torch.mean(centered.pow(2), dim=1) + eps)
    peak = centered.abs().amax(dim=1)
    peak_to_peak = centered.amax(dim=1) - centered.amin(dim=1)
    std = centered.std(dim=1, unbiased=False) + eps
    standardized = centered / std.unsqueeze(1)
    skewness = standardized.pow(3).mean(dim=1)
    kurtosis = standardized.pow(4).mean(dim=1)
    crest_factor = peak / (rms + eps)

    spectrum = torch.fft.rfft(centered, dim=1).abs()
    power = spectrum.pow(2)
    power_sum = power.sum(dim=1) + eps
    freqs = frequency_tensors["freqs"]
    freq_centroid = (power * freqs.unsqueeze(0)).sum(dim=1) / power_sum

    ratios = []
    for name in FEATURE_COLUMNS[7:]:
        mask = frequency_tensors[name].unsqueeze(0)
        ratios.append((power * mask).sum(dim=1) / power_sum)

    return torch.stack(
        [
            rms,
            peak,
            peak_to_peak,
            skewness,
            kurtosis,
            crest_factor,
            freq_centroid,
            *ratios,
        ],
        dim=1,
    )


def gradient_penalty(
    critic: FeatureConditionedCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    condition: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real.shape[0]
    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated, condition)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def save_training_snapshot(
    history: list[dict[str, float | int]],
    generator: FeatureConditionedGenerator,
    critic: FeatureConditionedCritic,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    wgan_cfg: FeatureWGANConfig,
    cfg,
    dirs: dict[str, Path],
    epoch: int,
    best_objective: float,
) -> None:
    """保存可恢复训练的最新快照，长轮次训练中断时可保留中间结果。"""
    pd.DataFrame(history).to_csv(
        dirs["model"] / "特征约束WGAN-GP训练历史_自动保存.csv",
        index=False,
        encoding="utf-8-sig",
    )
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "generator_optimizer_state_dict": opt_g.state_dict(),
            "critic_optimizer_state_dict": opt_d.state_dict(),
            "wgan_config": asdict(wgan_cfg),
            "feature_columns": FEATURE_COLUMNS,
            "window_length": cfg.window_length,
            "epoch": epoch,
            "best_objective": best_objective,
            "history": history,
        },
        dirs["model"] / "feature_wgan_gp_x2_latest.pt",
    )
    with (dirs["model"] / "最新训练快照信息.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "最新保存轮次": int(epoch),
                "当前最优目标值": float(best_objective),
                "最新模型文件": "feature_wgan_gp_x2_latest.pt",
                "自动保存历史文件": "特征约束WGAN-GP训练历史_自动保存.csv",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def train_feature_wgan_gp(
    windows_standardized: np.ndarray,
    conditions_raw: np.ndarray,
    conditions_normalized: np.ndarray,
    cfg,
    wgan_cfg: FeatureWGANConfig,
    dirs: dict[str, Path],
    device: torch.device,
) -> tuple[FeatureConditionedGenerator, FeatureConditionedCritic, pd.DataFrame]:
    tensor_x = torch.from_numpy(windows_standardized).float().unsqueeze(1)
    tensor_condition = torch.from_numpy(conditions_normalized).float()
    tensor_condition_raw = torch.from_numpy(conditions_raw).float()
    loader = DataLoader(
        TensorDataset(tensor_x, tensor_condition, tensor_condition_raw),
        batch_size=wgan_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("训练批次数为 0，请减小 batch_size 或检查预处理窗口数量。")

    condition_dim = conditions_normalized.shape[1]
    generator = FeatureConditionedGenerator(wgan_cfg.latent_dim, condition_dim, cfg.window_length).to(device)
    critic = FeatureConditionedCritic(condition_dim, cfg.window_length).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=wgan_cfg.generator_learning_rate, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=wgan_cfg.discriminator_learning_rate, betas=(0.0, 0.9))

    feature_mean = torch.from_numpy(conditions_raw.mean(axis=0, keepdims=True).astype(np.float32)).to(device)
    feature_std_np = conditions_raw.std(axis=0, keepdims=True).astype(np.float32)
    feature_std_np = np.where(feature_std_np < 1e-8, 1.0, feature_std_np)
    feature_std = torch.from_numpy(feature_std_np).to(device)
    frequency_tensors = make_frequency_tensors(cfg.window_length, cfg.sampling_frequency, cfg.spindle_speed_rpm, device)

    history: list[dict[str, float | int]] = []
    best_objective = float("inf")
    for epoch in range(1, wgan_cfg.epochs + 1):
        d_loss_sum = g_adv_sum = g_feature_sum = g_total_sum = wasserstein_sum = gp_sum = 0.0
        d_updates = g_updates = 0

        for real_batch, condition_batch, _ in loader:
            real_batch = real_batch.to(device)
            condition_batch = condition_batch.to(device)
            batch_size = real_batch.shape[0]

            for _ in range(wgan_cfg.critic_steps):
                z = torch.randn(batch_size, wgan_cfg.latent_dim, device=device)
                fake_batch = generator(z, condition_batch).detach()
                real_score = critic(real_batch, condition_batch)
                fake_score = critic(fake_batch, condition_batch)
                wasserstein = real_score.mean() - fake_score.mean()
                gp = gradient_penalty(critic, real_batch, fake_batch, condition_batch, device)
                d_loss = -wasserstein + wgan_cfg.gradient_penalty_weight * gp

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

                d_loss_sum += float(d_loss.detach().cpu())
                wasserstein_sum += float(wasserstein.detach().cpu())
                gp_sum += float(gp.detach().cpu())
                d_updates += 1

            z = torch.randn(batch_size, wgan_cfg.latent_dim, device=device)
            fake_batch = generator(z, condition_batch)
            g_adv_loss = -critic(fake_batch, condition_batch).mean()
            generated_features = torch_condition_features(fake_batch, frequency_tensors)
            generated_features_norm = (generated_features - feature_mean) / feature_std
            feature_loss = F.mse_loss(generated_features_norm, condition_batch)
            g_loss = g_adv_loss + wgan_cfg.feature_loss_weight * feature_loss

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            g_adv_sum += float(g_adv_loss.detach().cpu())
            g_feature_sum += float(feature_loss.detach().cpu())
            g_total_sum += float(g_loss.detach().cpu())
            g_updates += 1

        row = {
            "轮次": epoch,
            "判别器损失": d_loss_sum / max(d_updates, 1),
            "生成器对抗损失": g_adv_sum / max(g_updates, 1),
            "生成器特征损失": g_feature_sum / max(g_updates, 1),
            "生成器总损失": g_total_sum / max(g_updates, 1),
            "Wasserstein估计": wasserstein_sum / max(d_updates, 1),
            "梯度惩罚": gp_sum / max(d_updates, 1),
            "判别器更新次数": d_updates,
            "生成器更新次数": g_updates,
        }
        history.append(row)

        objective = float(row["生成器总损失"])
        if objective < best_objective:
            best_objective = objective
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "wgan_config": asdict(wgan_cfg),
                    "feature_columns": FEATURE_COLUMNS,
                    "window_length": cfg.window_length,
                    "epoch": epoch,
                    "best_objective": best_objective,
                },
                dirs["model"] / "feature_wgan_gp_x2_best.pt",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == wgan_cfg.epochs:
            save_training_snapshot(
                history,
                generator,
                critic,
                opt_g,
                opt_d,
                wgan_cfg,
                cfg,
                dirs,
                epoch,
                best_objective,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == wgan_cfg.epochs:
            print(
                f"轮次 {epoch:03d}/{wgan_cfg.epochs} | "
                f"D损失 {row['判别器损失']:.5f} | "
                f"G对抗 {row['生成器对抗损失']:.5f} | "
                f"G特征 {row['生成器特征损失']:.5f} | "
                f"GP {row['梯度惩罚']:.5f}"
            )

    checkpoint = torch.load(dirs["model"] / "feature_wgan_gp_x2_best.pt", map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    history_df = pd.DataFrame(history)
    history_df.to_csv(dirs["model"] / "特征约束WGAN-GP训练历史.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "wgan_config": asdict(wgan_cfg),
            "feature_columns": FEATURE_COLUMNS,
            "window_length": cfg.window_length,
            "history": history,
        },
        dirs["model"] / "feature_wgan_gp_x2_final.pt",
    )
    return generator, critic, history_df


def generate_samples(
    generator: FeatureConditionedGenerator,
    condition_pool: np.ndarray,
    count: int,
    wgan_cfg: FeatureWGANConfig,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    generator.eval()
    rng = np.random.default_rng(seed)
    selected = rng.choice(condition_pool.shape[0], size=count, replace=True)
    selected_conditions = condition_pool[selected]
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, count, wgan_cfg.batch_size):
            current = min(wgan_cfg.batch_size, count - start)
            condition = torch.from_numpy(selected_conditions[start : start + current]).float().to(device)
            z = torch.randn(current, wgan_cfg.latent_dim, device=device)
            generated = generator(z, condition).squeeze(1).cpu().numpy()
            outputs.append(generated)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def plot_training_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].plot(history["轮次"], history["判别器损失"], label="判别器损失", linewidth=1.2)
    axes[0].plot(history["轮次"], history["生成器总损失"], label="生成器总损失", linewidth=1.2)
    axes[0].set_title("特征约束 WGAN-GP 损失变化")
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("损失")
    axes[0].grid(alpha=0.22)
    axes[0].legend()

    axes[1].plot(history["轮次"], history["生成器对抗损失"], label="对抗损失", linewidth=1.2)
    axes[1].plot(history["轮次"], history["生成器特征损失"], label="特征损失", linewidth=1.2)
    axes[1].set_title("生成器损失组成")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("损失")
    axes[1].grid(alpha=0.22)
    axes[1].legend()

    axes[2].plot(history["轮次"], history["梯度惩罚"], color="#F58518", linewidth=1.2)
    axes[2].set_title("梯度惩罚项")
    axes[2].set_xlabel("训练轮次")
    axes[2].set_ylabel("GP")
    axes[2].grid(alpha=0.22)
    fig.suptitle("特征约束 WGAN-GP 训练过程")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_waveform_comparison(real_windows: np.ndarray, generated_windows: np.ndarray, sampling_frequency: int, path: Path) -> None:
    time_ms = np.arange(real_windows.shape[1]) / sampling_frequency * 1000
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(time_ms, real_windows[i], color="#4C78A8", linewidth=0.8, label="真实样本")
        ax.plot(time_ms, generated_windows[i], color="#F58518", linewidth=0.8, alpha=0.85, label="特征约束 WGAN-GP")
        ax.set_ylabel("幅值")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("时间 / ms")
    fig.suptitle("特征约束 WGAN-GP 生成样本与真实样本波形对比")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mean_spectrum_comparison(real_windows: np.ndarray, generated_windows: np.ndarray, sampling_frequency: int, path: Path) -> None:
    freqs = rfftfreq(real_windows.shape[1], d=1.0 / sampling_frequency)
    real_amp = np.abs(rfft(real_windows - real_windows.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
    gen_amp = np.abs(rfft(generated_windows - generated_windows.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(freqs, real_amp, color="#222222", linewidth=1.3, label="真实样本")
    ax.plot(freqs, gen_amp, color="#F58518", linewidth=1.0, alpha=0.9, label="特征约束 WGAN-GP")
    ax.set_title("特征约束 WGAN-GP 生成样本与真实样本平均频谱对比")
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
    for ax, col, title in zip(axes, metric_cols, titles):
        ax.bar(summary["方法"], summary[col], color="#F58518")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=10)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("特征约束 WGAN-GP 评价指标汇总")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def evaluate_generated(real_windows: np.ndarray, generated_windows: np.ndarray, cfg, dirs: dict[str, Path]) -> pd.DataFrame:
    method_name = "特征约束WGAN-GP"
    real_features = compute_features(real_windows, cfg.sampling_frequency, cfg.spindle_speed_rpm)
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
        cfg.seed + 400,
    )
    summary = pd.DataFrame([{"方法": method_name, "生成样本数量": generated_windows.shape[0], **metrics}])
    detail.to_csv(dirs["features"] / "特征约束WGAN-GP_单特征误差明细.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(dirs["features"] / "特征约束WGAN-GP_方法指标汇总.csv", index=False, encoding="utf-8-sig")
    plot_pca_comparison(
        real_features,
        generated_features.drop(columns=["样本编号"]),
        method_name,
        dirs["figures"] / "PCA特征分布_特征约束WGAN-GP.png",
    )
    return summary


def write_report(
    output_dir: Path,
    cfg,
    wgan_cfg: FeatureWGANConfig,
    device: torch.device,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    real_count: int,
) -> None:
    final_epoch = history.iloc[-1]
    metric_row = summary.iloc[0]
    report = f"""# 实验4：特征约束 WGAN-GP 振动信号数据扩充

## 实验目的

本实验在普通 WGAN-GP 的基础上加入振动特征条件和特征损失。目标是让生成样本不仅在波形分布上接近真实窗口，还在 RMS、峰值、峰峰值、峭度、频率重心、频带能量占比等公共评价特征上接近真实样本。

## 数据与模型参数

- 输入数据：实验0保存的 `{cfg.channel}` 通道标准化窗口；
- 真实窗口数量：{real_count}；
- 单个窗口长度：{cfg.window_length} 个采样点；
- 采样频率：{cfg.sampling_frequency} Hz；
- 主轴转速：{cfg.spindle_speed_rpm} rpm；
- 随机噪声维度：{wgan_cfg.latent_dim}；
- 条件特征数量：{len(FEATURE_COLUMNS)}；
- 训练轮数：{wgan_cfg.epochs}；
- 批大小：{wgan_cfg.batch_size}；
- 判别器每轮更新次数：{wgan_cfg.critic_steps}；
- 梯度惩罚权重：{wgan_cfg.gradient_penalty_weight}；
- 特征损失权重：{wgan_cfg.feature_loss_weight}；
- 训练设备：{device}。

## 方法说明

本实验使用的条件特征包括 RMS、峰值、峰峰值、偏度、峭度、峰值因子、频率重心、0-1 kHz、1-3 kHz、3-5 kHz 频带能量占比，以及一倍频、二倍频、三四倍频附近能量占比。训练时，生成器根据目标特征生成振动窗口，判别器判断“窗口和目标特征”这组数据是否真实；同时，代码会重新计算生成窗口的可微特征，并与目标特征做均方误差约束。

## 本次训练结果

- 最后一轮判别器损失：{final_epoch["判别器损失"]:.6g}；
- 最后一轮生成器对抗损失：{final_epoch["生成器对抗损失"]:.6g}；
- 最后一轮生成器特征损失：{final_epoch["生成器特征损失"]:.6g}；
- 最后一轮生成器总损失：{final_epoch["生成器总损失"]:.6g}；
- 最后一轮 Wasserstein 距离估计：{final_epoch["Wasserstein估计"]:.6g}；
- 最后一轮梯度惩罚项：{final_epoch["梯度惩罚"]:.6g}。

## 评价指标

| 方法 | 生成样本数量 | 平均均值相对误差_% | 平均JS散度 | MMD距离 |
| --- | --- | --- | --- | --- |
| {metric_row["方法"]} | {int(metric_row["生成样本数量"])} | {metric_row["平均均值相对误差_%"]:.6g} | {metric_row["平均JS散度"]:.6g} | {metric_row["MMD距离"]:.6g} |

## 输出文件说明

- `01_模型与训练记录/`：特征约束 WGAN-GP 最优模型、最终模型、训练历史和条件特征标准化参数；
- `02_生成数据/`：生成窗口数据，包含 `.npy` 和 `.csv`；
- `03_特征表/`：生成样本特征、单特征误差明细和方法指标汇总；
- `04_评价图表/`：训练曲线、波形对比、平均频谱、PCA 分布和指标汇总图。
"""
    (output_dir / "05_实验说明" / "实验4说明.md").write_text(report, encoding="utf-8")


def run(config_path: Path, args: argparse.Namespace) -> None:
    cfg = load_config(config_path)
    wgan_cfg = load_feature_wgan_config(config_path, args)
    set_seed(cfg.seed)
    setup_chinese_font()

    processed_dir = cfg.output_dir / "01_预处理数据"
    real_path = processed_dir / "X-2_窗口数据_原始.npy"
    standardized_path = processed_dir / "X-2_窗口数据_标准化.npy"
    stats_path = processed_dir / "标准化参数.json"
    if not real_path.exists() or not standardized_path.exists() or not stats_path.exists():
        raise FileNotFoundError("没有找到实验0的预处理结果，请先运行实验0。")

    output_dir = PROJECT_ROOT / "实验4_特征约束WGAN_GP"
    dirs = ensure_dirs(output_dir)
    real_windows = np.load(real_path).astype(np.float32)
    windows_standardized = np.load(standardized_path).astype(np.float32)
    mean, std = load_standardize_stats(stats_path)
    conditions_raw, conditions_normalized, condition_stats = build_condition_table(
        windows_standardized,
        cfg.sampling_frequency,
        cfg.spindle_speed_rpm,
    )
    generated_count = wgan_cfg.generated_count or real_windows.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备：{device}")
    print(f"训练样本形状：{windows_standardized.shape}")
    print(f"条件特征数量：{conditions_normalized.shape[1]}")
    print(f"计划生成样本数量：{generated_count}")

    with (dirs["model"] / "条件特征标准化参数.json").open("w", encoding="utf-8") as f:
        json.dump(condition_stats, f, ensure_ascii=False, indent=2)

    generator, _, history = train_feature_wgan_gp(
        windows_standardized,
        conditions_raw,
        conditions_normalized,
        cfg,
        wgan_cfg,
        dirs,
        device,
    )
    plot_training_history(history, dirs["figures"] / "特征约束WGAN-GP训练曲线.png")

    generated_std = generate_samples(generator, conditions_normalized, generated_count, wgan_cfg, cfg.seed + 4000, device)
    generated_std = clamp_generated(
        generated_std,
        windows_standardized,
        wgan_cfg.clamp_percentile_low,
        wgan_cfg.clamp_percentile_high,
    )
    generated_windows = restore_to_original_scale(generated_std, mean, std, cfg.remove_window_mean)
    np.save(dirs["generated"] / "特征约束WGAN-GP_窗口数据.npy", generated_windows)
    save_windows_csv(dirs["generated"] / "特征约束WGAN-GP_窗口数据.csv", generated_windows)

    summary = evaluate_generated(real_windows, generated_windows, cfg, dirs)
    plot_waveform_comparison(real_windows, generated_windows, cfg.sampling_frequency, dirs["figures"] / "特征约束WGAN-GP_波形对比.png")
    plot_mean_spectrum_comparison(real_windows, generated_windows, cfg.sampling_frequency, dirs["figures"] / "特征约束WGAN-GP_平均频谱对比.png")
    plot_metric_summary(summary, dirs["figures"] / "特征约束WGAN-GP_指标汇总.png")

    run_info = {
        "实验名称": "实验4：特征约束 WGAN-GP",
        "输入真实窗口": str(real_path),
        "输入标准化窗口": str(standardized_path),
        "输出目录": str(output_dir),
        "真实窗口数量": int(real_windows.shape[0]),
        "生成窗口数量": int(generated_count),
        "窗口长度": int(real_windows.shape[1]),
        "设备": str(device),
        "随机种子": int(cfg.seed),
        "条件特征名称": FEATURE_COLUMNS,
        "特征约束WGAN-GP参数": asdict(wgan_cfg),
    }
    with (dirs["report"] / "实验4运行信息.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    write_report(output_dir, cfg, wgan_cfg, device, history, summary, real_windows.shape[0])

    print("实验4完成")
    print(f"输出目录：{output_dir}")
    print("特征约束 WGAN-GP 方法指标汇总：")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验4：特征约束 WGAN-GP 振动信号数据扩充")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="配置文件路径，默认读取项目根目录 config.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖配置中的批大小")
    parser.add_argument("--generated-count", type=int, default=None, help="覆盖生成样本数量")
    parser.add_argument("--critic-steps", type=int, default=None, help="覆盖每轮判别器更新次数")
    parser.add_argument("--feature-loss-weight", type=float, default=None, help="覆盖特征损失权重")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args.config.resolve(), cli_args)
