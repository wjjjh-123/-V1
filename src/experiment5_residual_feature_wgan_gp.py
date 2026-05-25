"""实验5：残差特征约束 WGAN-GP 振动信号数据扩充。

本实验是在“特征约束 WGAN-GP”基础上加入残差结构：
1. 随机选择真实健康振动窗口作为基础窗口；
2. 生成器根据“基础窗口 + 随机噪声 + 目标特征”生成一个残差扰动；
3. 最终扩充样本 = 基础窗口 + 残差扰动；
4. 训练时同时使用 WGAN-GP 对抗损失、振动特征约束损失和残差幅值约束。

这种结构适合当前短周期健康振动信号扩充：它尽量保留真实波形主体，只学习合理的小变化。
"""

from __future__ import annotations

import argparse
import json
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
from src.experiment4_feature_wgan_gp import (
    FEATURE_COLUMNS,
    FeatureConditionedCritic,
    build_condition_table,
    gradient_penalty,
    load_feature_wgan_config,
    make_frequency_tensors,
    set_seed,
    torch_condition_features,
)


@dataclass(frozen=True)
class ResidualFeatureWGANConfig:
    latent_dim: int = 128
    epochs: int = 120
    batch_size: int = 64
    generator_learning_rate: float = 0.0001
    discriminator_learning_rate: float = 0.0001
    critic_steps: int = 3
    gradient_penalty_weight: float = 10.0
    feature_loss_weight: float = 8.0
    residual_scale: float = 0.35
    residual_l2_weight: float = 0.02
    generated_count: int | None = None
    clamp_percentile_low: float = 0.2
    clamp_percentile_high: float = 99.8


class ResidualFeatureGenerator(nn.Module):
    """残差特征约束生成器：基础窗口 + 随机噪声 + 条件特征 -> 残差扰动。"""

    def __init__(self, latent_dim: int, condition_dim: int, window_length: int) -> None:
        super().__init__()
        if window_length != 2048:
            raise ValueError("当前网络按 2048 点窗口设计，如需其他长度请同步调整网络结构。")
        self.base_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim + 128, 256 * 128),
            nn.BatchNorm1d(256 * 128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
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
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        base_code = self.base_encoder(base)
        x = torch.cat([z, condition, base_code], dim=1)
        x = self.fc(x).view(z.shape[0], 256, 128)
        return self.decoder(x)


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


def load_residual_feature_config(config_path: Path, args: argparse.Namespace) -> ResidualFeatureWGANConfig:
    base_cfg = load_feature_wgan_config(config_path, args)
    cfg = ResidualFeatureWGANConfig(**asdict(base_cfg))
    if args.residual_scale is not None:
        cfg = ResidualFeatureWGANConfig(**{**asdict(cfg), "residual_scale": args.residual_scale})
    if args.residual_l2_weight is not None:
        cfg = ResidualFeatureWGANConfig(**{**asdict(cfg), "residual_l2_weight": args.residual_l2_weight})
    if args.clamp_percentile_low is not None:
        cfg = ResidualFeatureWGANConfig(**{**asdict(cfg), "clamp_percentile_low": args.clamp_percentile_low})
    if args.clamp_percentile_high is not None:
        cfg = ResidualFeatureWGANConfig(**{**asdict(cfg), "clamp_percentile_high": args.clamp_percentile_high})
    return cfg


def train_residual_feature_wgan_gp(
    windows_standardized: np.ndarray,
    conditions_raw: np.ndarray,
    conditions_normalized: np.ndarray,
    cfg,
    wgan_cfg: ResidualFeatureWGANConfig,
    dirs: dict[str, Path],
    device: torch.device,
) -> tuple[ResidualFeatureGenerator, FeatureConditionedCritic, pd.DataFrame]:
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
    generator = ResidualFeatureGenerator(wgan_cfg.latent_dim, condition_dim, cfg.window_length).to(device)
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
        d_loss_sum = g_adv_sum = g_feature_sum = g_residual_sum = g_total_sum = wasserstein_sum = gp_sum = 0.0
        d_updates = g_updates = 0

        for real_batch, condition_batch, _ in loader:
            real_batch = real_batch.to(device)
            condition_batch = condition_batch.to(device)
            batch_size = real_batch.shape[0]

            for _ in range(wgan_cfg.critic_steps):
                z = torch.randn(batch_size, wgan_cfg.latent_dim, device=device)
                residual = generator(z, condition_batch, real_batch).detach()
                fake_batch = real_batch + wgan_cfg.residual_scale * residual
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
            residual = generator(z, condition_batch, real_batch)
            fake_batch = real_batch + wgan_cfg.residual_scale * residual
            g_adv_loss = -critic(fake_batch, condition_batch).mean()
            generated_features = torch_condition_features(fake_batch, frequency_tensors)
            generated_features_norm = (generated_features - feature_mean) / feature_std
            feature_loss = F.mse_loss(generated_features_norm, condition_batch)
            residual_loss = residual.pow(2).mean()
            g_loss = (
                g_adv_loss
                + wgan_cfg.feature_loss_weight * feature_loss
                + wgan_cfg.residual_l2_weight * residual_loss
            )

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            g_adv_sum += float(g_adv_loss.detach().cpu())
            g_feature_sum += float(feature_loss.detach().cpu())
            g_residual_sum += float(residual_loss.detach().cpu())
            g_total_sum += float(g_loss.detach().cpu())
            g_updates += 1

        row = {
            "轮次": epoch,
            "判别器损失": d_loss_sum / max(d_updates, 1),
            "生成器对抗损失": g_adv_sum / max(g_updates, 1),
            "生成器特征损失": g_feature_sum / max(g_updates, 1),
            "生成器残差约束损失": g_residual_sum / max(g_updates, 1),
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
                dirs["model"] / "residual_feature_wgan_gp_x2_best.pt",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == wgan_cfg.epochs:
            print(
                f"轮次 {epoch:03d}/{wgan_cfg.epochs} | "
                f"D损失 {row['判别器损失']:.5f} | "
                f"G对抗 {row['生成器对抗损失']:.5f} | "
                f"G特征 {row['生成器特征损失']:.5f} | "
                f"G残差 {row['生成器残差约束损失']:.5f} | "
                f"GP {row['梯度惩罚']:.5f}"
            )

    checkpoint = torch.load(dirs["model"] / "residual_feature_wgan_gp_x2_best.pt", map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    history_df = pd.DataFrame(history)
    history_df.to_csv(dirs["model"] / "残差特征约束WGAN-GP训练历史.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "wgan_config": asdict(wgan_cfg),
            "feature_columns": FEATURE_COLUMNS,
            "window_length": cfg.window_length,
            "history": history,
        },
        dirs["model"] / "residual_feature_wgan_gp_x2_final.pt",
    )
    return generator, critic, history_df


def generate_samples(
    generator: ResidualFeatureGenerator,
    windows_standardized: np.ndarray,
    condition_pool: np.ndarray,
    count: int,
    wgan_cfg: ResidualFeatureWGANConfig,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    generator.eval()
    rng = np.random.default_rng(seed)
    replace = count > windows_standardized.shape[0]
    selected = rng.choice(windows_standardized.shape[0], size=count, replace=replace)
    selected_windows = windows_standardized[selected]
    selected_conditions = condition_pool[selected]
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, count, wgan_cfg.batch_size):
            current = min(wgan_cfg.batch_size, count - start)
            base = torch.from_numpy(selected_windows[start : start + current]).float().unsqueeze(1).to(device)
            condition = torch.from_numpy(selected_conditions[start : start + current]).float().to(device)
            z = torch.randn(current, wgan_cfg.latent_dim, device=device)
            residual = generator(z, condition, base)
            generated = base + wgan_cfg.residual_scale * residual
            outputs.append(generated.squeeze(1).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def plot_training_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].plot(history["轮次"], history["判别器损失"], label="判别器损失", linewidth=1.2)
    axes[0].plot(history["轮次"], history["生成器总损失"], label="生成器总损失", linewidth=1.2)
    axes[0].set_title("残差特征约束 WGAN-GP 损失变化")
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("损失")
    axes[0].grid(alpha=0.22)
    axes[0].legend()

    axes[1].plot(history["轮次"], history["生成器对抗损失"], label="对抗损失", linewidth=1.2)
    axes[1].plot(history["轮次"], history["生成器特征损失"], label="特征损失", linewidth=1.2)
    axes[1].plot(history["轮次"], history["生成器残差约束损失"], label="残差约束损失", linewidth=1.2)
    axes[1].set_title("生成器损失组成")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("损失")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)

    axes[2].plot(history["轮次"], history["梯度惩罚"], color="#F58518", linewidth=1.2)
    axes[2].set_title("梯度惩罚项")
    axes[2].set_xlabel("训练轮次")
    axes[2].set_ylabel("GP")
    axes[2].grid(alpha=0.22)
    fig.suptitle("残差特征约束 WGAN-GP 训练过程")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_waveform_comparison(real_windows: np.ndarray, generated_windows: np.ndarray, sampling_frequency: int, path: Path) -> None:
    time_ms = np.arange(real_windows.shape[1]) / sampling_frequency * 1000
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(time_ms, real_windows[i], color="#4C78A8", linewidth=0.8, label="真实样本")
        ax.plot(time_ms, generated_windows[i], color="#F58518", linewidth=0.8, alpha=0.85, label="残差特征约束 WGAN-GP")
        ax.set_ylabel("幅值")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("时间 / ms")
    fig.suptitle("残差特征约束 WGAN-GP 生成样本与真实样本波形对比")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mean_spectrum_comparison(real_windows: np.ndarray, generated_windows: np.ndarray, sampling_frequency: int, path: Path) -> None:
    freqs = rfftfreq(real_windows.shape[1], d=1.0 / sampling_frequency)
    real_amp = np.abs(rfft(real_windows - real_windows.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
    gen_amp = np.abs(rfft(generated_windows - generated_windows.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(freqs, real_amp, color="#222222", linewidth=1.3, label="真实样本")
    ax.plot(freqs, gen_amp, color="#F58518", linewidth=1.0, alpha=0.9, label="残差特征约束 WGAN-GP")
    ax.set_title("残差特征约束 WGAN-GP 生成样本与真实样本平均频谱对比")
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
    fig.suptitle("残差特征约束 WGAN-GP 评价指标汇总")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def evaluate_generated(real_windows: np.ndarray, generated_windows: np.ndarray, cfg, dirs: dict[str, Path]) -> pd.DataFrame:
    method_name = "残差特征约束WGAN-GP"
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
        cfg.seed + 500,
    )
    summary = pd.DataFrame([{"方法": method_name, "生成样本数量": generated_windows.shape[0], **metrics}])
    detail.to_csv(dirs["features"] / "残差特征约束WGAN-GP_单特征误差明细.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(dirs["features"] / "残差特征约束WGAN-GP_方法指标汇总.csv", index=False, encoding="utf-8-sig")
    plot_pca_comparison(
        real_features,
        generated_features.drop(columns=["样本编号"]),
        method_name,
        dirs["figures"] / "PCA特征分布_残差特征约束WGAN-GP.png",
    )
    return summary


def write_report(
    output_dir: Path,
    cfg,
    wgan_cfg: ResidualFeatureWGANConfig,
    device: torch.device,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    real_count: int,
) -> None:
    final_epoch = history.iloc[-1]
    metric_row = summary.iloc[0]
    report = f"""# 实验5：残差特征约束 WGAN-GP 振动信号数据扩充

## 实验目的

本实验在特征约束 WGAN-GP 的基础上加入残差结构。生成器不直接从零生成完整振动窗口，而是在真实健康窗口上叠加一个较小的残差扰动，以增强生成样本对原始振动形态的保持能力。

## 数据与模型参数

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
- 残差缩放系数：{wgan_cfg.residual_scale}；
- 残差约束权重：{wgan_cfg.residual_l2_weight}；
- 训练设备：{device}。

## 本次训练结果

- 最后一轮判别器损失：{final_epoch["判别器损失"]:.6g}；
- 最后一轮生成器对抗损失：{final_epoch["生成器对抗损失"]:.6g}；
- 最后一轮生成器特征损失：{final_epoch["生成器特征损失"]:.6g}；
- 最后一轮生成器残差约束损失：{final_epoch["生成器残差约束损失"]:.6g}；
- 最后一轮生成器总损失：{final_epoch["生成器总损失"]:.6g}；
- 最后一轮 Wasserstein 距离估计：{final_epoch["Wasserstein估计"]:.6g}；
- 最后一轮梯度惩罚项：{final_epoch["梯度惩罚"]:.6g}。

## 评价指标

| 方法 | 生成样本数量 | 平均均值相对误差_% | 平均JS散度 | MMD距离 |
| --- | --- | --- | --- | --- |
| {metric_row["方法"]} | {int(metric_row["生成样本数量"])} | {metric_row["平均均值相对误差_%"]:.6g} | {metric_row["平均JS散度"]:.6g} | {metric_row["MMD距离"]:.6g} |

说明：MMD 是最大均值差异，越小表示生成样本和真实样本在整体特征分布上越接近。本次结果只能说明当前配置下的实验输出，论文正式结果建议固定随机种子后统一重跑并与 VAE、普通 WGAN-GP、特征约束 WGAN-GP 进行对比。
"""
    (output_dir / "05_实验说明" / "实验5说明.md").write_text(report, encoding="utf-8")


def run(config_path: Path, args: argparse.Namespace) -> None:
    cfg = load_config(config_path)
    wgan_cfg = load_residual_feature_config(config_path, args)
    set_seed(cfg.seed)
    setup_chinese_font()

    processed_dir = cfg.output_dir / "01_预处理数据"
    real_path = processed_dir / "X-2_窗口数据_原始.npy"
    standardized_path = processed_dir / "X-2_窗口数据_标准化.npy"
    stats_path = processed_dir / "标准化参数.json"
    if not real_path.exists() or not standardized_path.exists() or not stats_path.exists():
        raise FileNotFoundError("没有找到实验0的预处理结果，请先运行实验0。")

    output_dir = PROJECT_ROOT / args.output_dir_name
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
    print(f"残差缩放系数：{wgan_cfg.residual_scale}")

    with (dirs["model"] / "条件特征标准化参数.json").open("w", encoding="utf-8") as f:
        json.dump(condition_stats, f, ensure_ascii=False, indent=2)

    generator, _, history = train_residual_feature_wgan_gp(
        windows_standardized,
        conditions_raw,
        conditions_normalized,
        cfg,
        wgan_cfg,
        dirs,
        device,
    )
    plot_training_history(history, dirs["figures"] / "残差特征约束WGAN-GP训练曲线.png")

    generated_std = generate_samples(
        generator,
        windows_standardized,
        conditions_normalized,
        generated_count,
        wgan_cfg,
        cfg.seed + 5000,
        device,
    )
    if not args.disable_clamp:
        generated_std = clamp_generated(
            generated_std,
            windows_standardized,
            wgan_cfg.clamp_percentile_low,
            wgan_cfg.clamp_percentile_high,
        )
    generated_windows = restore_to_original_scale(generated_std, mean, std, cfg.remove_window_mean)
    np.save(dirs["generated"] / "残差特征约束WGAN-GP_窗口数据.npy", generated_windows)
    save_windows_csv(dirs["generated"] / "残差特征约束WGAN-GP_窗口数据.csv", generated_windows)

    summary = evaluate_generated(real_windows, generated_windows, cfg, dirs)
    plot_waveform_comparison(real_windows, generated_windows, cfg.sampling_frequency, dirs["figures"] / "残差特征约束WGAN-GP_波形对比.png")
    plot_mean_spectrum_comparison(real_windows, generated_windows, cfg.sampling_frequency, dirs["figures"] / "残差特征约束WGAN-GP_平均频谱对比.png")
    plot_metric_summary(summary, dirs["figures"] / "残差特征约束WGAN-GP_指标汇总.png")

    run_info = {
        "实验名称": "实验5：残差特征约束 WGAN-GP",
        "输入真实窗口": str(real_path),
        "输入标准化窗口": str(standardized_path),
        "输出目录": str(output_dir),
        "真实窗口数量": int(real_windows.shape[0]),
        "生成窗口数量": int(generated_count),
        "窗口长度": int(real_windows.shape[1]),
        "设备": str(device),
        "随机种子": int(cfg.seed),
        "条件特征名称": FEATURE_COLUMNS,
        "残差特征约束WGAN-GP参数": asdict(wgan_cfg),
        "基础窗口抽样方式": "无放回抽样" if generated_count <= real_windows.shape[0] else "有放回抽样",
        "是否执行分位数截断": not args.disable_clamp,
    }
    with (dirs["report"] / "实验5运行信息.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    write_report(output_dir, cfg, wgan_cfg, device, history, summary, real_windows.shape[0])

    print("实验5完成")
    print(f"输出目录：{output_dir}")
    print("残差特征约束 WGAN-GP 方法指标汇总：")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验5：残差特征约束 WGAN-GP 振动信号数据扩充")
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
    parser.add_argument("--residual-scale", type=float, default=None, help="覆盖残差缩放系数")
    parser.add_argument("--residual-l2-weight", type=float, default=None, help="覆盖残差幅值约束权重")
    parser.add_argument("--clamp-percentile-low", type=float, default=None, help="覆盖截断下分位数")
    parser.add_argument("--clamp-percentile-high", type=float, default=None, help="覆盖截断上分位数")
    parser.add_argument("--disable-clamp", action="store_true", help="不执行分位数截断，保留生成样本极值")
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="实验5_残差特征约束WGAN_GP",
        help="输出目录名称，默认写入实验5_残差特征约束WGAN_GP",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args.config.resolve(), cli_args)
