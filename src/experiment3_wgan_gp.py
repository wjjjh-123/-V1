"""实验3：普通 WGAN-GP 振动信号数据扩充。

本脚本训练无条件 WGAN-GP（带梯度惩罚的 Wasserstein 生成对抗网络）。
它只从随机噪声生成振动窗口，不加入特征约束，也不使用残差扩充结构，
因此适合作为后续“特征约束 WGAN-GP”和“残差特征约束 WGAN-GP”的基础对照实验。
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


@dataclass(frozen=True)
class WGANConfig:
    latent_dim: int = 128
    epochs: int = 120
    batch_size: int = 64
    generator_learning_rate: float = 0.0001
    discriminator_learning_rate: float = 0.0001
    critic_steps: int = 3
    gradient_penalty_weight: float = 10.0
    generated_count: int | None = None
    clamp_percentile_low: float = 0.2
    clamp_percentile_high: float = 99.8


class WGANGenerator(nn.Module):
    """普通 WGAN-GP 生成器：随机噪声 -> 2048 点振动窗口。"""

    def __init__(self, latent_dim: int, window_length: int) -> None:
        super().__init__()
        if window_length != 2048:
            raise ValueError("当前 WGAN-GP 网络按 2048 点窗口设计，如需其他长度请同步调整网络结构。")
        self.latent_dim = latent_dim
        self.window_length = window_length
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 128),
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], 256, 128)
        return self.net(x)


class WGANCritic(nn.Module):
    """普通 WGAN-GP 判别器，也常称为 critic（评论器）。"""

    def __init__(self, window_length: int) -> None:
        super().__init__()
        if window_length != 2048:
            raise ValueError("当前 WGAN-GP 网络按 2048 点窗口设计，如需其他长度请同步调整网络结构。")
        self.net = nn.Sequential(
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
        self.fc = nn.Linear(256 * (window_length // 16), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x).flatten(start_dim=1)
        return self.fc(features).view(-1)


def read_raw_config(config_path: Path) -> dict[str, object]:
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        return {}


def replace_config(cfg: WGANConfig, **kwargs: object) -> WGANConfig:
    data = asdict(cfg)
    data.update(kwargs)
    return WGANConfig(**data)


def load_wgan_config(config_path: Path, args: argparse.Namespace) -> WGANConfig:
    raw = read_raw_config(config_path)
    exp_cfg = raw.get("experiment3", {}) if isinstance(raw, dict) else {}
    cfg = WGANConfig(
        latent_dim=int(exp_cfg.get("latent_dim", 128)),
        epochs=int(exp_cfg.get("epochs", 120)),
        batch_size=int(exp_cfg.get("batch_size", 64)),
        generator_learning_rate=float(exp_cfg.get("generator_learning_rate", 0.0001)),
        discriminator_learning_rate=float(exp_cfg.get("discriminator_learning_rate", 0.0001)),
        critic_steps=int(exp_cfg.get("critic_steps", 3)),
        gradient_penalty_weight=float(exp_cfg.get("gradient_penalty_weight", 10.0)),
        generated_count=exp_cfg.get("generated_count"),
        clamp_percentile_low=float(exp_cfg.get("clamp_percentile_low", 0.2)),
        clamp_percentile_high=float(exp_cfg.get("clamp_percentile_high", 99.8)),
    )
    if args.epochs is not None:
        cfg = replace_config(cfg, epochs=args.epochs)
    if args.batch_size is not None:
        cfg = replace_config(cfg, batch_size=args.batch_size)
    if args.generated_count is not None:
        cfg = replace_config(cfg, generated_count=args.generated_count)
    if args.critic_steps is not None:
        cfg = replace_config(cfg, critic_steps=args.critic_steps)
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


def gradient_penalty(
    critic: WGANCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real.shape[0]
    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated)
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


def train_wgan_gp(
    windows_standardized: np.ndarray,
    cfg,
    wgan_cfg: WGANConfig,
    dirs: dict[str, Path],
    device: torch.device,
) -> tuple[WGANGenerator, WGANCritic, pd.DataFrame]:
    tensor = torch.from_numpy(windows_standardized).float().unsqueeze(1)
    loader = DataLoader(TensorDataset(tensor), batch_size=wgan_cfg.batch_size, shuffle=True, drop_last=True)
    generator = WGANGenerator(wgan_cfg.latent_dim, cfg.window_length).to(device)
    critic = WGANCritic(cfg.window_length).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=wgan_cfg.generator_learning_rate, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=wgan_cfg.discriminator_learning_rate, betas=(0.0, 0.9))
    history: list[dict[str, float | int]] = []

    best_generator_loss = float("inf")
    step = 0
    for epoch in range(1, wgan_cfg.epochs + 1):
        d_loss_sum = g_loss_sum = wasserstein_sum = gp_sum = 0.0
        d_updates = g_updates = 0

        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.shape[0]

            for _ in range(wgan_cfg.critic_steps):
                z = torch.randn(batch_size, wgan_cfg.latent_dim, device=device)
                fake_batch = generator(z).detach()
                real_score = critic(real_batch)
                fake_score = critic(fake_batch)
                wasserstein = real_score.mean() - fake_score.mean()
                gp = gradient_penalty(critic, real_batch, fake_batch, device)
                d_loss = -wasserstein + wgan_cfg.gradient_penalty_weight * gp

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

                d_loss_sum += float(d_loss.detach().cpu())
                wasserstein_sum += float(wasserstein.detach().cpu())
                gp_sum += float(gp.detach().cpu())
                d_updates += 1
                step += 1

            z = torch.randn(batch_size, wgan_cfg.latent_dim, device=device)
            fake_batch = generator(z)
            g_loss = -critic(fake_batch).mean()
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            g_loss_sum += float(g_loss.detach().cpu())
            g_updates += 1

        row = {
            "轮次": epoch,
            "判别器损失": d_loss_sum / max(d_updates, 1),
            "生成器损失": g_loss_sum / max(g_updates, 1),
            "Wasserstein估计": wasserstein_sum / max(d_updates, 1),
            "梯度惩罚": gp_sum / max(d_updates, 1),
            "判别器更新次数": d_updates,
            "生成器更新次数": g_updates,
        }
        history.append(row)

        if row["生成器损失"] < best_generator_loss:
            best_generator_loss = float(row["生成器损失"])
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "wgan_config": asdict(wgan_cfg),
                    "window_length": cfg.window_length,
                    "epoch": epoch,
                    "generator_loss": best_generator_loss,
                },
                dirs["model"] / "wgan_gp_x2_best.pt",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == wgan_cfg.epochs:
            print(
                f"轮次 {epoch:03d}/{wgan_cfg.epochs} | "
                f"D损失 {row['判别器损失']:.5f} | G损失 {row['生成器损失']:.5f} | "
                f"W距离估计 {row['Wasserstein估计']:.5f} | GP {row['梯度惩罚']:.5f}"
            )

    checkpoint = torch.load(dirs["model"] / "wgan_gp_x2_best.pt", map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    history_df = pd.DataFrame(history)
    history_df.to_csv(dirs["model"] / "WGAN-GP训练历史.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "wgan_config": asdict(wgan_cfg),
            "window_length": cfg.window_length,
            "history": history,
        },
        dirs["model"] / "wgan_gp_x2_final.pt",
    )
    return generator, critic, history_df


def generate_samples(generator: WGANGenerator, count: int, wgan_cfg: WGANConfig, device: torch.device) -> np.ndarray:
    generator.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, count, wgan_cfg.batch_size):
            current = min(wgan_cfg.batch_size, count - start)
            z = torch.randn(current, wgan_cfg.latent_dim, device=device)
            generated = generator(z).squeeze(1).cpu().numpy()
            outputs.append(generated)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def clamp_generated(generated: np.ndarray, reference: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    low = float(np.percentile(reference, low_percentile))
    high = float(np.percentile(reference, high_percentile))
    return np.clip(generated, low, high).astype(np.float32)


def plot_training_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].plot(history["轮次"], history["判别器损失"], label="判别器损失", linewidth=1.2)
    axes[0].plot(history["轮次"], history["生成器损失"], label="生成器损失", linewidth=1.2)
    axes[0].set_title("WGAN-GP 损失变化")
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("损失")
    axes[0].grid(alpha=0.22)
    axes[0].legend()

    axes[1].plot(history["轮次"], history["Wasserstein估计"], color="#4C78A8", linewidth=1.2)
    axes[1].set_title("Wasserstein 距离估计")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("估计值")
    axes[1].grid(alpha=0.22)

    axes[2].plot(history["轮次"], history["梯度惩罚"], color="#F58518", linewidth=1.2)
    axes[2].set_title("梯度惩罚项")
    axes[2].set_xlabel("训练轮次")
    axes[2].set_ylabel("GP")
    axes[2].grid(alpha=0.22)
    fig.suptitle("普通 WGAN-GP 训练过程")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_waveform_comparison(real_windows: np.ndarray, generated_windows: np.ndarray, sampling_frequency: int, path: Path) -> None:
    time_ms = np.arange(real_windows.shape[1]) / sampling_frequency * 1000
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(time_ms, real_windows[i], color="#4C78A8", linewidth=0.8, label="真实样本")
        ax.plot(time_ms, generated_windows[i], color="#F58518", linewidth=0.8, alpha=0.85, label="普通 WGAN-GP")
        ax.set_ylabel("幅值")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("时间 / ms")
    fig.suptitle("普通 WGAN-GP 生成样本与真实样本波形对比")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mean_spectrum_comparison(real_windows: np.ndarray, generated_windows: np.ndarray, sampling_frequency: int, path: Path) -> None:
    freqs = rfftfreq(real_windows.shape[1], d=1.0 / sampling_frequency)
    real_amp = np.abs(rfft(real_windows - real_windows.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
    gen_amp = np.abs(rfft(generated_windows - generated_windows.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(freqs, real_amp, color="#222222", linewidth=1.3, label="真实样本")
    ax.plot(freqs, gen_amp, color="#F58518", linewidth=1.0, alpha=0.9, label="普通 WGAN-GP")
    ax.set_title("普通 WGAN-GP 生成样本与真实样本平均频谱对比")
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
    fig.suptitle("普通 WGAN-GP 评价指标汇总")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def evaluate_generated(real_windows: np.ndarray, generated_windows: np.ndarray, cfg, dirs: dict[str, Path]) -> pd.DataFrame:
    method_name = "普通WGAN-GP"
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
        cfg.seed + 300,
    )
    summary = pd.DataFrame([{"方法": method_name, "生成样本数量": generated_windows.shape[0], **metrics}])
    detail.to_csv(dirs["features"] / "普通WGAN-GP_单特征误差明细.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(dirs["features"] / "普通WGAN-GP_方法指标汇总.csv", index=False, encoding="utf-8-sig")
    plot_pca_comparison(
        real_features,
        generated_features.drop(columns=["样本编号"]),
        method_name,
        dirs["figures"] / "PCA特征分布_普通WGAN-GP.png",
    )
    return summary


def write_report(
    output_dir: Path,
    cfg,
    wgan_cfg: WGANConfig,
    device: torch.device,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    real_count: int,
) -> None:
    final_epoch = history.iloc[-1]
    metric_row = summary.iloc[0]
    report = f"""# 实验3：普通 WGAN-GP 振动信号数据扩充

## 实验目的

本实验训练普通 WGAN-GP（带梯度惩罚的 Wasserstein 生成对抗网络）生成主轴振动窗口。该模型只输入随机噪声，不使用振动特征作为条件，也不使用残差叠加，因此主要作为后续特征约束 WGAN-GP 和残差特征约束 WGAN-GP 的对照基线。

## 数据与模型参数

- 输入数据：实验0保存的 `{cfg.channel}` 通道标准化窗口；
- 真实窗口数量：{real_count}；
- 单个窗口长度：{cfg.window_length} 个采样点；
- 采样频率：{cfg.sampling_frequency} Hz；
- 主轴转速：{cfg.spindle_speed_rpm} rpm；
- 随机噪声维度：{wgan_cfg.latent_dim}；
- 训练轮数：{wgan_cfg.epochs}；
- 批大小：{wgan_cfg.batch_size}；
- 判别器每轮更新次数：{wgan_cfg.critic_steps}；
- 梯度惩罚权重：{wgan_cfg.gradient_penalty_weight}；
- 训练设备：{device}。

## 方法说明

WGAN-GP 使用 Wasserstein 距离衡量真实样本分布与生成样本分布的差异，并通过梯度惩罚约束判别器，使训练过程比普通 GAN 更稳定。这里的“普通”指模型没有额外输入 RMS、峰值、频带能量等振动特征，因此它主要学习整体波形分布，而不显式保证具体物理特征一致。

## 本次训练结果

- 最后一轮判别器损失：{final_epoch["判别器损失"]:.6g}；
- 最后一轮生成器损失：{final_epoch["生成器损失"]:.6g}；
- 最后一轮 Wasserstein 距离估计：{final_epoch["Wasserstein估计"]:.6g}；
- 最后一轮梯度惩罚项：{final_epoch["梯度惩罚"]:.6g}。

## 评价指标

| 方法 | 生成样本数量 | 平均均值相对误差_% | 平均JS散度 | MMD距离 |
| --- | --- | --- | --- | --- |
| {metric_row["方法"]} | {int(metric_row["生成样本数量"])} | {metric_row["平均均值相对误差_%"]:.6g} | {metric_row["平均JS散度"]:.6g} | {metric_row["MMD距离"]:.6g} |

需要注意，普通 WGAN-GP 只说明对抗生成链路是否有效。若其特征误差或 MMD 距离较大，并不代表后续 WGAN-GP 系列不可用，反而可以作为引入“特征约束”和“残差结构”的理由。

## 输出文件说明

- `01_模型与训练记录/`：普通 WGAN-GP 最优模型、最终模型和训练历史；
- `02_生成数据/`：生成窗口数据，包含 `.npy` 和 `.csv`；
- `03_特征表/`：生成样本特征、单特征误差明细和方法指标汇总；
- `04_评价图表/`：训练曲线、波形对比、平均频谱、PCA 分布和指标汇总图。
"""
    (output_dir / "05_实验说明" / "实验3说明.md").write_text(report, encoding="utf-8")


def run(config_path: Path, args: argparse.Namespace) -> None:
    cfg = load_config(config_path)
    wgan_cfg = load_wgan_config(config_path, args)
    set_seed(cfg.seed)
    setup_chinese_font()

    processed_dir = cfg.output_dir / "01_预处理数据"
    real_path = processed_dir / "X-2_窗口数据_原始.npy"
    standardized_path = processed_dir / "X-2_窗口数据_标准化.npy"
    stats_path = processed_dir / "标准化参数.json"
    if not real_path.exists() or not standardized_path.exists() or not stats_path.exists():
        raise FileNotFoundError("没有找到实验0的预处理结果，请先运行实验0。")

    output_dir = PROJECT_ROOT / "实验3_普通WGAN_GP"
    dirs = ensure_dirs(output_dir)
    real_windows = np.load(real_path).astype(np.float32)
    windows_standardized = np.load(standardized_path).astype(np.float32)
    mean, std = load_standardize_stats(stats_path)
    generated_count = wgan_cfg.generated_count or real_windows.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备：{device}")
    print(f"训练样本形状：{windows_standardized.shape}")
    print(f"计划生成样本数量：{generated_count}")

    generator, _, history = train_wgan_gp(windows_standardized, cfg, wgan_cfg, dirs, device)
    plot_training_history(history, dirs["figures"] / "普通WGAN-GP训练曲线.png")

    generated_std = generate_samples(generator, generated_count, wgan_cfg, device)
    generated_std = clamp_generated(
        generated_std,
        windows_standardized,
        wgan_cfg.clamp_percentile_low,
        wgan_cfg.clamp_percentile_high,
    )
    generated_windows = restore_to_original_scale(generated_std, mean, std, cfg.remove_window_mean)
    np.save(dirs["generated"] / "普通WGAN-GP_窗口数据.npy", generated_windows)
    save_windows_csv(dirs["generated"] / "普通WGAN-GP_窗口数据.csv", generated_windows)

    summary = evaluate_generated(real_windows, generated_windows, cfg, dirs)
    plot_waveform_comparison(real_windows, generated_windows, cfg.sampling_frequency, dirs["figures"] / "普通WGAN-GP_波形对比.png")
    plot_mean_spectrum_comparison(real_windows, generated_windows, cfg.sampling_frequency, dirs["figures"] / "普通WGAN-GP_平均频谱对比.png")
    plot_metric_summary(summary, dirs["figures"] / "普通WGAN-GP_指标汇总.png")

    run_info = {
        "实验名称": "实验3：普通 WGAN-GP",
        "输入真实窗口": str(real_path),
        "输入标准化窗口": str(standardized_path),
        "输出目录": str(output_dir),
        "真实窗口数量": int(real_windows.shape[0]),
        "生成窗口数量": int(generated_count),
        "窗口长度": int(real_windows.shape[1]),
        "设备": str(device),
        "随机种子": int(cfg.seed),
        "WGAN-GP参数": asdict(wgan_cfg),
    }
    with (dirs["report"] / "实验3运行信息.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    write_report(output_dir, cfg, wgan_cfg, device, history, summary, real_windows.shape[0])

    print("实验3完成")
    print(f"输出目录：{output_dir}")
    print("普通 WGAN-GP 方法指标汇总：")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验3：普通 WGAN-GP 振动信号数据扩充")
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
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args.config.resolve(), cli_args)
