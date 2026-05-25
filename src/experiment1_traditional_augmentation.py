"""实验1：传统振动信号数据扩充方法。

本脚本复用实验0保存的真实窗口数据，构造若干不需要训练模型的传统扩充基线：
1. 加性高斯噪声；
2. 幅值缩放；
3. 时间平移；
4. 组合扰动。

随后使用实验0中的公共评价特征，对真实样本和扩充样本进行统一评价。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.spatial.distance import cdist
from scipy.stats import entropy, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.experiment0_preprocess_eval import (
    PROJECT_ROOT,
    build_feature_summary,
    compute_features,
    load_config,
    setup_chinese_font,
)


METHODS = {
    "noise": "加噪扩充",
    "scale": "幅值缩放扩充",
    "shift": "时间平移扩充",
    "mixed": "组合传统扩充",
}


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "generated": output_dir / "01_生成数据",
        "features": output_dir / "02_特征表",
        "figures": output_dir / "03_评价图表",
        "report": output_dir / "04_实验说明",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def remove_each_window_mean(windows: np.ndarray) -> np.ndarray:
    return (windows - windows.mean(axis=1, keepdims=True)).astype(np.float32)


def add_gaussian_noise(windows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    window_std = windows.std(axis=1, keepdims=True)
    noise_level = rng.uniform(0.02, 0.08, size=(windows.shape[0], 1)).astype(np.float32)
    noise = rng.normal(0.0, noise_level * window_std, size=windows.shape).astype(np.float32)
    return windows + noise


def amplitude_scaling(windows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale = rng.uniform(0.90, 1.10, size=(windows.shape[0], 1)).astype(np.float32)
    return windows * scale


def time_shift(windows: np.ndarray, rng: np.random.Generator, max_shift: int = 128) -> np.ndarray:
    shifts = rng.integers(-max_shift, max_shift + 1, size=windows.shape[0])
    shifted = np.empty_like(windows)
    for i, shift in enumerate(shifts):
        shifted[i] = np.roll(windows[i], int(shift))
    return shifted


def mixed_augmentation(windows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = amplitude_scaling(windows, rng)
    augmented = time_shift(augmented, rng, max_shift=96)
    window_std = augmented.std(axis=1, keepdims=True)
    noise_level = rng.uniform(0.01, 0.05, size=(windows.shape[0], 1)).astype(np.float32)
    noise = rng.normal(0.0, noise_level * window_std, size=windows.shape).astype(np.float32)
    return augmented + noise


def build_traditional_samples(
    windows: np.ndarray,
    seed: int,
    remove_window_mean: bool,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    generated = {
        "noise": add_gaussian_noise(windows, rng),
        "scale": amplitude_scaling(windows, rng),
        "shift": time_shift(windows, rng),
        "mixed": mixed_augmentation(windows, rng),
    }
    if remove_window_mean:
        generated = {name: remove_each_window_mean(value) for name, value in generated.items()}
    return generated


def js_divergence(real: np.ndarray, generated: np.ndarray, bins: int = 60) -> float:
    combined = np.concatenate([real, generated])
    if np.allclose(combined.min(), combined.max()):
        return 0.0
    hist_range = (float(combined.min()), float(combined.max()))
    p, _ = np.histogram(real, bins=bins, range=hist_range, density=False)
    q, _ = np.histogram(generated, bins=bins, range=hist_range, density=False)
    p = p.astype(np.float64) + 1e-12
    q = q.astype(np.float64) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m) + 0.5 * entropy(q, m))


def median_heuristic_gamma(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    combined = np.vstack([x, y])
    sample_size = min(600, combined.shape[0])
    sample_idx = rng.choice(combined.shape[0], size=sample_size, replace=False)
    sample = combined[sample_idx]
    distances = cdist(sample, sample, metric="sqeuclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    median_sq = float(np.median(upper[upper > 0]))
    if median_sq <= 1e-12:
        return 1.0
    return 1.0 / (2.0 * median_sq)


def rbf_mmd(real_features: np.ndarray, generated_features: np.ndarray, seed: int) -> float:
    scaler = StandardScaler()
    combined = scaler.fit_transform(np.vstack([real_features, generated_features]))
    x = combined[: real_features.shape[0]]
    y = combined[real_features.shape[0] :]
    rng = np.random.default_rng(seed)
    gamma = median_heuristic_gamma(x, y, rng)

    k_xx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
    k_yy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean"))
    k_xy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))
    mmd = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return max(0.0, mmd)


def compare_features(
    real_features: pd.DataFrame,
    generated_features: pd.DataFrame,
    method_name: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | str]] = []
    feature_cols = list(real_features.columns)
    for col in feature_cols:
        real = real_features[col].to_numpy(dtype=np.float64)
        generated = generated_features[col].to_numpy(dtype=np.float64)
        real_mean = float(real.mean())
        gen_mean = float(generated.mean())
        real_std = float(real.std(ddof=1))
        gen_std = float(generated.std(ddof=1))
        rows.append(
            {
                "方法": method_name,
                "特征名称": col,
                "真实均值": real_mean,
                "生成均值": gen_mean,
                "均值相对误差_%": abs(gen_mean - real_mean) / (abs(real_mean) + 1e-12) * 100,
                "真实标准差": real_std,
                "生成标准差": gen_std,
                "标准差相对误差_%": abs(gen_std - real_std) / (abs(real_std) + 1e-12) * 100,
                "JS散度": js_divergence(real, generated),
                "Wasserstein距离": float(wasserstein_distance(real, generated)),
            }
        )

    detail = pd.DataFrame(rows)
    metrics = {
        "平均均值相对误差_%": float(detail["均值相对误差_%"].mean()),
        "平均标准差相对误差_%": float(detail["标准差相对误差_%"].mean()),
        "平均JS散度": float(detail["JS散度"].mean()),
        "平均Wasserstein距离": float(detail["Wasserstein距离"].mean()),
        "MMD距离": rbf_mmd(
            real_features.to_numpy(dtype=np.float64),
            generated_features.to_numpy(dtype=np.float64),
            seed,
        ),
    }
    return detail, metrics


def save_windows_csv(path: Path, windows: np.ndarray) -> None:
    columns = [f"采样点_{i}" for i in range(windows.shape[1])]
    pd.DataFrame(windows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def plot_waveform_comparison(
    real_windows: np.ndarray,
    generated_by_method: dict[str, np.ndarray],
    sampling_frequency: int,
    path: Path,
) -> None:
    time_ms = np.arange(real_windows.shape[1]) / sampling_frequency * 1000
    fig, axes = plt.subplots(len(generated_by_method), 1, figsize=(11, 8), sharex=True)
    if len(generated_by_method) == 1:
        axes = [axes]

    for ax, (method_key, generated) in zip(axes, generated_by_method.items()):
        ax.plot(time_ms, real_windows[0], color="#4C78A8", linewidth=0.85, label="真实样本")
        ax.plot(time_ms, generated[0], color="#F58518", linewidth=0.85, alpha=0.82, label=METHODS[method_key])
        ax.set_ylabel("幅值")
        ax.set_title(METHODS[method_key])
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("时间 / ms")
    fig.suptitle("传统扩充方法的波形对比")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mean_spectrum_comparison(
    real_windows: np.ndarray,
    generated_by_method: dict[str, np.ndarray],
    sampling_frequency: int,
    path: Path,
) -> None:
    freqs = rfftfreq(real_windows.shape[1], d=1.0 / sampling_frequency)
    real_amp = np.abs(rfft(real_windows - real_windows.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(freqs, real_amp, color="#222222", linewidth=1.3, label="真实样本")
    colors = ["#F58518", "#54A24B", "#B279A2", "#E45756"]
    for color, (method_key, generated) in zip(colors, generated_by_method.items()):
        gen_amp = np.abs(rfft(generated - generated.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
        ax.plot(freqs, gen_amp, linewidth=0.95, alpha=0.9, color=color, label=METHODS[method_key])

    ax.set_title("真实样本与传统扩充样本平均频谱对比")
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("平均幅值")
    ax.set_xlim(0, sampling_frequency / 2)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_metric_summary(summary: pd.DataFrame, path: Path) -> None:
    metric_cols = ["平均均值相对误差_%", "平均JS散度", "MMD距离"]
    metric_titles = {
        "平均均值相对误差_%": "平均均值相对误差 / %",
        "平均JS散度": "平均 JS 散度",
        "MMD距离": "MMD 距离",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for ax, col in zip(axes, metric_cols):
        ax.bar(summary["方法"], summary[col], color=colors[: len(summary)])
        ax.set_title(metric_titles[col])
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("传统扩充方法评价指标汇总")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_pca_comparison(real_features: pd.DataFrame, generated_features: pd.DataFrame, method_name: str, path: Path) -> None:
    real = real_features.to_numpy(dtype=np.float64)
    generated = generated_features.to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.vstack([real, generated]))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    real_coords = coords[: real.shape[0]]
    gen_coords = coords[real.shape[0] :]
    ratio = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(real_coords[:, 0], real_coords[:, 1], s=12, alpha=0.55, label="真实样本", color="#4C78A8")
    ax.scatter(gen_coords[:, 0], gen_coords[:, 1], s=12, alpha=0.55, label=method_name, color="#F58518")
    ax.set_title(f"{method_name} 与真实样本特征 PCA 对比")
    ax.set_xlabel(f"第一主成分 / {ratio[0]:.1f}%")
    ax.set_ylabel(f"第二主成分 / {ratio[1]:.1f}%")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    output_dir: Path,
    real_count: int,
    window_length: int,
    summary: pd.DataFrame,
) -> None:
    best_mean = summary.sort_values("平均均值相对误差_%").iloc[0]
    best_mmd = summary.sort_values("MMD距离").iloc[0]
    table_columns = [
        "方法",
        "生成样本数",
        "平均均值相对误差_%",
        "平均JS散度",
        "MMD距离",
    ]
    table_lines = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join(["---"] * len(table_columns)) + " |",
    ]
    for _, row in summary[table_columns].iterrows():
        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["方法"]),
                    str(int(row["生成样本数"])),
                    f"{row['平均均值相对误差_%']:.6g}",
                    f"{row['平均JS散度']:.6g}",
                    f"{row['MMD距离']:.6g}",
                ]
            )
            + " |"
        )
    table_text = "\n".join(table_lines)
    report = f"""# 实验1：传统扩充方法

## 实验目的

本实验用于建立不依赖深度学习模型的传统数据扩充基线。后续 VAE（变分自编码器）和 WGAN-GP（带梯度惩罚的 Wasserstein 生成对抗网络）等方法，需要与这些简单方法比较，才能说明复杂生成模型是否真的带来收益。

## 使用的数据

- 输入数据来自 `实验0_公共预处理与评价框架/01_预处理数据/X-2_窗口数据_原始.npy`；
- 真实窗口数量：{real_count}；
- 单个窗口长度：{window_length} 个采样点；
- 每种传统方法均生成 {real_count} 个扩充窗口。

## 传统扩充方法

- 加噪扩充：在每个窗口上叠加小幅高斯噪声，用于模拟测量噪声和轻微随机扰动；
- 幅值缩放扩充：对整个窗口乘以 0.90 到 1.10 范围内的随机系数，用于模拟振动幅值轻微变化；
- 时间平移扩充：对窗口做小范围循环平移，用于模拟采样起点不同；
- 组合传统扩充：同时加入幅值缩放、时间平移和小幅噪声。

## 指标说明

- 均值相对误差越小，说明生成样本的平均特征越接近真实样本；
- JS 散度越小，说明单个特征的分布形状越接近；
- MMD 距离（最大均值差异）越小，说明多维特征整体分布越接近；
- Wasserstein 距离可理解为两个特征分布之间的“搬运距离”，越小越好。

## 本次运行的指标汇总

{table_text}

从本次结果看，按平均特征均值误差排序，较优方法是 **{best_mean["方法"]}**；按 MMD 距离排序，较优方法是 **{best_mmd["方法"]}**。需要注意，传统扩充方法通常只能产生局部扰动，结果接近真实样本并不一定代表学到了新的退化规律，它更适合作为后续生成模型实验的基线。

## 输出文件

- `01_生成数据/`：每种传统扩充方法生成的窗口数据，包含 `.npy` 和 `.csv`；
- `02_特征表/`：生成样本特征、单特征误差表和方法指标汇总；
- `03_评价图表/`：波形对比、平均频谱对比、PCA 对比和指标柱状图；
- `04_实验说明/实验1说明.md`：本说明文件。
"""
    (output_dir / "04_实验说明" / "实验1说明.md").write_text(report, encoding="utf-8")


def run(config_path: Path) -> None:
    cfg = load_config(config_path)
    setup_chinese_font()

    experiment0_dir = cfg.output_dir
    processed_dir = experiment0_dir / "01_预处理数据"
    real_window_path = processed_dir / "X-2_窗口数据_原始.npy"
    if not real_window_path.exists():
        raise FileNotFoundError(f"没有找到实验0窗口数据：{real_window_path}，请先运行实验0。")

    output_dir = PROJECT_ROOT / "实验1_传统扩充方法"
    dirs = ensure_dirs(output_dir)

    real_windows = np.load(real_window_path).astype(np.float32)
    generated_by_method = build_traditional_samples(real_windows, cfg.seed, cfg.remove_window_mean)

    real_features = compute_features(real_windows, cfg.sampling_frequency, cfg.spindle_speed_rpm)
    all_detail: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str | int]] = []

    for offset, (method_key, generated_windows) in enumerate(generated_by_method.items()):
        method_name = METHODS[method_key]
        np.save(dirs["generated"] / f"{method_name}_窗口数据.npy", generated_windows)
        save_windows_csv(dirs["generated"] / f"{method_name}_窗口数据.csv", generated_windows)

        generated_features = compute_features(generated_windows, cfg.sampling_frequency, cfg.spindle_speed_rpm)
        generated_features.insert(0, "样本编号", np.arange(generated_features.shape[0]))
        generated_features.to_csv(dirs["features"] / f"{method_name}_公共评价特征.csv", index=False, encoding="utf-8-sig")
        feature_summary = build_feature_summary(generated_features.drop(columns=["样本编号"]))
        feature_summary.to_csv(dirs["features"] / f"{method_name}_特征统计摘要.csv", encoding="utf-8-sig")

        detail, metrics = compare_features(
            real_features,
            generated_features.drop(columns=["样本编号"]),
            method_name,
            cfg.seed + offset,
        )
        all_detail.append(detail)
        summary_rows.append({"方法": method_name, "生成样本数": generated_windows.shape[0], **metrics})

        plot_pca_comparison(
            real_features,
            generated_features.drop(columns=["样本编号"]),
            method_name,
            dirs["figures"] / f"PCA特征分布_{method_name}.png",
        )

    detail_table = pd.concat(all_detail, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    detail_table.to_csv(dirs["features"] / "传统扩充_单特征误差明细.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(dirs["features"] / "传统扩充_方法指标汇总.csv", index=False, encoding="utf-8-sig")

    plot_waveform_comparison(
        real_windows,
        generated_by_method,
        cfg.sampling_frequency,
        dirs["figures"] / "传统扩充_波形对比.png",
    )
    plot_mean_spectrum_comparison(
        real_windows,
        generated_by_method,
        cfg.sampling_frequency,
        dirs["figures"] / "传统扩充_平均频谱对比.png",
    )
    plot_metric_summary(summary, dirs["figures"] / "传统扩充_指标汇总.png")
    write_report(output_dir, real_windows.shape[0], real_windows.shape[1], summary)

    save_json = {
        "实验名称": "实验1：传统扩充方法",
        "输入窗口数据": str(real_window_path),
        "输出目录": str(output_dir),
        "真实窗口数量": int(real_windows.shape[0]),
        "窗口长度": int(real_windows.shape[1]),
        "随机种子": int(cfg.seed),
        "扩充方法": METHODS,
    }
    with (dirs["report"] / "实验1运行信息.json").open("w", encoding="utf-8") as f:
        json.dump(save_json, f, ensure_ascii=False, indent=2)

    print("实验1完成")
    print(f"输出目录：{output_dir}")
    print(f"真实窗口数量：{real_windows.shape[0]}")
    print("方法指标汇总：")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验1：传统振动信号数据扩充方法")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="配置文件路径，默认读取项目根目录 config.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config.resolve())
