"""补充评价图表：箱线图、分位数误差和分布检验。

用于在已有生成结果基础上补充更直观的特征分布评价。默认分析当前较优的
“实验5_残差特征约束WGAN_GP_优化版_无截断”结果。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance

from src.experiment0_preprocess_eval import PROJECT_ROOT, compute_features, load_config, setup_chinese_font


FEATURE_GROUPS = {
    "幅值统计特征": [
        "均方根_RMS",
        "峰值_Peak",
        "峰峰值_PeakToPeak",
        "偏度_Skewness",
        "峭度_Kurtosis",
        "峰值因子_CrestFactor",
    ],
    "频域能量特征": [
        "频率重心_Hz",
        "0到1kHz能量占比",
        "1到3kHz能量占比",
        "3到5kHz能量占比",
        "一倍频能量占比",
        "二倍频能量占比",
        "三四倍频能量占比",
    ],
}


def histogram_overlap(real: np.ndarray, generated: np.ndarray, bins: int = 60) -> float:
    combined = np.concatenate([real, generated])
    if np.allclose(combined.min(), combined.max()):
        return 1.0
    hist_range = (float(combined.min()), float(combined.max()))
    p, _ = np.histogram(real, bins=bins, range=hist_range, density=True)
    q, edges = np.histogram(generated, bins=bins, range=hist_range, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(p, q) * widths))


def build_distribution_table(real_features: pd.DataFrame, generated_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in real_features.columns:
        real = real_features[col].to_numpy(dtype=np.float64)
        generated = generated_features[col].to_numpy(dtype=np.float64)
        ks = ks_2samp(real, generated)
        real_q = np.percentile(real, [5, 25, 50, 75, 95])
        gen_q = np.percentile(generated, [5, 25, 50, 75, 95])
        rows.append(
            {
                "特征名称": col,
                "真实Q5": real_q[0],
                "生成Q5": gen_q[0],
                "Q5相对误差_%": abs(gen_q[0] - real_q[0]) / (abs(real_q[0]) + 1e-12) * 100,
                "真实Q25": real_q[1],
                "生成Q25": gen_q[1],
                "Q25相对误差_%": abs(gen_q[1] - real_q[1]) / (abs(real_q[1]) + 1e-12) * 100,
                "真实中位数": real_q[2],
                "生成中位数": gen_q[2],
                "中位数相对误差_%": abs(gen_q[2] - real_q[2]) / (abs(real_q[2]) + 1e-12) * 100,
                "真实Q75": real_q[3],
                "生成Q75": gen_q[3],
                "Q75相对误差_%": abs(gen_q[3] - real_q[3]) / (abs(real_q[3]) + 1e-12) * 100,
                "真实Q95": real_q[4],
                "生成Q95": gen_q[4],
                "Q95相对误差_%": abs(gen_q[4] - real_q[4]) / (abs(real_q[4]) + 1e-12) * 100,
                "KS统计量": float(ks.statistic),
                "KS检验p值": float(ks.pvalue),
                "Wasserstein距离": float(wasserstein_distance(real, generated)),
                "直方图重叠度": histogram_overlap(real, generated),
            }
        )
    return pd.DataFrame(rows)


def plot_feature_boxplots(real_features: pd.DataFrame, generated_features: pd.DataFrame, output_dir: Path) -> None:
    for group_name, columns in FEATURE_GROUPS.items():
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
        axes = axes.ravel()
        for ax, col in zip(axes, columns):
            data = pd.DataFrame(
                {
                    "真实样本": real_features[col].to_numpy(dtype=np.float64),
                    "生成样本": generated_features[col].to_numpy(dtype=np.float64),
                }
            ).melt(var_name="样本类型", value_name="特征值")
            sns.boxplot(data=data, x="样本类型", y="特征值", ax=ax, width=0.55, showfliers=True)
            ax.set_title(col)
            ax.set_xlabel("")
            ax.grid(axis="y", alpha=0.22)
        for ax in axes[len(columns) :]:
            ax.axis("off")
        fig.suptitle(f"{group_name}箱线图对比")
        fig.tight_layout()
        fig.savefig(output_dir / f"{group_name}_箱线图对比.png", dpi=220)
        plt.close(fig)


def plot_feature_violin(real_features: pd.DataFrame, generated_features: pd.DataFrame, output_dir: Path) -> None:
    selected = ["均方根_RMS", "峰值_Peak", "峰峰值_PeakToPeak", "峰值因子_CrestFactor", "频率重心_Hz"]
    fig, axes = plt.subplots(1, len(selected), figsize=(17, 4.8))
    for ax, col in zip(axes, selected):
        data = pd.DataFrame(
            {
                "真实样本": real_features[col].to_numpy(dtype=np.float64),
                "生成样本": generated_features[col].to_numpy(dtype=np.float64),
            }
        ).melt(var_name="样本类型", value_name="特征值")
        sns.violinplot(data=data, x="样本类型", y="特征值", ax=ax, inner="quartile", cut=0)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("关键特征小提琴图对比")
    fig.tight_layout()
    fig.savefig(output_dir / "关键特征_小提琴图对比.png", dpi=220)
    plt.close(fig)


def plot_error_bar(table: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    plot_df = table.sort_values("KS统计量", ascending=False)
    ax.bar(plot_df["特征名称"], plot_df["KS统计量"], color="#4C78A8")
    ax.set_title("各特征 KS 统计量对比")
    ax.set_ylabel("KS统计量，越小越接近")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "各特征KS统计量.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为生成结果补充箱线图和分布检验指标")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--generated-npy",
        type=Path,
        default=PROJECT_ROOT
        / "实验5_残差特征约束WGAN_GP_优化版_无截断"
        / "02_生成数据"
        / "残差特征约束WGAN-GP_窗口数据.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "实验5_残差特征约束WGAN_GP_优化版_无截断" / "06_补充评价图表",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config.resolve())
    setup_chinese_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_path = cfg.output_dir / "01_预处理数据" / "X-2_窗口数据_原始.npy"
    real_windows = np.load(real_path).astype(np.float32)
    generated_windows = np.load(args.generated_npy.resolve()).astype(np.float32)

    real_features = compute_features(real_windows, cfg.sampling_frequency, cfg.spindle_speed_rpm)
    generated_features = compute_features(generated_windows, cfg.sampling_frequency, cfg.spindle_speed_rpm)
    table = build_distribution_table(real_features, generated_features)
    table.to_csv(args.output_dir / "补充分布评价指标.csv", index=False, encoding="utf-8-sig")

    plot_feature_boxplots(real_features, generated_features, args.output_dir)
    plot_feature_violin(real_features, generated_features, args.output_dir)
    plot_error_bar(table, args.output_dir)

    print(f"补充评价完成，输出目录：{args.output_dir}")
    print(table[["特征名称", "KS统计量", "KS检验p值", "直方图重叠度", "Wasserstein距离"]].to_string(index=False))


if __name__ == "__main__":
    main()
