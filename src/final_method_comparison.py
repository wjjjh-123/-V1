"""最终多方法对比与论文图表整理。

对比对象：
1. 传统加噪扩充；
2. VAE 残差扩充；
3. 普通 WGAN-GP 扩充；
4. 残差特征约束 WGAN-GP 扩充。

输出内容包括统一指标表、单特征误差表、PCA 对比、平均频谱对比、
典型波形对比、关键特征箱线图和论文图表说明。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.experiment0_preprocess_eval import PROJECT_ROOT, compute_features, load_config, setup_chinese_font
from src.experiment1_traditional_augmentation import compare_features


@dataclass(frozen=True)
class MethodSpec:
    name: str
    short_name: str
    npy_path: Path
    color: str
    description: str


def build_method_specs(project_root: Path) -> list[MethodSpec]:
    return [
        MethodSpec(
            name="传统加噪扩充",
            short_name="加噪",
            npy_path=project_root / "实验1_传统扩充方法" / "01_生成数据" / "加噪扩充_窗口数据.npy",
            color="#4C78A8",
            description="在真实窗口上叠加小幅高斯噪声，作为最基础的传统信号扩充基线。",
        ),
        MethodSpec(
            name="VAE残差扩充",
            short_name="VAE",
            npy_path=project_root / "实验2_VAE" / "02_生成数据" / "VAE残差扩充_窗口数据.npy",
            color="#F58518",
            description="VAE 是变分自编码器；这里采用残差扩充方式，保留真实窗口主形态并叠加潜在空间扰动。",
        ),
        MethodSpec(
            name="普通WGAN-GP",
            short_name="WGAN-GP",
            npy_path=project_root / "实验3_普通WGAN_GP" / "02_生成数据" / "普通WGAN-GP_窗口数据.npy",
            color="#54A24B",
            description="WGAN-GP 是带梯度惩罚的 Wasserstein 生成对抗网络；本方法不使用特征约束和残差结构。",
        ),
        MethodSpec(
            name="残差特征约束WGAN-GP",
            short_name="残差特征WGAN",
            npy_path=project_root
            / "实验5_残差特征约束WGAN_GP_优化版_无截断"
            / "02_生成数据"
            / "残差特征约束WGAN-GP_窗口数据.npy",
            color="#E45756",
            description="在 WGAN-GP 中加入振动特征约束和残差生成结构，使生成样本更贴近真实主轴振动分布。",
        ),
    ]


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "tables": output_dir / "01_汇总表",
        "figures": output_dir / "02_论文图表",
        "report": output_dir / "03_论文说明",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_windows(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"未找到窗口数据文件：{path}")
    return np.load(path).astype(np.float32)


def method_quality_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "平均均值相对误差_%",
        "平均标准差相对误差_%",
        "平均JS散度",
        "平均Wasserstein距离",
        "MMD距离",
    ]
    ranking = summary[["方法"] + metrics].copy()
    for metric in metrics:
        ranking[f"{metric}_排名"] = ranking[metric].rank(method="min", ascending=True).astype(int)

    normalized_parts = []
    for metric in metrics:
        values = ranking[metric].to_numpy(dtype=np.float64)
        value_min = float(values.min())
        value_max = float(values.max())
        if np.isclose(value_min, value_max):
            normalized = np.ones_like(values)
        else:
            normalized = 1.0 - (values - value_min) / (value_max - value_min)
        normalized_parts.append(normalized)
    ranking["综合得分"] = np.vstack(normalized_parts).mean(axis=0)
    ranking["综合排序"] = ranking["综合得分"].rank(method="min", ascending=False).astype(int)
    return ranking.sort_values("综合排序")


def plot_metric_bars(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("平均均值相对误差_%", "均值相对误差 / %", False),
        ("平均标准差相对误差_%", "标准差相对误差 / %", False),
        ("平均JS散度", "JS散度", True),
        ("平均Wasserstein距离", "Wasserstein距离", True),
        ("MMD距离", "MMD距离", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    axes = axes.ravel()
    colors = summary["颜色"].tolist()
    for ax, (metric, ylabel, use_log) in zip(axes, metrics):
        values = summary[metric].to_numpy(dtype=np.float64)
        ax.bar(summary["短名称"], values, color=colors, width=0.62)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        if use_log:
            min_positive = max(float(values[values > 0].min()) * 0.5, 1e-9)
            ax.set_yscale("log")
            ax.set_ylim(bottom=min_positive)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        for x, value in enumerate(values):
            ax.text(x, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.22)
    axes[-1].axis("off")
    fig.suptitle("图1 多方法生成样本质量评价指标对比", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_pca_all(real_features: pd.DataFrame, feature_by_method: dict[str, pd.DataFrame], summary: pd.DataFrame, path: Path) -> None:
    arrays = [real_features.to_numpy(dtype=np.float64)]
    labels = ["真实样本"]
    for method in summary["方法"]:
        arrays.append(feature_by_method[method].to_numpy(dtype=np.float64))
        labels.append(method)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.vstack(arrays))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    ratio = pca.explained_variance_ratio_ * 100

    counts = [arr.shape[0] for arr in arrays]
    starts = np.cumsum([0] + counts[:-1])
    colors = ["#222222"] + summary["颜色"].tolist()

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    for label, start, count, color in zip(labels, starts, counts, colors):
        sub = coords[start : start + count]
        size = 12 if label == "真实样本" else 10
        alpha = 0.46 if label == "真实样本" else 0.34
        ax.scatter(sub[:, 0], sub[:, 1], s=size, alpha=alpha, label=label, color=color, edgecolors="none")
    ax.set_title("图2 多方法样本的 PCA 特征空间分布")
    ax.set_xlabel(f"第一主成分 / {ratio[0]:.1f}%")
    ax.set_ylabel(f"第二主成分 / {ratio[1]:.1f}%")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def mean_spectrum(windows: np.ndarray, sampling_frequency: int) -> tuple[np.ndarray, np.ndarray]:
    centered = windows - windows.mean(axis=1, keepdims=True)
    freqs = rfftfreq(windows.shape[1], d=1.0 / sampling_frequency)
    amp = np.abs(rfft(centered, axis=1)).mean(axis=0)
    return freqs, amp


def plot_mean_spectrum(
    real_windows: np.ndarray,
    windows_by_method: dict[str, np.ndarray],
    summary: pd.DataFrame,
    sampling_frequency: int,
    path: Path,
) -> None:
    freqs, real_amp = mean_spectrum(real_windows, sampling_frequency)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(freqs, real_amp, color="#222222", linewidth=1.45, label="真实样本")
    for _, row in summary.iterrows():
        _, gen_amp = mean_spectrum(windows_by_method[row["方法"]], sampling_frequency)
        ax.plot(freqs, gen_amp, color=row["颜色"], linewidth=1.0, alpha=0.92, label=row["方法"])

    ax.set_title("图3 真实样本与不同扩充方法的平均频谱对比")
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("平均幅值")
    ax.set_xlim(0, sampling_frequency / 2)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_waveforms(
    real_windows: np.ndarray,
    windows_by_method: dict[str, np.ndarray],
    summary: pd.DataFrame,
    sampling_frequency: int,
    path: Path,
) -> None:
    time_ms = np.arange(real_windows.shape[1]) / sampling_frequency * 1000
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, summary.iterrows()):
        method = row["方法"]
        ax.plot(time_ms, real_windows[0], color="#222222", linewidth=0.9, label="真实样本")
        ax.plot(time_ms, windows_by_method[method][0], color=row["颜色"], linewidth=0.9, alpha=0.82, label=method)
        ax.set_title(method)
        ax.set_xlabel("时间 / ms")
        ax.set_ylabel("振动幅值")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle("图4 真实样本与不同扩充方法的典型波形对比", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_feature_boxplots(
    real_features: pd.DataFrame,
    feature_by_method: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    path: Path,
) -> None:
    selected_features = [
        "均方根_RMS",
        "峰值_Peak",
        "峰峰值_PeakToPeak",
        "峭度_Kurtosis",
        "频率重心_Hz",
        "一倍频能量占比",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    axes = axes.ravel()
    labels = ["真实"] + summary["短名称"].tolist()
    box_colors = ["#CCCCCC"] + summary["颜色"].tolist()

    for ax, feature in zip(axes, selected_features):
        data = [real_features[feature].to_numpy(dtype=np.float64)]
        for method in summary["方法"]:
            data.append(feature_by_method[method][feature].to_numpy(dtype=np.float64))
        box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.62)
        for patch, color in zip(box["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax.set_title(feature)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("图5 关键振动特征分布箱线图对比", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_feature_error_heatmap(detail: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    pivot = detail.pivot(index="特征名称", columns="方法", values="均值相对误差_%")
    pivot = pivot[summary["方法"].tolist()]
    values = pivot.to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=summary["短名称"].tolist(), rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index.tolist())
    ax.set_title("图6 各方法单特征均值相对误差热力图")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("均值相对误差 / %")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2g}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_report(
    dirs: dict[str, Path],
    cfg,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    method_specs: list[MethodSpec],
) -> None:
    table_cols = [
        "方法",
        "生成样本数量",
        "平均均值相对误差_%",
        "平均标准差相对误差_%",
        "平均JS散度",
        "平均Wasserstein距离",
        "MMD距离",
    ]
    md_lines = [
        "# 最终多方法对比与论文图表整理",
        "",
        "## 对比方法",
        "",
    ]
    for spec in method_specs:
        md_lines.append(f"- **{spec.name}**：{spec.description}")

    md_lines += [
        "",
        "## 数据与评价设置",
        "",
        f"- 使用通道：`{cfg.channel}`；",
        f"- 采样频率：{cfg.sampling_frequency} Hz；",
        f"- 主轴转速：{cfg.spindle_speed_rpm} rpm；",
        f"- 单个窗口长度：{cfg.window_length} 点；",
        "- 评价特征：RMS、峰值、峰峰值、偏度、峭度、峰值因子、频率重心、频带能量占比和倍频能量占比；",
        "- JS 散度、Wasserstein 距离和 MMD 距离均为越小越接近真实样本。",
        "",
        "## 主要指标汇总",
        "",
    ]

    md_lines.append("| " + " | ".join(table_cols) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(table_cols)) + " |")
    for _, row in summary[table_cols].iterrows():
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["方法"]),
                    str(int(row["生成样本数量"])),
                    f"{row['平均均值相对误差_%']:.6g}",
                    f"{row['平均标准差相对误差_%']:.6g}",
                    f"{row['平均JS散度']:.6g}",
                    f"{row['平均Wasserstein距离']:.6g}",
                    f"{row['MMD距离']:.6g}",
                ]
            )
            + " |"
        )

    best = ranking.iloc[0]
    md_lines += [
        "",
        "## 结果解读",
        "",
        f"从综合排序看，当前四种方法中 **{best['方法']}** 的整体指标最优。",
        "传统加噪扩充可以作为简单基线，优点是波形扰动可控、不会明显改变原始信号结构；但它本质上只能在原始样本附近做局部扰动，难以学习复杂分布。",
        "VAE 残差扩充相比普通先验采样更适合当前健康主轴短周期振动数据，因为它保留真实窗口的主形态，只在潜在空间引入变化。",
        "普通 WGAN-GP 能生成独立样本，但在没有条件特征和残差约束时，部分时域/频域统计量容易偏离真实样本。",
        "残差特征约束 WGAN-GP 同时利用残差结构和振动特征约束，在统计特征、频谱结构和特征空间分布上更接近真实样本。",
        "",
        "## 论文图表清单",
        "",
        "- 图1：多方法生成样本质量评价指标对比；",
        "- 图2：多方法样本的 PCA 特征空间分布；",
        "- 图3：真实样本与不同扩充方法的平均频谱对比；",
        "- 图4：真实样本与不同扩充方法的典型波形对比；",
        "- 图5：关键振动特征分布箱线图对比；",
        "- 图6：各方法单特征均值相对误差热力图。",
        "",
        "## 写作提示",
        "",
        "这些结果主要证明生成样本与真实振动样本在时域、频域和特征分布上具有一致性。若要进一步证明数据扩充对最终精度退化建模的实际作用，后续还应加入下游预测任务验证，例如比较加入扩充样本前后精度预测模型的 MAE、RMSE 和 R²。",
    ]
    (dirs["report"] / "多方法对比结果解读.md").write_text("\n".join(md_lines), encoding="utf-8")


def run(config_path: Path, output_dir: Path) -> None:
    cfg = load_config(config_path.resolve())
    setup_chinese_font()
    dirs = ensure_dirs(output_dir.resolve())

    real_path = cfg.output_dir / "01_预处理数据" / "X-2_窗口数据_原始.npy"
    real_windows = load_windows(real_path)
    real_features = compute_features(real_windows, cfg.sampling_frequency, cfg.spindle_speed_rpm)

    method_specs = build_method_specs(PROJECT_ROOT)
    windows_by_method: dict[str, np.ndarray] = {}
    feature_by_method: dict[str, pd.DataFrame] = {}
    detail_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for spec in method_specs:
        windows = load_windows(spec.npy_path)
        features = compute_features(windows, cfg.sampling_frequency, cfg.spindle_speed_rpm)
        detail, metrics = compare_features(real_features, features, spec.name, cfg.seed)
        detail["短名称"] = spec.short_name
        detail_tables.append(detail)
        summary_rows.append(
            {
                "方法": spec.name,
                "短名称": spec.short_name,
                "生成样本数量": int(windows.shape[0]),
                "颜色": spec.color,
                **metrics,
            }
        )
        windows_by_method[spec.name] = windows
        feature_by_method[spec.name] = features

    detail_all = pd.concat(detail_tables, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    summary = summary.rename(
        columns={
            "平均均值相对误差_%": "平均均值相对误差_%",
            "平均标准差相对误差_%": "平均标准差相对误差_%",
        }
    )
    ranking = method_quality_summary(summary)

    summary.to_csv(dirs["tables"] / "多方法指标汇总.csv", index=False, encoding="utf-8-sig")
    paper_summary_columns = [
        "方法",
        "生成样本数量",
        "平均均值相对误差_%",
        "平均标准差相对误差_%",
        "平均JS散度",
        "平均Wasserstein距离",
        "MMD距离",
    ]
    summary[paper_summary_columns].to_csv(
        dirs["tables"] / "多方法指标汇总_论文版.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detail_all.to_csv(dirs["tables"] / "多方法单特征误差明细.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(dirs["tables"] / "多方法综合排序.csv", index=False, encoding="utf-8-sig")

    real_features.to_csv(dirs["tables"] / "真实样本_公共评价特征.csv", index=False, encoding="utf-8-sig")
    for method, features in feature_by_method.items():
        safe_name = method.replace("/", "_")
        features.to_csv(dirs["tables"] / f"{safe_name}_公共评价特征.csv", index=False, encoding="utf-8-sig")

    plot_metric_bars(summary, dirs["figures"] / "图1_多方法生成质量指标对比.png")
    plot_pca_all(real_features, feature_by_method, summary, dirs["figures"] / "图2_多方法PCA特征空间分布.png")
    plot_mean_spectrum(
        real_windows,
        windows_by_method,
        summary,
        cfg.sampling_frequency,
        dirs["figures"] / "图3_平均频谱对比.png",
    )
    plot_waveforms(
        real_windows,
        windows_by_method,
        summary,
        cfg.sampling_frequency,
        dirs["figures"] / "图4_典型波形对比.png",
    )
    plot_feature_boxplots(
        real_features,
        feature_by_method,
        summary,
        dirs["figures"] / "图5_关键特征箱线图对比.png",
    )
    plot_feature_error_heatmap(detail_all, summary, dirs["figures"] / "图6_单特征均值误差热力图.png")
    write_report(dirs, cfg, summary, ranking, method_specs)

    print(f"多方法对比整理完成，输出目录：{output_dir}")
    print(summary.drop(columns=["颜色"]).to_string(index=False))
    print("\n综合排序：")
    print(ranking[["方法", "综合得分", "综合排序"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最终多方法对比与论文图表整理")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "最终多方法对比与论文图表")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
