"""实验0：公共预处理与评价框架。

本脚本完成三件事：
1. 从源 CSV 文件读取指定振动通道；
2. 按固定窗口切分并标准化，保存后续生成模型可复用的数据；
3. 提取公共评价特征并绘制中文图表，作为后续扩充方法对比的基准。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_simple_yaml(config_path: Path) -> dict[str, object]:
    """解析本项目这种简单两级 YAML，避免额外依赖 pyyaml。"""
    result: dict[str, object] = {}
    current_section: dict[str, object] | None = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section_name = line[:-1].strip()
            current_section = {}
            result[section_name] = current_section
            continue
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = parse_scalar(value.strip())
        if raw_line.startswith(" ") and current_section is not None:
            current_section[key] = value
        else:
            result[key] = value
            current_section = None
    return result


def parse_scalar(value: str) -> object:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


@dataclass(frozen=True)
class ExperimentConfig:
    source_dir: Path
    output_dir: Path
    channel: str
    sampling_frequency: int
    spindle_speed_rpm: int
    window_length: int
    step_size: int
    remove_window_mean: bool
    seed: int
    max_example_windows: int
    pca_random_state: int


def load_config(config_path: Path) -> ExperimentConfig:
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except ModuleNotFoundError:
        raw = parse_simple_yaml(config_path)

    data_cfg = raw["data"]
    exp_cfg = raw.get("experiment0", {})
    return ExperimentConfig(
        source_dir=(PROJECT_ROOT / data_cfg["source_dir"]).resolve(),
        output_dir=(PROJECT_ROOT / data_cfg["output_dir"]).resolve(),
        channel=str(data_cfg["channel"]),
        sampling_frequency=int(data_cfg["sampling_frequency"]),
        spindle_speed_rpm=int(data_cfg["spindle_speed_rpm"]),
        window_length=int(data_cfg["window_length"]),
        step_size=int(data_cfg["step_size"]),
        remove_window_mean=bool(data_cfg["remove_window_mean"]),
        seed=int(raw.get("seed", 42)),
        max_example_windows=int(exp_cfg.get("max_example_windows", 6)),
        pca_random_state=int(exp_cfg.get("pca_random_state", raw.get("seed", 42))),
    )


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "processed": output_dir / "01_预处理数据",
        "features": output_dir / "02_特征表",
        "figures": output_dir / "03_评价图表",
        "report": output_dir / "04_实验说明",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def list_csv_files(source_dir: Path) -> list[Path]:
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"源数据目录中没有找到 CSV 文件：{source_dir}")
    return files


def read_channel(csv_path: Path, channel: str) -> np.ndarray:
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [str(col).strip() for col in df.columns]
    if channel not in df.columns:
        raise KeyError(f"{csv_path.name} 中没有找到通道列 {channel}，实际列名为：{list(df.columns)}")
    values = pd.to_numeric(df[channel], errors="coerce").to_numpy(dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"{csv_path.name} 的 {channel} 通道没有有效数值")
    return values


def sliding_windows(signal: np.ndarray, length: int, step: int) -> np.ndarray:
    if signal.size < length:
        return np.empty((0, length), dtype=np.float32)
    starts = np.arange(0, signal.size - length + 1, step, dtype=np.int64)
    windows = np.stack([signal[start : start + length] for start in starts])
    return windows.astype(np.float32, copy=False)


def build_windows(
    files: Iterable[Path],
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    all_windows: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    for file_index, csv_path in enumerate(files):
        signal = read_channel(csv_path, cfg.channel)
        windows = sliding_windows(signal, cfg.window_length, cfg.step_size)
        if cfg.remove_window_mean and windows.size:
            windows = windows - windows.mean(axis=1, keepdims=True)

        all_windows.append(windows)
        for window_index in range(windows.shape[0]):
            start = window_index * cfg.step_size
            meta_rows.append(
                {
                    "样本编号": len(meta_rows),
                    "源文件": csv_path.name,
                    "源文件序号": file_index,
                    "文件内窗口序号": window_index,
                    "起始采样点": start,
                    "结束采样点": start + cfg.window_length - 1,
                    "起始时间_s": start / cfg.sampling_frequency,
                    "结束时间_s": (start + cfg.window_length - 1) / cfg.sampling_frequency,
                }
            )

    if not all_windows:
        raise RuntimeError("没有生成任何窗口，请检查源数据和窗口参数")

    windows_all = np.concatenate(all_windows, axis=0)
    metadata = pd.DataFrame(meta_rows)
    return windows_all, metadata


def standardize_windows(windows: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    mean = float(windows.mean())
    std = float(windows.std(ddof=0))
    if std <= 1e-12:
        raise ValueError("窗口数据标准差过小，无法标准化")
    standardized = ((windows - mean) / std).astype(np.float32)
    stats = {"全局均值": mean, "全局标准差": std}
    return standardized, stats


def band_energy_ratio(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    total = float(power.sum()) + 1e-12
    mask = (freqs >= low) & (freqs < high)
    return float(power[mask].sum() / total)


def harmonic_energy_ratio(
    freqs: np.ndarray,
    power: np.ndarray,
    center_hz: float,
    bandwidth_hz: float = 5.0,
) -> float:
    total = float(power.sum()) + 1e-12
    mask = (freqs >= center_hz - bandwidth_hz) & (freqs <= center_hz + bandwidth_hz)
    return float(power[mask].sum() / total)


def compute_features(
    windows: np.ndarray,
    sampling_frequency: int,
    spindle_speed_rpm: int,
) -> pd.DataFrame:
    base_freq = spindle_speed_rpm / 60.0
    freqs = rfftfreq(windows.shape[1], d=1.0 / sampling_frequency)
    rows: list[dict[str, float]] = []

    for window in windows:
        centered = window - window.mean()
        spectrum = np.abs(rfft(centered))
        power = spectrum**2
        power_sum = float(power.sum()) + 1e-12
        rms = float(np.sqrt(np.mean(centered**2)))
        peak = float(np.max(np.abs(centered)))

        rows.append(
            {
                "均方根_RMS": rms,
                "峰值_Peak": peak,
                "峰峰值_PeakToPeak": float(centered.max() - centered.min()),
                "偏度_Skewness": float(skew(centered, bias=False)),
                "峭度_Kurtosis": float(kurtosis(centered, fisher=False, bias=False)),
                "峰值因子_CrestFactor": float(peak / (rms + 1e-12)),
                "频率重心_Hz": float((freqs * power).sum() / power_sum),
                "0到1kHz能量占比": band_energy_ratio(freqs, power, 0, 1000),
                "1到3kHz能量占比": band_energy_ratio(freqs, power, 1000, 3000),
                "3到5kHz能量占比": band_energy_ratio(freqs, power, 3000, 5000),
                "一倍频能量占比": harmonic_energy_ratio(freqs, power, base_freq),
                "二倍频能量占比": harmonic_energy_ratio(freqs, power, 2 * base_freq),
                "三四倍频能量占比": harmonic_energy_ratio(freqs, power, 3.5 * base_freq, bandwidth_hz=base_freq),
            }
        )

    return pd.DataFrame(rows)


def save_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_wave_examples(windows: np.ndarray, cfg: ExperimentConfig, path: Path) -> None:
    count = min(cfg.max_example_windows, windows.shape[0])
    time_ms = np.arange(cfg.window_length) / cfg.sampling_frequency * 1000
    fig, axes = plt.subplots(count, 1, figsize=(11, 1.8 * count), sharex=True)
    if count == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time_ms, windows[i], linewidth=0.8)
        ax.set_ylabel(f"窗口{i + 1}\n幅值")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("时间 / ms")
    fig.suptitle(f"{cfg.channel} 通道真实振动窗口示例", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mean_spectrum(windows: np.ndarray, cfg: ExperimentConfig, path: Path) -> None:
    centered = windows - windows.mean(axis=1, keepdims=True)
    freqs = rfftfreq(cfg.window_length, d=1.0 / cfg.sampling_frequency)
    amp = np.abs(rfft(centered, axis=1)).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, amp, linewidth=1.0)
    ax.set_title(f"{cfg.channel} 通道平均频谱")
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("平均幅值")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, cfg.sampling_frequency / 2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_feature_histograms(features: pd.DataFrame, path: Path) -> None:
    selected = [
        "均方根_RMS",
        "峰值_Peak",
        "峰峰值_PeakToPeak",
        "峭度_Kurtosis",
        "峰值因子_CrestFactor",
        "频率重心_Hz",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), selected):
        ax.hist(features[col], bins=35, color="#4472C4", alpha=0.82, edgecolor="white")
        ax.set_title(col)
        ax.set_ylabel("窗口数量")
        ax.grid(alpha=0.2)
    fig.suptitle("公共评价特征分布")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_pca(features: pd.DataFrame, metadata: pd.DataFrame, path: Path, random_state: int) -> None:
    numeric = features.select_dtypes(include=[np.number])
    scaled = StandardScaler().fit_transform(numeric)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    file_names = metadata["源文件"].to_numpy()
    for file_name in sorted(metadata["源文件"].unique()):
        mask = file_names == file_name
        ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.72, label=file_name)

    ratio = pca.explained_variance_ratio_ * 100
    ax.set_title("真实窗口特征 PCA 分布（按源文件区分）")
    ax.set_xlabel(f"第一主成分 / {ratio[0]:.1f}%")
    ax.set_ylabel(f"第二主成分 / {ratio[1]:.1f}%")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    summary = features.describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary.rename(
        columns={
            "count": "数量",
            "mean": "均值",
            "std": "标准差",
            "min": "最小值",
            "25%": "25分位数",
            "50%": "中位数",
            "75%": "75分位数",
            "max": "最大值",
        }
    )
    summary.index.name = "特征名称"
    return summary


def write_report(
    cfg: ExperimentConfig,
    files: list[Path],
    windows: np.ndarray,
    metadata: pd.DataFrame,
    stats: dict[str, float],
    dirs: dict[str, Path],
) -> None:
    approx_duration = float(metadata.groupby("源文件")["结束时间_s"].max().sum())
    report = f"""# 实验0：公共预处理与评价框架

## 实验目的

本实验不训练生成模型，主要建立后续实验共同使用的数据入口和评价基准：

- 读取源 CSV 文件中的 `{cfg.channel}` 通道；
- 按窗口长度 `{cfg.window_length}`、滑动步长 `{cfg.step_size}` 切分真实振动窗口；
- 对窗口数据进行去均值和全局标准化；
- 提取 RMS、峰值、峭度、频率重心、频带能量等公共评价特征；
- 保存中文命名的预处理数据、特征表和基准图表。

## 数据与参数

- 源数据目录：`{cfg.source_dir.name}`
- CSV 文件数量：{len(files)}
- 使用通道：`{cfg.channel}`
- 采样频率：{cfg.sampling_frequency} Hz
- 主轴转速：{cfg.spindle_speed_rpm} rpm
- 窗口长度：{cfg.window_length} 点
- 滑动步长：{cfg.step_size} 点
- 是否去除每个窗口均值：{cfg.remove_window_mean}
- 生成窗口总数：{windows.shape[0]}
- 单个窗口时长：{cfg.window_length / cfg.sampling_frequency:.4f} s
- 全局标准化均值：{stats["全局均值"]:.8g}
- 全局标准化标准差：{stats["全局标准差"]:.8g}
- 按窗口统计得到的累计覆盖时长约：{approx_duration:.2f} s

## 输出文件说明

- `01_预处理数据/X-2_窗口数据_原始.npy`：未做全局标准化的窗口数据；
- `01_预处理数据/X-2_窗口数据_标准化.npy`：后续模型训练建议优先使用的标准化窗口数据；
- `01_预处理数据/X-2_窗口元信息.csv`：每个窗口来自哪个源文件、起止采样点和时间；
- `01_预处理数据/标准化参数.json`：把标准化数据还原到原始量纲所需的均值和标准差；
- `02_特征表/真实样本_公共评价特征.csv`：真实窗口的公共评价特征；
- `02_特征表/真实样本_特征统计摘要.csv`：各特征的均值、标准差和分位数；
- `03_评价图表/`：波形、频谱、特征分布和 PCA 基准图。

## 后续实验使用建议

实验 1 到实验 6 生成新样本后，应使用同一套特征提取逻辑计算生成样本特征，再与本实验保存的真实样本特征对比。这样可以避免不同实验之间因为预处理方式不同导致评价结果不可比。
"""
    (dirs["report"] / "实验0说明.md").write_text(report, encoding="utf-8")


def run(config_path: Path) -> None:
    cfg = load_config(config_path)
    np.random.seed(cfg.seed)
    setup_chinese_font()

    dirs = ensure_dirs(cfg.output_dir)
    files = list_csv_files(cfg.source_dir)
    windows, metadata = build_windows(files, cfg)
    standardized, stats = standardize_windows(windows)
    features = compute_features(windows, cfg.sampling_frequency, cfg.spindle_speed_rpm)
    features.insert(0, "样本编号", metadata["样本编号"].to_numpy())

    np.save(dirs["processed"] / "X-2_窗口数据_原始.npy", windows)
    np.save(dirs["processed"] / "X-2_窗口数据_标准化.npy", standardized)
    metadata.to_csv(dirs["processed"] / "X-2_窗口元信息.csv", index=False, encoding="utf-8-sig")
    save_json(
        dirs["processed"] / "窗口数据元信息.json",
        {
            "源数据目录": str(cfg.source_dir),
            "输出目录": str(cfg.output_dir),
            "CSV文件": [file.name for file in files],
            "使用通道": cfg.channel,
            "采样频率_Hz": cfg.sampling_frequency,
            "主轴转速_rpm": cfg.spindle_speed_rpm,
            "窗口长度": cfg.window_length,
            "滑动步长": cfg.step_size,
            "是否去除窗口均值": cfg.remove_window_mean,
            "窗口数量": int(windows.shape[0]),
        },
    )
    save_json(dirs["processed"] / "标准化参数.json", stats)

    feature_values = features.drop(columns=["样本编号"])
    summary = build_feature_summary(feature_values)
    features.to_csv(dirs["features"] / "真实样本_公共评价特征.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(dirs["features"] / "真实样本_特征统计摘要.csv", encoding="utf-8-sig")

    plot_wave_examples(windows, cfg, dirs["figures"] / "波形窗口示例.png")
    plot_mean_spectrum(windows, cfg, dirs["figures"] / "平均频谱.png")
    plot_feature_histograms(feature_values, dirs["figures"] / "公共评价特征分布.png")
    plot_pca(feature_values, metadata, dirs["figures"] / "PCA特征分布_按源文件.png", cfg.pca_random_state)
    write_report(cfg, files, windows, metadata, stats, dirs)

    print("实验0完成")
    print(f"输出目录：{cfg.output_dir}")
    print(f"CSV 文件数量：{len(files)}")
    print(f"窗口数量：{windows.shape[0]}")
    print(f"窗口形状：{windows.shape}")
    print(f"标准化前全局均值：{stats['全局均值']:.8g}")
    print(f"标准化前全局标准差：{stats['全局标准差']:.8g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验0：公共预处理与评价框架")
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
