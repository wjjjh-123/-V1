# 项目说明与协作规则

## 研究背景

用户是一名研究生，研究课题是“基于数据扩充的主轴系统精度退化建模方法研究”。

总体技术路线：

1. 在一台新的机床上采集主轴径向回转误差和振动信号数据。
2. 对采集数据进行数据扩充和融合处理。
3. 构建主轴精度模型：输入为加工中心运行时间 `t`，输出为该时刻主轴精度。
4. 计划连续采集约 30 天数据，再将模型外扩，拟合主轴精度退化模型。
5. 最后在一台已经运行很久的机床上验证模型可靠性：输入该机床运行时间 `t`，输出预测精度，并与实际测量精度对比。

用户英语基础较弱，生成图表、说明文字、注释和结果解释时应尽量使用中文；如果必须出现英文术语，需要给出中文解释。

用户目前主要使用 Python 编程，但还不能熟练使用。回答和代码应尽量清晰、可运行、少绕弯。

## 当前项目概况

当前文件夹是一个“主轴振动信号数据扩充”项目，主要围绕 `初始振动信号` 文件夹中的主轴空载振动 CSV 数据进行建模和生成样本。

已有 README 说明当前重点是：

- 数据对象：主轴空载振动信号。
- 使用通道：`X-2`。
- 数据文件数量：15 个 CSV 文件，文件名类似 `5000_1.csv` 到 `5000_15.csv`。
- 主轴转速：5000 rpm。
- 采样频率：10000 Hz。
- 每个文件有效采样点：60000。
- 窗口长度：2048。
- VAE 默认滑动步长：1024。
- VAE 样本数量：855 个窗口。
- WGAN-GP 使用更密的滑动步长：512。

## 主要目录

- `初始振动信号/`：原始振动信号 CSV 数据。
- `data/processed/`：预处理后的窗口数据，例如 `x2_windows.npz` 和元信息 `x2_windows_meta.json`。
- `data/generated/`：生成模型扩充出的振动窗口数据，包含 `.npy` 和 `.csv`。
- `checkpoints/`：训练好的模型权重、训练历史和统计信息。
- `results/`：评价结果、特征统计表和可视化图片。
- `results/comparison_plots/`：不同生成方法之间的对比图和误差表。
- `src/`：Python 源代码。
- `.idea/`：PyCharm 项目配置，说明用户可能使用 PyCharm。

## Python 环境与依赖

README 中记录的已验证解释器路径：

```powershell
D:/pycharm/anaconda/envs/pytorch_310/python.exe
```

主要依赖见 `requirements.txt`：

- `torch`：深度学习框架 PyTorch。
- `numpy`：数组和数值计算。
- `pandas`：CSV 和表格数据处理。
- `scipy`：信号处理和统计计算。
- `scikit-learn`：PCA、t-SNE、核函数等机器学习工具。
- `matplotlib`、`seaborn`：画图。
- `tqdm`：训练进度条。
- `pyyaml`：读取 `config.yaml` 配置。

写运行命令时优先使用 README 中的解释器路径，或者清楚说明可以换成当前 Python 环境。

## 配置文件重点

主要配置在 `config.yaml`：

- 随机种子：`42`。
- 原始数据目录：`初始振动信号`。
- 处理后数据目录：`data/processed`。
- 生成数据目录：`data/generated`。
- 使用列名：`X-2`。
- 采样频率：`10000` Hz。
- 主轴转速：`5000` rpm。
- 窗口长度：`2048`。
- VAE 滑动步长：`1024`。
- WGAN-GP 滑动步长：`512`。
- 是否去除每个窗口均值：`remove_window_mean: true`。

VAE 配置：

- 潜在维度 `latent_dim: 32`。
- `beta: 0.0005`。
- 训练轮数 `epochs: 120`。
- 批大小 `batch_size: 64`。
- 学习率 `learning_rate: 0.001`。
- 验证集比例 `val_ratio: 0.15`。

WGAN-GP 配置：

- 普通条件 WGAN-GP：`wgan_gp`。
- 残差条件 WGAN-GP：`residual_wgan_gp`。
- 二者都使用特征约束和梯度惩罚，输出文件名分别是：
  - `fc_wgan_gp_generated_x2_windows.npy`
  - `fcr_wgan_gp_generated_x2_windows.npy`

## 主要代码文件

- `src/config.py`：读取配置、处理项目路径。
- `src/data_utils.py`：读取 CSV、提取 `X-2` 列、滑动窗口切分、标准化、保存 `.npz`。
- `src/check_data.py`：检查并缓存窗口数据。
- `src/model_vae.py`：1D-CNN-VAE 模型和 VAE 损失函数。
- `src/train_vae.py`：训练 VAE，保存最佳模型和训练历史。
- `src/generate.py`：使用 VAE 生成扩充样本，支持 `residual`、`empirical`、`prior` 三种模式。
- `src/model_wgan_gp.py`：条件 WGAN-GP 生成器和判别器。
- `src/train_fc_wgan_gp.py`：训练条件 WGAN-GP。
- `src/train_fcr_wgan_gp.py`：训练残差条件 WGAN-GP。
- `src/generate_fc_wgan_gp.py`：用条件 WGAN-GP 生成样本。
- `src/generate_fcr_wgan_gp.py`：用残差条件 WGAN-GP 生成样本。
- `src/wgan_features.py`：提取 RMS、峰值、峰峰值、频率重心、频带能量、倍频能量等条件特征。
- `src/evaluate.py`：计算生成样本和真实样本的特征差异，并输出评价图和统计表。
- `src/plot_comparison.py`：对 Residual-VAE、FC-WGAN-GP、FCR-WGAN-GP 三种方法进行对比绘图。

## 已有模型与结果

`checkpoints/` 中已有模型文件：

- `vae_x2_best.pt`
- `fc_wgan_gp_x2_best.pt`
- `fcr_wgan_gp_x2_best.pt`

同时有训练历史 JSON 和 WGAN-GP 统计信息 JSON。

`data/generated/` 中已有生成样本：

- `vae_generated_x2_windows.npy` / `.csv`
- `fc_wgan_gp_generated_x2_windows.npy` / `.csv`
- `fcr_wgan_gp_generated_x2_windows.npy` / `.csv`

`results/` 中已有评价结果：

- 特征表：`real_x2_features.csv`、`generated_x2_features.csv`
- 特征分布摘要：`feature_distribution_summary.csv`
- MMD 距离：`mmd_summary.json`
- 波形图：`waveform_examples.png`
- RMS 分布图：`rms_distribution.png`
- 平均频谱图：`mean_spectrum.png`
- PCA 特征分布图：`pca_feature_distribution.png`
- 多方法对比图和误差表位于 `results/comparison_plots/`。

README 提醒：仓库里已有一次 1 轮训练的冒烟测试输出，只能说明代码链路可运行；论文实验结果应使用完整训练后的模型重新生成和评价。

## 常用运行流程

检查并缓存 X-2 窗口数据：

```powershell
& 'D:/pycharm/anaconda/envs/pytorch_310/python.exe' -m src.check_data --force
```

训练 VAE：

```powershell
& 'D:/pycharm/anaconda/envs/pytorch_310/python.exe' -m src.train_vae
```

使用 VAE 生成扩充样本：

```powershell
& 'D:/pycharm/anaconda/envs/pytorch_310/python.exe' -m src.generate --num-samples 1000
```

默认生成模式是 `residual`，含义是：保留真实健康振动窗口的基本形态，再叠加 VAE 潜在空间扰动产生的变化量。该模式更适合短周期健康振动数据扩充。

如果要使用普通 VAE 先验采样：

```powershell
& 'D:/pycharm/anaconda/envs/pytorch_310/python.exe' -m src.generate --num-samples 1000 --mode prior
```

评价生成样本质量：

```powershell
& 'D:/pycharm/anaconda/envs/pytorch_310/python.exe' -m src.evaluate
```

对比 VAE、条件 WGAN-GP、残差条件 WGAN-GP：

```powershell
& 'D:/pycharm/anaconda/envs/pytorch_310/python.exe' -m src.plot_comparison
```

## 评价指标与图表含义

当前代码会提取以下常见振动特征：

- `rms`：均方根值，反映振动能量大小。
- `peak`：峰值，反映最大瞬时幅值。
- `peak_to_peak`：峰峰值，即最大值与最小值之差。
- `skewness`：偏度，反映波形分布是否偏斜。
- `kurtosis`：峭度，反映冲击或尖峰程度。
- `crest_factor`：峰值因子，峰值除以 RMS。
- `freq_centroid`：频率重心，反映频谱能量集中位置。
- `band_0_1k`：0 到 1000 Hz 频带能量占比。
- `band_1_3k`：1000 到 3000 Hz 频带能量占比。
- `band_3_5k`：3000 到 5000 Hz 频带能量占比。
- `harmonic_1x`、`harmonic_2x`、`harmonic_3x4x`：一倍频、二倍频、三四倍频附近能量占比。

主要评价方式：

- 特征均值和标准差对比。
- 相对均值误差。
- JS 散度，用于比较特征分布差异。
- MMD 距离，用于比较真实样本和生成样本整体分布差异。
- PCA 和 t-SNE 图，用于观察真实样本与生成样本在特征空间中的重叠情况。
- 波形图、RMS 分布图、平均频谱图，用于直观看生成数据是否接近真实数据。

## 协作与写作要求

1. 尽量用中文解释代码、图表和结果。
2. 如果图表中有英文标签，应同时说明中文含义；后续修改图表时优先把坐标轴、图例、标题改成中文。
3. 用户 Python 不熟练，给命令时要完整，说明每一步输入后会得到什么输出。
4. 修改代码时保持现有项目结构，不随意大改目录。
5. 生成论文相关说明时，语言应偏学术但不要太生硬。
6. 涉及模型效果时，要区分“代码已跑通”和“论文可用实验结果”，不要把冒烟测试结果当成最终结论。
7. 做图时注意中文字体显示，避免中文乱码和负号显示错误。
8. 解释英文缩写时要补中文，例如 VAE 是变分自编码器，WGAN-GP 是带梯度惩罚的 Wasserstein 生成对抗网络，PCA 是主成分分析，MMD 是最大均值差异。

## 后续可能的研究衔接

当前项目主要完成的是振动信号层面的数据扩充。后续若要服务完整研究课题，还需要进一步把振动信号特征、径向回转误差和运行时间 `t` 融合起来，建立“运行时间 -> 主轴精度”的精度退化模型，并在长时间运行机床上验证预测精度。
