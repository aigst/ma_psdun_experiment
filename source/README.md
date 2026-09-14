# MA-PSDUN v2 复现实验包

本目录是 2026-08-30 在指定 8-GPU Pod 中完成的 MA-PSDUN v2 代码快照。目标是复现两条链路：

1. synthetic：结构化光学目标 -> 0/1 Bernoulli 随机散斑 bucket 测量 -> MAPSDUN 重建。
2. real：真实 0/1 DMD pattern 与 dark/bright bucket 序列 -> MAPSDUN 重建，使用 `SI_AP.png` 作为参考标签。

本包不包含模型权重、`real_data` 原始数据、kubeconfig、ServiceAccount token 或其他凭据。将数据放在本地后即可按下面命令重跑。

## 模型结构

`MAPSDUN` 是一个 model-based unrolled network，共有五个部分：

```text
raw,dark
  |
  v
TCM: 1-D temporal correction/normalization
  |
  v
centered 0/1 MeasurementOperator.adjoint + learnable gain + 7x7 low-pass
  |
  v
repeat K stages:
    residual = A(x) - y
    x <- x - rho_k * A^T(residual)
    x <- conditional residual U-Net prior(x, condition)
  |
  v
reconstructed image in [0,1]
```

### 1. 物理测量算子

`ma_psdun/core.py:MeasurementOperator` 生成真实 `0/1` Bernoulli pattern，尺寸为 `M x N`。计算重建时对 pattern 按测量集合做中心化：

```text
A(x)   = x @ centered_patterns.T / sqrt(M)
A^T(y) = y @ centered_patterns   / sqrt(M)
```

`forward_raw()` 只用于正值 detector bucket signal；不会对 signed measurement 使用 `abs()`。`forward()` 和 `adjoint()` 共享相同缩放，并由测试验证伴随内积关系。

### 2. TCM 测量校正

`ma_psdun/model.py:TCM` 是三层 `Conv1d` 网络，输入 raw/dark 两条测量序列。它先计算 `baseline = raw - dark`，再在 baseline 周围学习有界残差，最大残差为 `0.15 * (|baseline| + 0.05)`，输出乘以固定参考尺度 `0.35`。这样 TCM 不能凭空制造远大于观测信号的测量。

### 3. 反投影初始化

TCM 输出按 `sqrt(0.2/r)` 适配采样率 `r`，然后做伴随反投影。synthetic 的可学习增益初始化为 `1.5`，real full-rate 初始化为 `4.0`。随机散斑反投影存在高频 cross-talk，因此加入固定 `7x7` 平均池化；没有逐图 min-max，避免噪声被放大到整个 `[0,1]`。

### 4. 展开数据一致性阶段

每个 stage 使用一个可学习步长 `rho_k = 0.5 * sigmoid(rho_logits[k])`，先执行一次物理梯度式更新，再执行一个条件先验网络。默认 synthetic 为 6 stages，real sweep 使用 4 或 8 stages。

### 5. 条件残差 U-Net prior

`Prior` 是两层 encoder/decoder：`1 -> 32 -> 64` 下采样，在 bottleneck 拼接 32 维条件编码，再上采样并和浅层特征 skip connection 融合。输出是受限残差：

```text
correction = 0.05 * tanh(head(...))
output = clamp(x + correction, 0, 1)
```

零初始化输出层使每个 prior 初始接近 identity，避免早期噪声把图像推入全零或全一饱和状态。

条件编码为 `[sampling_rate, intensity, wavelength]`，分别使用 `logit(r)`、`log10(s)` 和 `(wavelength-1050)/650` 后输入 MLP。

### 6. 训练目标

`loss_fn()` 同时优化 image L1、image MSE、7x7 SSIM、measurement consistency 和 TV 平滑项：

```text
L = L1 + 0.5*MSE + 0.2*(1-SSIM) + 0.05*consistency + 0.01*TV
```

checkpoint 默认按 validation SSIM 选择，同时保留 validation MSE 最优 checkpoint。评估固定 test seed，并记录 MSE、PSNR、SSIM、相关性、prediction std。

## 环境

建议 Python 3.10+、PyTorch 2.x、NumPy、Pillow。最小依赖见 `requirements.txt`。在 GPU Pod 中运行时，`torch.cuda.is_available()` 会自动选择 CUDA；CPU 仅适合小尺寸 smoke test。

```bash
python3 -m pip install -r requirements.txt
pytest -q
```

## Synthetic 单实验

下面命令会生成固定的结构化目标、0/1 随机散斑和 photon/readout 噪声，并训练一个实验目录：

```bash
python3 train.py \
  --exp-dir experiments/syn-r50s1 \
  --size 128 \
  --sampling-rate 0.50 \
  --intensity 0.1 \
  --stages 6 \
  --steps 1500 \
  --batch-size 16 \
  --seed 123 \
  --operator-seed 123 \
  --photon-peak 200 \
  --device cuda:0
```

## Synthetic 8-GPU sweep

`scripts/run_synthetic_sweep_8gpu.sh` 将 8 个独立实验分别绑定到 8 张 GPU；这不是 DDP，而是可直接比较的独立配置 sweep：

```bash
bash scripts/run_synthetic_sweep_8gpu.sh experiments/final2-syn
```

脚本默认覆盖采样率 `0.05/0.10/0.20/0.30/0.50` 与光强 `0.1/0.01/0.001` 的代表性组合。每个实验会保存 config、validation 日志、checkpoint、eval 指标和样例 tensor。

## Real 数据

真实数据目录应具有如下结构（本包不含原始数据）：

```text
real_data/
  4096.tif
  bar1-20260830/OD0/traindata.txt
  bar1-20260830/SI/SI_AP.png
  ...
  bar4-20260830/OD3/traindata.txt
```

运行：

```bash
python3 real_train.py \
  --data-root /path/to/real_data \
  --patterns /path/to/real_data/4096.tif \
  --exp-dir experiments/real-dyn-lr1e4-s8 \
  --stages 8 \
  --lr 1e-4 \
  --epochs 300 \
  --device cuda:0
```

数据划分固定为 `bar1 + bar2` train、`bar3` validation、`bar4` test。loader 保留真实 `0/1` TIFF，并把每个 dark/bright pair 变成标准化 dark-subtracted `y`。真实 `SI_AP.png` 是参考重建标签，不是独立物理 groundtruth。

## 当前结果快照

synthetic 最佳配置 `final2-syn-r50s1`：采样率 `0.50`、光强 `0.1`，MSE `0.013956`、PSNR `18.552 dB`、SSIM `0.7606`、Corr `0.9243`。图片 `artifacts/final2-syn-r50s1-preview.png` 每行左侧为 target，右侧为 prediction。

真实数据 SSIM 最优配置 `real-dyn-lr1e4-s8`：bar4 MSE `0.03622`、PSNR `14.41 dB`、SSIM `0.3956`、Corr `0.7231`。真实低光 OD3 受 photon noise 限制明显。

新采集数据（`2026-09-01`）的多 OD 训练入口为 `new_sample_train.py`。严格按物体留出的最佳候选为 SSIM `0.702611`（obj18/19/20 分别为 `0.7927/0.5793/0.7359`）。若需要生成当前 19 个已标注物体的最终重建，可显式使用 `--fit-all`；该模式的最佳训练拟合 SSIM 为 `0.867793`，不能作为未知物体泛化分数。`--adaptive-fusion-scale`、`--od-dropout`、`--train-objects`、`--repeat-new` 和 `--foreground-weight` 用于实验性消融。

full-fit 示例：

```bash
python3 new_sample_train.py \
  --data-root /path/to/sample \
  --patterns /path/to/4096.tif \
  --exp-dir experiments/full-fit \
  --preprocess detrend --detrend-width 31 --fusion multi \
  --label-resample bilinear --stages 12 --lowpass-kernel 3 \
  --prior-residual-scale 0.50 --edge-weight 0.05 \
  --fit-all --epochs 1000 --device cuda:0
```

完整分析见 `analysis_report.md`，结果表见 `results/`。

## 2026-09-02 additional controls

`ridge_recon.py` is a classical batched CG control on the measured 4096x4096
centered operator. It solves a ridge/smooth inverse and calibrates only on the
train/validation objects. Four multi-GPU settings reached test SSIM
`0.1380--0.1887`, so the learned prior is necessary and the issue is not only
insufficient unrolled iterations.

`graphic_transfer_train.py` performs synthetic pretraining with real patterns,
train-object residual bootstrap noise, and graphic/label-augmentation targets,
then fine-tunes on real objects. Four A800 runs reached test SSIM
`0.6666--0.6862`; the existing strict candidate `0.702611` remains preferred.
These results use the strict object split: train has 13 independent objects,
validation has 3, and test has 3.

## 源码校验

以下 SHA256 与 Pod 中运行版本一致：

```text
e07b7007f575c5ab5af68a08860c398efbc92a8aa9fd79a7dd4548994ea68f32  train.py
b7d54a8489ac9d8521880a15a5529c449d569554e741669d9c4e1d6a68106a2c  ma_psdun/core.py
6f2f4463bce29c7332febefa17ca399c712af1c1f53603444f6aeeb5d033a198  ma_psdun/model.py
8b058c617cbf456b6a779b80d3738c22886f8f1668491d8515d37f6fc7c534d4  ma_psdun/real_data.py
f00c7286b6be410bf6122b9d3651081558b5390510e7da4b73e1e4d4fe886b94  real_train.py
```

测试结果：`23 passed`。
