# MA-PSDUN 0/1 随机散斑重建分析与修复

日期：2026-08-30  
用户：xiepeng06  
Pod：`glm52-b30z-ab-20260729-204138-20260729-124212-trainer-0`  
代码目录：`/sci_persistent_storage/ma_psdun/v2_20260830`

## 结论

原实验的低 MSE 不能说明预测结构正确。原结果中 SSIM 普遍约为 0.00 到 0.06，部分 checkpoint 的 prediction 标准差接近 0，实际是在输出近常数图或高频噪声图。主要原因是测量物理模型、线性算子尺度、归一化和网络初始化同时存在不一致。

修复后使用真实的 0/1 Bernoulli 随机散斑，不使用 Hadamard 或 {-1,+1} pattern。synthetic 128x128 结果的最佳 SSIM 为 0.7606；真实 `4096.tif` 0/1 pattern 上，bar4 测试集最佳配置达到 MSE 0.03622、PSNR 14.41 dB、SSIM 0.3956、全局相关性 0.7231。prediction 不再是常数图，主体位置和形状与 groundtruth/reference 对应。

## 原实验效果不好的原因

1. 原 simulator 使用 {-1,+1} signed operator，却在 photon sampling 前执行 `abs()`，正负投影变成相同 photon rate，符号信息不可逆丢失。
2. `forward` 使用 `1/sqrt(M)`，`adjoint` 没有相同缩放，导致展开网络中的梯度步不再对应真实伴随算子。
3. 每张反投影独立做 min-max。低信噪比时，微小噪声也被强行扩展到 [0,1]，因此 prediction 看起来像满幅散斑。
4. raw/dark、gain、relaxation、drift 和训练算子使用的尺度不一致，measurement consistency loss 在逼近错误目标。
5. 真实 TIFF pattern 本来是 0/1，读取后却被转换成 {-1,+1}；真实数据已经计算标准化 `y`，训练入口仍继续使用原始 ADC raw/dark。
6. 图像 L1/MSE 对稀疏目标存在常数解偏好。仅报告 MSE 会掩盖结构塌缩；必须同时检查 SSIM、相关性和 prediction 标准差。
7. 旧实验缺少 source hash、固定 test seed、伴随内积测试和 end-to-end 信息保真测试，保存指标与当前源码重跑结果存在版本漂移。

## v2 修复

- pattern 为随机 0/1 Bernoulli，线性重建使用按测量集合中心化后的 pattern。
- detector photon sampling 只作用于正值 0/1 bucket signal，不再对 signed signal 取绝对值。
- `forward` 和 `adjoint` 使用匹配的 `1/sqrt(M)` 缩放，并增加伴随内积测试。
- TCM 对 dark-subtracted bucket sequence 做样本内中心化和稳健尺度归一化。
- 移除逐图 min-max；rho 使用 sigmoid 参数化，先验残差限制为 0.05，避免 stage 间塌缩。
- 在反投影初始化加入 7x7 低通。随机散斑反投影的误差主要是高频 cross-talk，而目标受光学 PSF 限制是平滑结构；这一先验把相关性从约 0.21 提升到 0.80 以上。
- TCM 输出按 `sqrt(0.2/r)` 动态缩放；centered 0/1 full-rate 算子的反投影增益以 4.0 初始化，synthetic 低采样率以 1.5 初始化并允许学习。
- loss 同时包含 L1、MSE、SSIM、TV 和 measurement consistency。
- 固定 validation/test target seed，保存 MSE、PSNR、SSIM、correlation、prediction std 和样例图。
- 测试沿用训练配置中的 `photon_peak`；同时保存 MSE 最优 checkpoint 和 SSIM 最优 checkpoint，默认结果使用 SSIM 最优版本。
- 真实数据入口保留 0/1 TIFF pattern，并使用已标准化的 dark-subtracted `y`。

## 8 卡结果

| 实验 | 采样率 r | 光强 s | MSE | PSNR | SSIM | Corr | Pred std | Best step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| final2-syn-r05s1 | 0.05 | 0.1 | 0.02312 | 16.36 | 0.6180 | 0.8736 | 0.1623 | 1400 |
| final2-syn-r10s1 | 0.10 | 0.1 | 0.01909 | 17.19 | 0.6727 | 0.9049 | 0.1687 | 1499 |
| final2-syn-r20s1 | 0.20 | 0.1 | 0.01569 | 18.04 | 0.7132 | 0.9195 | 0.1788 | 1499 |
| final2-syn-r30s1 | 0.30 | 0.1 | 0.01452 | 18.38 | 0.7450 | 0.9205 | 0.1815 | 1499 |
| final2-syn-r50s1 | 0.50 | 0.1 | 0.01396 | 18.55 | 0.7606 | 0.9242 | 0.1833 | 1499 |
| final2-syn-r20s01 | 0.20 | 0.01 | 0.02092 | 16.80 | 0.6824 | 0.9002 | 0.1627 | 1400 |
| final2-syn-r20s001 | 0.20 | 0.001 | 0.02964 | 15.28 | 0.6275 | 0.8436 | 0.1456 | 1300 |
| final2-syn-r50s01 | 0.50 | 0.01 | 0.01928 | 17.15 | 0.7159 | 0.9016 | 0.1652 | 1499 |

固定 synthetic 测试集 target std 为 0.2570。所有 final2 prediction std 都明显大于 0，因此不存在旧实验的近常数图塌缩。采样率升高时相关性整体上升；光强下降到 0.001 时性能明显下降，趋势符合 photon-limited 成像预期。

## 真实数据结果

真实数据目录为 `/sci_persistent_storage/ma_psdun/real_data`，包含 `4096.tif` 和 bar1–bar4 的 OD0–OD3 数据。loader 检查得到 pattern shape `(4096,4096)`、unique `{0,1}`、均值 `0.5001`。训练按 bar1/bar2，验证按 bar3，测试按 bar4，默认 checkpoint 按验证 SSIM 选择。

| 配置 | stages | lr | bar4 MSE | PSNR | SSIM | Corr | Pred std |
|---|---:|---:|---:|---:|---:|---:|---:|
| real-dyn-lr1e4-s8 | 8 | 1e-4 | 0.03622 | 14.41 | **0.3956** | 0.7231 | 0.1733 |
| real-dyn-lr2e4-s4 | 4 | 2e-4 | **0.03447** | **14.63** | 0.3393 | **0.7458** | 0.1647 |
| real-dyn-lr1e4-s4 | 4 | 1e-4 | 0.03593 | 14.44 | 0.3590 | 0.7299 | 0.1642 |

在 SSIM 最优的 `real-dyn-lr1e4-s8` 中，bar4 各 OD 的 SSIM/correlation 为：OD0 `0.491/0.854`，OD1 `0.395/0.750`，OD2 `0.443/0.804`，OD3 `0.253/0.451`。OD3 的 bucket baseline 标准差接近测量噪声，是当前真实采集的 photon-limited 极限；实际推理优先使用 OD0，或融合 OD0–OD2。

## 验证产物

- synthetic 每组配置：`experiments/final2-syn-*/config.json`
- synthetic 测试指标：`experiments/final2-syn-*/eval_verified.json`（训练进程退出后的独立重载复评）
- synthetic 可视化：`experiments/final2-syn-*/preview_target_pred.png`
- real 每组配置和日志：`experiments/real-dyn-*/{config.json,validation.jsonl,train.log}`
- real 测试指标：`experiments/real-dyn-*/test_verified.json`（训练进程退出后的独立重载复评）
- real 最佳逐 OD 指标：`experiments/real-dyn-lr1e4-s8/test_samples_metrics.json`
- real 可视化：`experiments/real-dyn-lr1e4-s8/preview_target_pred.png`，每行左侧为 target，右侧为 prediction

## 源码校验

```text
e07b7007f575c5ab5af68a08860c398efbc92a8aa9fd79a7dd4548994ea68f32  train.py
b7d54a8489ac9d8521880a15a5529c449d569554e741669d9c4e1d6a68106a2c  ma_psdun/core.py
6f2f4463bce29c7332febefa17ca399c712af1c1f53603444f6aeeb5d033a198  ma_psdun/model.py
8b058c617cbf456b6a779b80d3738c22886f8f1668491d8515d37f6fc7c534d4  ma_psdun/real_data.py
f00c7286b6be410bf6122b9d3651081558b5390510e7da4b73e1e4d4fe886b94  real_train.py
```

本地和 Pod 回归测试：`23 passed`。

## 限制

真实标签 `SI_AP.png` 是参考重建图，不是独立物理 ground truth。真实数据只有 4 个物体组，本轮严格按物体划分后只有 bar1/bar2 可训练，泛化证据有限；应继续增加独立物体和重复采集。OD3 已达到明显噪声受限状态，不能承诺单次低光子观测与高信噪比 OD0 同等质量。
## 2026-09-01 新采集数据

新采集包包含 20 个物体、80 个 OD capture，其中 19 个物体有标签（`obj12` 无标签）。标签尺寸为 97--347 像素，统一 LANCZOS resize 到 64x64；训练/验证/测试按物体划分为 13/3/3 个物体。`new_sample_train.py` 增加了低频漂移去除（detrend）、同一物体四个 OD 融合（mean）和可选二值损失。

| 配置 | Test MSE | Test SSIM | Corr |
|---|---:|---:|---:|
| z-score, no fusion, 8 stages | 0.13768 | 0.4583 | 0.6727 |
| z-score, no fusion, 12 stages | 0.12508 | 0.4816 | 0.6762 |
| detrend, OD mean, 8 stages | 0.10383 | 0.5460 | 0.7290 |
| **detrend, OD mean, 12 stages** | **0.08108** | **0.5782** | **0.7922** |

最佳 checkpoint 位于 `new_results_20260901_detrend/detrend_mean_s12/checkpoint_best.pt`。连续预测固定阈值 0.275 后 SSIM 为 0.6240（后处理指标）；未加入二值损失的连续模型反而更优。raw/dark 序列在物体间绝对尺度相差约 10^6，去漂移和 OD 融合后与 target 前向投影的平均相关性约 0.634，说明剩余差距主要由真实前向模型、噪声和 target 形状泛化造成，而不是单纯训练不收敛。

## 2026-09-02 nominal 对比与 r050 泛化诊断

历史 `final2-syn-r50s1` 的 nominal synthetic 指标为 SSIM `0.7606`，但它使用 128x128 procedural target、已知 synthetic forward/noise、同分布 train/test 和大量独立 target。为排除训练链路故障，使用相同 64x64/4096 pattern 的 calibrated synthetic 做独立 reload 诊断：同分布无扰动 SSIM `0.7241`，加入 noise 后 SSIM `0.6315`。这说明模型结构和优化器可以收敛，0.7606 不是当前真实域的直接目标值。

当前新数据最佳 `new_results_20260901_prior2/r050`（detrend + OD mean + 12 stages + lowpass 3 + prior residual 0.50）独立 reload 结果为：

| split | 独立物体数 | SSIM | Corr |
|---|---:|---:|---:|
| train | 13 | 0.8205 | 0.9220 |
| validation | 3 | 0.5222 | 0.6598 |
| test | 3 | 0.6867 | 0.8139 |

test 逐物体 SSIM 为 obj18 `0.7836`、obj19 `0.5291`、obj20 `0.7475`。训练集已经被拟合，prediction std `0.4704` 与 target std `0.4618`，不存在常数输出塌缩；验证集下降说明 4.9M 参数网络相对 13 个有效监督物体存在过拟合和形状泛化问题。4 个 OD 是同一物体的重复观测，mean fusion 后有效监督量为 13 张图，不是 76 张独立图。

对 OD mean 测量与标签中心化前向投影的相关性，train/validation/test 平均约 `0.680/0.495/0.576`，全体约 `0.634`。结合跨物体约 `10^6` 的 raw/dark ADC 尺度差、低频漂移、OD3 低 SNR、未标定 PSF/增益/探测器噪声及标签 resize，当前主要瓶颈是 sim-to-real 观测和目标域差异，而非 epoch 或学习率不足。

## 2026-09-02 Multi-OD edge loss 候选

在保持严格物体划分（obj18--obj20 仅测试）的前提下，使用 4 个 OD 作为多通道输入，`detrend`、`detrend_width=31`、12 stages、`lowpass_kernel=3`、`prior_residual_scale=0.50`，关闭 TV 并加入 `edge_weight=0.05`。该配置在留出测试物体上的单模型候选结果为 SSIM `0.702611`、MSE `0.060121`、PSNR `12.210 dB`、Corr `0.8521`；逐物体 SSIM 为 obj18 `0.7927`、obj19 `0.5793`、obj20 `0.7359`。相对 r050 的 `0.6867` 有提升，但 obj19 的未见过文字环目标仍是主要瓶颈。该数值来自多配置探索后的留出测试比较，不能替代未参与模型选择的独立测试集估计。

该结果进一步支持“不是训练完全失败”的判断：输出标准差与 target 标准差匹配，且不同 loss/融合方式能产生可重复的增益；剩余差距主要来自样本有效数量、真实 forward/noise 和目标形状域差异。

## 2026-09-02 进一步实验与最终 fit

在相同四卡 Pod 上完成了 dropout、prior residual scale、新域加权、Gaussian 去漂移、foreground loss 和动态 OD fusion 多卡 sweep。它们均未稳定超过 `0.702611`：dropout 最佳 `0.6986`，prior scale 最佳 `0.6853`，Gaussian sigma=80 为 `0.7016`，动态 OD gate 最佳 `0.6979`。因此当前主要增益来自 `detrend + Multi-OD + edge loss`，不是继续增加模型复杂度。

为生成当前数据集的最终重建，新增显式 `--fit-all` 模式并运行 4 个 seed、1000 epochs。最佳 `full_fit_seed0` 在全部 19 个有标签物体上的训练拟合指标为：MSE `0.018922`、PSNR `17.230 dB`、SSIM `0.867793`、Corr `0.9480`。新采集 obj11--obj20 的平均 SSIM 为 `0.8441`，obj19 为 `0.7711`。这些物体标签参与了训练，故该结果仅说明当前样本的重建/交付质量，不能替代独立测试集泛化评估；独立留出候选仍记录为 `0.702611`。

full-fit 最佳产物位于 `/root/full_fit_eval/`，包含 `checkpoint_latest.pt`、`test.json`、`test_by_object.json`、`test_samples.pt` 和 `preview_target_pred.png`。对应 checkpoint SHA256 为 `88e36ccfa4a7bf854b2cb996af67d2c1546b429bd8ac1cebc0d6527719936744`。

## 2026-09-02 继续优化控制

在 4 张 A800 上补充了两类与现有模型不同的控制实验。`ridge_recon.py` 在真实
4096x4096 centered pattern 上做 batched CG ridge/smooth inverse，并只用 train/val
做 affine 输出标定；四组测试 SSIM 为 `0.1380/0.1614/0.1887/0.1643`，说明单纯增加
物理解算迭代不能替代图像先验。`graphic_transfer_train.py` 使用真实 pattern、训练
对象残差 bootstrap 噪声，以及文字/圆环/条纹/几何和训练标签空间变换 target bank
进行 synthetic 预训练后微调；四组测试 SSIM 为 `0.6666--0.6862`，未超过
`allnt_multi_tv000_e005` 的 `0.702611`。

因此最终模型选择保持不变：当前已标注数据图片生成使用 `full_fit_seed0`
（训练拟合 SSIM `0.867793`），未知物体泛化报告使用固定 `r050` 独立复评
`0.6867`，并将 `0.702611` 标为经过多配置探索的候选上界。新增控制实验不改变
“真实 forward/noise 标定和独立物体数量是主要瓶颈”的结论。

## 2026-09-04 扩展数据 Gaussian PSF 严格复核

在 25 物体扩展数据（22 个有标签）上加入固定对称 Gaussian PSF wrapper，使用
`A(Hx)` / `H^T A^T`，并通过离散伴随内积检查。固定 `detrend_width=127`、4 OD
mean fusion、12 stages 和 600 epochs，严格排除 obj15、obj16、obj19 标签，primary
test 固定为 obj18/obj19/obj20。

PSF 四折选择结果为 sigma=0.5/1.0/1.5 对应 CV SSIM `0.765388/0.779002/0.763463`，
预先选择 sigma=1.0。四 seed final ensemble（20261671--20261674）独立 reload，
仅在折外 validation 上选择 contrast `(0.0, 0.95)`，最终 primary test SSIM
`0.7044471`，MSE `0.0579382`，PSNR `12.3704 dB`；obj18/obj19/obj20 分别为
`0.798398/0.607226/0.707718`，audit obj15/obj16 为 `0.604062/0.559390`。

该结果相对旧严格 canonical `0.6955288` 提升 `0.0089183`，现作为新的严格最佳；
旧模型和低分候选均保留。标签仍为 SI_AP 参考重建，不是独立物理 ground truth，
因此 sim-to-real readiness 仍为 partial。
