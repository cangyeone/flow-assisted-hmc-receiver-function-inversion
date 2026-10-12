# 条件流辅助的多起点 HMC 接收函数反演

本项目以英文说明为主：[English README](README.md)。提供 CPU 正演、Transformer RF 生成、条件流候选生成、多起点物理优化和 HMC / DRAM / MWG 反演接口。

**仓库只发布我们自己的程序与模型，不发布第三方反演代码，也不包含 CUDA 程序。** 第三方方法仅引用文章；GRTM 通过独立社区安装包使用。复现材料集中在 `reproducibility/`，避免用户使用时被论文目录和历史实验干扰。

## 安装

可微物理正演使用 GRTM 社区 `v0.6.0.dev2` 预览版，目前支持 Apple Silicon macOS / Python 3.12 和 Linux x86_64 / Python 3.13（glibc≥2.28）。神经网络推理和已有结果绘图不依赖 GRTM；该预览暂无 Windows 物理求解器。

```bash
git clone https://github.com/cangyeone/flow-assisted-hmc-receiver-function-inversion.git
cd flow-assisted-hmc-receiver-function-inversion
python3.12 -m venv .venv  # Linux 使用 python3.13
source .venv/bin/activate
# Linux 先安装 CPU 版 PyTorch：
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[plot]'
flowrf install-grtm
flowrf doctor
flowrf demo --output outputs/demo
```

也可运行 `PYTHON_BIN=python3.12 ./install.sh`，Linux 改为 `python3.13`。演示使用短链，仅检查安装，不能用于声称收敛或精度优势。

## 接口

```python
from flowrf import RFData, RFInverter, NeuralModels, VelocityControls

data = RFData.load("my-rfs.npz")
# 单频数据等不符合神经模型输入时，用 initializer="sobol"。
result = RFInverter().invert(data, initializer="sobol", sampler="hmc")
result.save("outputs/station")
result.plot("outputs/station/rf-fit.png")
print(result.summary["status"])

# 符合 3 个慢度分组 × 3 个 Gaussian 频带的输入，可用全局条件流选起点：
# candidates = NeuralModels().propose(data, draws=256)
# result = RFInverter(VelocityControls(4)).invert(data, initializer="flow")
```

[输入格式与参数先验](docs/api.md)：深度 km、速度 km/s、密度 g/cm³、慢度 s/km，P 到时处 RF 必须归一化为 1。似然须排除该确定性归一化样点，并提供合理误差协方差。已有 EARS 单频数据可直接用 Sobol 初始化；不要生成缺失频带来冒充观测。

条件流只提供候选起点，进入物理模型族后进行优化；最终 HMC / DRAM / MWG 仍由物理正演、观测误差和明确先验决定。物理正演接受任意层数，发布条件流的训练支持为 2–16 个有限层；普通 HMC 每次运行的参数维度固定。

## 模型与复现

生产模型含两个完整版本的 RF Transformer 和条件流权重，默认 `global-v2`。早期 pilot/bootstrap、判别器和局部似然近似模型放在 `reproducibility/models/`，仅供实验检查，不作为默认约束。[模型说明](docs/models.md)明确记录训练范围、权重 SHA-256 和精度局限。

[复现说明](reproducibility/README.zh-CN.md)包含真实数据来源、42 台站数据、合成实验、消融与迭代预算结果、CPU 重跑方法及论文真实数据绘图。第三方 MCMC 程序不收录、不自动下载、不改写；如需运行作者实现，由使用者单独获取，再导入其输出作对比。

目前证据支持盆地情况下更好的初始化和更少优化工作，不能概括成所有情况更准确；无沉积层迁移失败、未收敛链和边界解均保留。真实数据拟合改善不等于独立地质精度或后验覆盖率得到证明。

## 依赖与许可

[GRTM 社区](https://github.com/cangyeone/seismicx-grtm-community)是物理求解器；[DNN-RFQC 社区](https://github.com/cangyeone/dnn-rfqc-community)是可选的接收函数质量筛选工具，两者分开安装。仓库没有复制它们的实现。

本项目自身代码、文档与发布的模型权重从 v0.1.1 起采用 [GNU GPL 第 3 版](LICENSE)（仅第 3 版，`GPL-3.0-only`），版权和适用范围见 [NOTICE](NOTICE)。已经发布的 v0.1.0 保留原 MIT 许可。外部 GRTM 保留其研究/非商业许可，观测数据与地图保留各自来源说明。完整物理依赖栈的使用仍受 GRTM 许可约束。软件引用见 [CITATION.cff](CITATION.cff)，算法和对比文章见 [references](docs/references.md)。
