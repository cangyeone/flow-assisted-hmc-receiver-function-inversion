# 复现资料

本目录集中放置我们自己的 CPU 研究程序、模型、观测处理结果、实验数组、配置和来源记录。**不含第三方作者的反演程序，不自动下载他们的代码。** 第三方方法在参考文献中引用；如要运行原始实现，由使用者自行从作者获取，只导入计算结果。

1. 完整性检查：`python reproducibility/scripts/verify_release.py`。
2. 重绘真实数据图：`python reproducibility/scripts/build_field_figures.py`；输出在 `reproducibility/outputs/field-figures/`，不需要重跑反演。
3. CPU 真实数据反演：`flowrf invert reproducibility/data/processed/V20A.npz --initializer sobol --output outputs/V20A`。已有 EARS 为单频，不能作为三频条件流输入，不能通过生成网络补频带冒充观测。
4. 逐事件资料：`python reproducibility/scripts/download_observations.py --station V20A`；仅下载已记录的观测压缩包并检查原始 SHA-256，不执行包中内容。
5. 新 CPU 对比：`python reproducibility/scripts/benchmark_cpu.py`。同一物理目标下比较 Sobol 多起点与条件流辅助多起点，覆盖 HMC、我们独立实现的 DRAM、MWG，记录结构误差、波形拟合、区间与诊断。短链 smoke 不能用作精度或覆盖率结论。
6. CPU 训练：按英文说明依次运行 `python -m flowrf.training` 的 `generate/pilot/bootstrap/full` 四个阶段；`smoke.json` 检查执行，`million.json` 是完整规模。百万样本 CPU 训练耗时较长，发布版没有承诺训练加速。

历史实验数组保留了失败和未收敛情况。原实验可能使用不同硬件，历史时间不是新 CPU 接口的速度。`export-manifest.json` 分别保存原始哈希和发布哈希；改成公开依赖、去私人路径后的 CPU 脚本不应拿当前源码哈希强行匹配原记录。默认用户 API 也不能冒充所有历史参数化的逐位复现。

真实数据为 42 台站，保留 35 条通过局部诊断和 7 条未解决链；紧凑模型 32 个台站靠近边界，选定复杂模型均存在边界参数。地图不插值，拟合改善不代表发现新地质结构。详见 [English reproduction guide](README.md)。
