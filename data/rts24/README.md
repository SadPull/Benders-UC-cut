# RTS-24 核心输入

`inputs/` 保存当前 RTS-24 派生算例：32 台机组、24 小时、3 个场景、5% 备用及二次燃料费用。其原始字节与来源记录保持完整；求解核心读取输入不依赖实验目录。

历史独立 Gurobi 参考的 manifest 和 result 已移至 [v0.26.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.26.0) 清理前源码包，原路径为 `data/rts24/reference/`。参考方案成本 885066.704430；2026-10-13 轮次重新核验其开机矩阵后用于 UB，未导入参考整数下界。更早日志和源码仍在 v0.25.0 的实验包中。

默认经典 10 机组输入位于上级 `data/`，独立历史回归夹具保存在 `tests/fixtures/rts24/`。
