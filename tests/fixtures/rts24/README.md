# RTS-24 固定启动费用测试夹具

此处的 `units.json`、`deterministic.json`、`stochastic.json` 和 `provenance.json` 原样保留默认算例替换前的数据：32 台机组、24 小时、装机 3405 MW、峰负荷 2850 MW。来源版本、哈希及字段映射见 `provenance.json`。

`tests/test_core.py::test_rts_model_reference` 继续核对 MATLAB 提交 `aacef8a6d49ce126a28f70a694c4013f9a1c3b24` 的模型指纹、连续松弛和固定方案成本。此夹具不包含冷热启动字段，检验旧固定启动费用行为。

恢复原 RTS-24 记录时，将 `run_suc` 的 `data_file` 与 `scenario_file` 指向此目录内相应文件。当前 `data/` 默认使用 Kazarlis 经典 10 机组算例。
