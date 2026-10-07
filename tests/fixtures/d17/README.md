# D/17 independent reference

These four Python files are unmodified source fixtures from commit
`1d590f8a920c2b9b77d9608071409776e8a24271`, under
`experiments/reproduction-c17-d17-20261007/rts24-reproduction-D17-20261007/snapshot/suc_benders/`.

They are used only by `test_legacy.py` to compare repair, quantization and QUBO
energy against the frozen implementation. The test installs a private import
namespace and supplies the data type annotation; production code never imports
these fixtures. No historical results, credentials or experiment controller are
included. Line-ending normalization does not change the reference algorithms.
