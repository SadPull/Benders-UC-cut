# 纯 SA 分级诊断：固定协议初筛

源码提交：`5d6cb71d999ab3df0f7a129dc8d7f32d78272db6`。
3 个种子；24 小时；固定参数。初筛不代表统计显著性。完整原始结果见 summary.json。
写入失败或进程包络超时的试验保留已有样本，但对应三种子初筛判定为结论不足。统计展示所有按时保存的样本。

原始主问题可行率不包含辅助变量重优化或修复。NA 表示没有可行样本或参考未认证。

| 快照 | 种子策略 | 有可行解种子 | 主问题可行率中位数 | 误差中位数 | 最差误差 | 初筛 |
|---|---|---:|---:|---:|---:|---|
| n3-h1-first | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n3-h1-first | varying | 3/3 | 50.000% | 4.081% | 4.228% | not_passed |
| n3-h1-peak | fixed | 1/3 | 0.000% | 0.006% | 0.006% | not_passed |
| n3-h1-peak | varying | 3/3 | 24.688% | 0.006% | 0.006% | passed |
| n3-h2-first | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n3-h2-first | varying | 3/3 | 10.000% | 30.030% | 30.109% | not_passed |
| n3-h2-peak | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n3-h2-peak | varying | 2/3 | 5.000% | 0.002% | 0.002% | not_passed |
| n3-h3-first | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n3-h3-first | varying | 2/3 | 5.000% | 86.507% | 88.708% | not_passed |
| n3-h3-peak | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n3-h3-peak | varying | 2/3 | 5.000% | 0.001% | 0.001% | not_passed |
| n5-h1-first | fixed | 2/3 | 100.000% | 47.113% | 56.193% | not_passed |
| n5-h1-first | varying | 3/3 | 60.000% | 29.891% | 32.374% | not_passed |
| n5-h1-peak | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n5-h1-peak | varying | 3/3 | 30.000% | 0.722% | 4.730% | not_passed |
| n5-h2-first | fixed | 1/3 | 0.000% | 755.632% | 755.632% | not_passed |
| n5-h2-first | varying | 2/3 | 20.000% | 687.128% | 701.668% | not_passed |
| n5-h2-peak | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n5-h2-peak | varying | 2/3 | 5.000% | 9.517% | 10.208% | not_passed |
| n5-h3-first | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n5-h3-first | varying | 3/3 | 5.000% | 263.132% | 297.407% | not_passed |
| n5-h3-peak | fixed | 1/3 | 0.000% | 10.908% | 10.908% | not_passed |
| n5-h3-peak | varying | 3/3 | 10.000% | 10.908% | 11.404% | not_passed |
| n10-h1-first | fixed | 1/3 | 0.000% | 120.392% | 120.392% | not_passed |
| n10-h1-first | varying | 3/3 | 75.000% | 171.241% | 182.013% | not_passed |
| n10-h1-peak | fixed | 1/3 | 0.000% | 7.528% | 7.528% | not_passed |
| n10-h1-peak | varying | 3/3 | 15.000% | 8.634% | 10.217% | not_passed |
| n10-h2-first | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n10-h2-first | varying | 3/3 | 15.000% | 75.486% | 79.993% | not_passed |
| n10-h2-peak | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n10-h2-peak | varying | 1/3 | 0.000% | 14.972% | 14.972% | not_passed |
| n10-h3-first | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n10-h3-first | varying | 0/3 | 0.000% | NA | NA | not_passed |
| n10-h3-peak | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n10-h3-peak | varying | 3/3 | 15.000% | 46.930% | 49.697% | not_passed |
| n32-h1-first | fixed | 3/3 | 100.000% | 286.163% | 323.366% | not_passed |
| n32-h1-first | varying | 3/3 | 55.000% | 165.715% | 190.934% | not_passed |
| n32-h1-peak | fixed | 1/3 | 0.000% | 34.768% | 34.768% | not_passed |
| n32-h1-peak | varying | 3/3 | 20.000% | 30.623% | 32.073% | not_passed |
| n32-h2-first | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n32-h2-first | varying | 2/3 | 5.000% | 69.219% | 70.664% | not_passed |
| n32-h2-peak | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n32-h2-peak | varying | 0/3 | 0.000% | NA | NA | not_passed |
| n32-h3-first | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n32-h3-first | varying | 0/3 | 0.000% | NA | NA | not_passed |
| n32-h3-peak | fixed | 0/3 | 0.000% | NA | NA | not_passed |
| n32-h3-peak | varying | 0/3 | 0.000% | NA | NA | not_passed |

| 完整运行 | 状态 | SA及恢复成本改进 | 成本高于参考 | 原始上下界gap | 论文口径gap | 首轮完成块 | 乘子更新 |
|---|---|---:|---:|---:|---:|---:|---:|
| n3-det-h1-varying-7 | DUAL_CONVERGED | 10725.120 | -0.000% | 0.000% | 0.000% | 24/24 | 6 |
| n3-det-h1-varying-100007 | DUAL_CONVERGED | 10725.120 | -0.000% | 0.000% | 0.000% | 24/24 | 6 |
| n3-det-h1-varying-200007 | DUAL_CONVERGED | 10725.120 | -0.000% | 0.000% | 0.000% | 24/24 | 6 |
| n10-det-h1-varying-7 | INNER_NOT_CONVERGED | 0.000 | 15.720% | 14.795% | 1.400% | 0/24 | 0 |
| n10-det-h1-varying-100007 | INNER_NOT_CONVERGED | 0.000 | 15.720% | 14.795% | 1.400% | 0/24 | 0 |
| n10-det-h1-varying-200007 | INNER_NOT_CONVERGED | 0.000 | 15.720% | 14.795% | 1.400% | 0/24 | 0 |
| n32-det-h1-varying-7 | INNER_NOT_CONVERGED | 0.000 | 6.131% | 7.727% | 2.070% | 0/24 | 0 |
| n32-det-h1-varying-100007 | TIME_LIMIT | 0.000 | 6.131% | 7.727% | 2.070% | 0/24 | 0 |
| n32-det-h1-varying-200007 | INNER_NOT_CONVERGED | 0.000 | 6.131% | 7.727% | 2.070% | 0/24 | 0 |
| n32-det-h3-varying-7 | INNER_NOT_CONVERGED | 0.000 | 6.131% | 7.727% | 2.070% | 0/8 | 0 |
| n32-det-h3-varying-100007 | INNER_NOT_CONVERGED | 0.000 | 6.131% | 7.727% | 2.070% | 0/8 | 0 |
| n32-det-h3-varying-200007 | INNER_NOT_CONVERGED | 0.000 | 6.131% | 7.727% | 2.070% | 0/8 | 0 |
| n32-det-h3-fixed-7 | INNER_NOT_CONVERGED | 0.000 | 6.131% | 7.727% | 2.070% | 0/8 | 0 |
| n32-det-h3-fixed-100007 | INNER_NOT_CONVERGED | 0.000 | 6.131% | 7.727% | 2.070% | 0/8 | 0 |
| n32-det-h3-fixed-200007 | INNER_NOT_CONVERGED | 0.000 | 6.131% | 7.727% | 2.070% | 0/8 | 0 |
| n32-stoch-h3-varying-7 | INNER_NOT_CONVERGED | 0.000 | NA | 9.871% | NA | 0/8 | 0 |
| n32-stoch-h3-varying-100007 | INNER_NOT_CONVERGED | 0.000 | NA | 9.871% | NA | 0/8 | 0 |
| n32-stoch-h3-varying-200007 | INNER_NOT_CONVERGED | 0.000 | NA | 9.871% | NA | 0/8 | 0 |

原始离散可行、原始主问题可行、编码约束可行分别统计。QUBO经典参考具有数值限制，不能仅凭OPTIMAL宣称精确全局证书。
完整算法保留原有全日恢复；成本改进归为SA及恢复流程，初始化成本单列。DUAL_CONVERGED不表示原始问题最优。
