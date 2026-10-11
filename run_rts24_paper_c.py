"""Run one paper-style C experiment: SA integer masters and continuous Gurobi only."""
import argparse
from pathlib import Path

from benchmarks.rts24_paper_c import execute


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all', 'preflight', 'run', 'audit', 'report'), default='all')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--directory', type=Path, default=Path('results/rts24-paper-c-20261011'))
    parser.add_argument('--allow-low-memory', action='store_true',
                        help='Explicitly waive the 4 GiB gate; continue monitoring and record the waiver')
    args = parser.parse_args()
    raise SystemExit(execute(args.directory, args.phase, args.seed, args.allow_low_memory))
