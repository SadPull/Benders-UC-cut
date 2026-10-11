"""Run only the approved RTS24 C/D experiment, with independent preflight and audits."""
import argparse
from pathlib import Path

from benchmarks.rts24_improved import execute

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all', 'preflight', 'run', 'audit', 'report'), default='all')
    parser.add_argument('--arm', choices=('ALL', 'C', 'D'), default='ALL')
    parser.add_argument('--seed', type=int, choices=(7, 11, 19))
    parser.add_argument('--directory', type=Path, default=Path('results/rts24-improved-20261011'))
    args = parser.parse_args()
    raise SystemExit(execute(args.directory, args.phase, args.arm, args.seed))
