"""Run the approved SMS100 first round in its isolated Python environment."""
import argparse
from pathlib import Path

from benchmarks.azema.controller import execute
from benchmarks.azema.protocol import Protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all', 'preflight', 'run', 'audit', 'report'), default='all')
    parser.add_argument('--arm', choices=('ALL', 'A', 'B', 'C', 'D'), default='ALL')
    parser.add_argument('--base-file', default='100_0_1_w.nc4')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--time-limit', type=float, default=3600.)
    parser.add_argument('--directory', type=Path, default=Path('results/sms100-first-round-20261010'))
    args = parser.parse_args()
    protocol = Protocol(base_file=args.base_file, batch=args.batch, seed=args.seed, time_limit=args.time_limit)
    protocol.validate()
    return execute(args.directory.resolve(), protocol, args.phase, args.arm)


if __name__ == '__main__':
    raise SystemExit(main())
