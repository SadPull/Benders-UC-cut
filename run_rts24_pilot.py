"""Run the fixed 600-second RTS-24 full-method pilot; Codex dispatches proposal agents."""
import argparse
from pathlib import Path

from benchmarks.rts24 import execute


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all', 'preflight', 'run', 'audit', 'report'), default='all')
    parser.add_argument('--directory', type=Path, default=Path('results/rts24-pilot-20261011'))
    args = parser.parse_args()
    raise SystemExit(execute(args.directory, args.phase))
