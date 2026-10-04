"""Independent local controller entry; logs survive a chat-turn interruption."""
import os
from pathlib import Path
import sys

root = Path(r"C:\Users\12433\Desktop\LLM-benders-SUC")
os.chdir(root)
sys.path.insert(0, str(root))
sys.stdout = (root / "results/minimal-resumed.stdout.log").open("a", encoding="utf-8", buffering=1)
sys.stderr = (root / "results/minimal-resumed.stderr.log").open("a", encoding="utf-8", buffering=1)
sys.argv = ["minimal_experiment", "--output-dir", str(root / "experiments/rts24-minimal-20261003"), "--resume"]
from suc_benders.minimal_experiment import main
main()
