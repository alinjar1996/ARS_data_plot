"""One-shot queue for the November 23 tray and ball-lift sweeps."""
import os
import select
import subprocess
from datetime import datetime
from pathlib import Path


def log(message):
    print(datetime.now().astimezone().isoformat(), message, flush=True)


def wait_for_tray():
    pid = 3429475
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return
        if stat.rsplit(")", 1)[1].split()[19] == "43993887":
            log(f"Queued behind tray sweep PID {pid}")
            poll = select.poll()
            poll.register(fd, select.POLLIN)
            poll.poll()
    finally:
        os.close(fd)


wait_for_tray()
log("Tray launcher exited. Starting matched ball-lift ARS sweep.")
result = subprocess.run(["bash", "real_demo/real_demo/run_eval_ht_bayes_ball_lift_100.sh", "--normalized-ars"])
log(f"Ball-lift launcher exited with status {result.returncode}")
raise SystemExit(result.returncode)
