#!/usr/bin/env python3
import subprocess
import time
import csv
import os
import signal
from datetime import datetime
from pathlib import Path

# ==== Fixed parameters ====
DATASET_PATH = "/workspace/dataset_vript/gaming"
DATASET_NAME = "vript_gaming"
BATCH_SIZE_PER_GPU = "1"  # fixed
SEED = "0"

# If a run fails (non-zero return code), how many retries do we allow?
# Keep 0 for now (simple & predictable, overnight friendly). Set to 1 if you want an automatic retry.
RETRIES = 0

# Extra env for child processes (helps stability with NCCL)
ENV_EXTRA = {
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    # "TORCH_NCCL_BLOCKING_WAIT": "1",  # uncomment for debugging; can slow/lock runs
}

# ==== 3 chosen experiments ====
# name, lora_rank, scale, learning_rate, grad_accum_iter, epochs, max_iter (precomputed), global_batch (for reference)
EXPERIMENTS = [
    {
        "name": "E1_baseline_r16_b32_ep3",
        "lora_rank": "16",
        "scale": "2.0",
        "learning_rate": "1e-4",
        "grad_accum_iter": "4",   # global_batch = 8 * 4 = 32
        "epochs": 3,
        "max_iter": "1478",
        "global_batch": 32,
    },
    {
        "name": "E2_bigbatch_r8_b64_ep4",
        "lora_rank": "8",
        "scale": "1.5",
        "learning_rate": "1.5e-4",
        "grad_accum_iter": "8",   # global_batch = 8 * 8 = 64
        "epochs": 4,
        "max_iter": "985",
        "global_batch": 64,
    },
    {
        "name": "E5_perf_r24_b32_ep4",
        "lora_rank": "24",
        "scale": "2.0",
        "learning_rate": "1e-4",
        "grad_accum_iter": "4",   # global_batch = 8 * 4 = 32
        "epochs": 4,
        "max_iter": "1970",
        "global_batch": 32,
    },
    {
        "name": "E4_long_r16_b32_ep10",
        "lora_rank": "16",
        "scale": "2.0",
        "learning_rate": "1e-4",
        "grad_accum_iter": "4",   # global_batch = 8 * 4 = 32
        "epochs": 10,
        "max_iter": "5000",
        "global_batch": 32,
    },
]


def build_cmd(exp):
    return [
        "python", "my_scripts/posttrain_single.py",
        "--dataset_path", DATASET_PATH,
        "--dataset_name", DATASET_NAME,
        "--lora_rank", exp["lora_rank"],
        "--max_iter", exp["max_iter"],
        "--batch_size_per_gpu", BATCH_SIZE_PER_GPU,
        "--learning_rate", exp["learning_rate"],
        "--scale", exp["scale"],
        "--grad_accum_iter", exp["grad_accum_iter"],
        "--seed", SEED,
    ]


def run_once(exp, log_path: Path, attempt: int, env: dict):
    """Run one attempt of the experiment. Returns (rc, start_iso, end_iso, duration_sec)."""
    mode = "w" if attempt == 0 else "a"
    cmd = build_cmd(exp)

    start_ts = time.perf_counter()
    start_iso = datetime.now().isoformat(timespec="seconds")
    rc = None

    print(f"\n=== START {exp['name']} (attempt {attempt+1}) ===")
    print("Command:", " ".join(cmd))
    print(f"Log file: {log_path}")

    with open(log_path, mode, buffering=1) as lf:
        lf.write(f"# {exp['name']} attempt {attempt+1} started at {start_iso}\n")
        if attempt == 0:
            lf.write("# CMD: " + " ".join(cmd) + "\n\n")
        else:
            lf.write("# RETRY CMD: " + " ".join(cmd) + "\n\n")

        proc_env = os.environ.copy()
        proc_env.update(ENV_EXTRA)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=proc_env,
        )
        try:
            for line in process.stdout:
                print(line, end="")
                lf.write(line)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Sending SIGINT to the process...")
            try:
                process.send_signal(signal.SIGINT)
                rc = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("[WARN] Process didn't exit on SIGINT; terminating...")
                process.terminate()
                try:
                    rc = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print("[WARN] Forcing kill...")
                    process.kill()
                    rc = process.wait()
        finally:
            # If not interrupted path, ensure rc is set
            if rc is None:
                rc = process.wait()

    end_ts = time.perf_counter()
    end_iso = datetime.now().isoformat(timespec="seconds")
    duration_sec = end_ts - start_ts

    print(f"=== END {exp['name']} (attempt {attempt+1}, rc={rc}) ===")
    print(f"Start: {start_iso} | End: {end_iso} | Duration: {duration_sec:.1f}s\n")

    return rc, start_iso, end_iso, duration_sec


def run_stable(exp, log_dir: Path):
    """Run with retries and robust exception handling; never raises to caller."""
    log_path = log_dir / f"{exp['name']}.log"
    attempts = RETRIES + 1
    last_rc = None
    first_start = None
    last_end = None
    total_duration = 0.0

    for a in range(attempts):
        try:
            rc, s, e, dur = run_once(exp, log_path, a, os.environ.copy())
            last_rc = rc
            if first_start is None:
                first_start = s
            last_end = e
            total_duration += dur
            if rc == 0:
                break  # success
        except Exception as ex:
            # Catch-all to prevent the driver from aborting
            now = datetime.now().isoformat(timespec="seconds")
            with open(log_path, "a") as lf:
                lf.write(f"\n[DRIVER-EXCEPTION @ {now}] {type(ex).__name__}: {ex}\n")
            print(f"[ERROR] Driver exception on {exp['name']}: {ex}")
            last_rc = -999  # sentinel
            if first_start is None:
                first_start = now
            last_end = now
            # continue to next attempt if any

    return {
        "name": exp["name"],
        "lora_rank": exp["lora_rank"],
        "scale": exp["scale"],
        "learning_rate": exp["learning_rate"],
        "grad_accum_iter": exp["grad_accum_iter"],
        "global_batch": exp["global_batch"],
        "epochs": exp["epochs"],
        "max_iter": exp["max_iter"],
        "dataset_path": DATASET_PATH,
        "dataset_name": DATASET_NAME,
        "seed": SEED,
        "start_time": first_start,
        "end_time": last_end,
        "duration_sec": f"{total_duration:.1f}",
        "return_code": last_rc,
        "status": "ok" if last_rc == 0 else "error",
        "log_file": str(log_path),
        "retries_used": attempts - 1 if last_rc != 0 else (attempts - 1) - (RETRIES - (attempts - 1)),
    }


def write_csv(path: Path, fieldnames, rows):
    # Always rewrite the whole CSV and always include header
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs_cosmos_lora/{run_tag}")
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    fieldnames = [
        "name","lora_rank","scale","learning_rate","grad_accum_iter",
        "global_batch","epochs","max_iter",
        "dataset_path","dataset_name","seed",
        "start_time","end_time","duration_sec","return_code","status","log_file","retries_used"
    ]
    results = []

    for exp in EXPERIMENTS:
        res = run_stable(exp, log_dir)
        results.append(res)
        write_csv(summary_csv, fieldnames, results)  # rewrite each time to be robust against crashes
        print(f"[SUMMARY] {res['name']} -> rc={res['return_code']} ({res['status']}), duration={res['duration_sec']}s")

    print(f"\nSummary CSV: {summary_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
