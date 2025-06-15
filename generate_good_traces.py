import os
import subprocess
from pathlib import Path
import random

MODEL_PATH = "resources/uppaal_resources/thermostat.xml"
QUERY_PATH = "resources/uppaal_resources/thermostat.q"
VERIFYTA = "/Applications/UPPAAL-5.1.0-beta5.app/Contents/Resources/uppaal/bin/verifyta"
SAVE_FOLDER = Path("resources/good_traces")
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

def extract_trace_part(output: str) -> str:
    """保留从 't.ON:' 开始的部分（训练时 parse_data 用到）"""
    start = output.find("t.ON:")
    return output[start:] if start != -1 else ""

def generate_and_save_trace(i):
    seed = random.randint(0, 2**32 - 1)
    trace_path = SAVE_FOLDER / f"trace_{i}.txt"

    cmd = [
        VERIFYTA,
        MODEL_PATH,
        QUERY_PATH,
        "-s", "-S", "1", "-r", str(seed), "-t", "1", "-o", "1",
    ]

    print(f"Generating trace {i} with seed {seed}...")

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        output = proc.stdout

        if "-- Formula is satisfied." in output:
            trace_only = extract_trace_part(output)
            if not trace_only.strip():
                print(f"✘ Trace {i} skipped: No valid 't.ON:' section found.")
                return False

            with open(trace_path, "w") as f:
                f.write(trace_only)
            print(f"✔ Trace {i} saved at {trace_path}")
            return True
        else:
            print(f"✘ Trace {i} failed: formula not satisfied.")
            return False

    except Exception as e:
        print(f"✘ Trace {i} failed: {e}")
        return False

def generate_many(n):
    # 删除旧文件
    for f in SAVE_FOLDER.glob("trace_*.txt"):
        f.unlink()

    i = 0
    attempts = 0
    while i < n and attempts < 5 * n:
        success = generate_and_save_trace(i)
        if success:
            i += 1
        attempts += 1

if __name__ == "__main__":
    generate_many(800)
