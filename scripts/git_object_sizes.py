import subprocess
from pathlib import Path


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, cwd=Path(__file__).resolve().parents[1], errors="replace")


print(run(["git", "count-objects", "-vH"]))
print("---LARGEST---")
objects = run(["git", "rev-list", "--objects", "--all"]).splitlines()
rows = []
for line in objects:
    parts = line.split(" ", 1)
    sha = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    try:
        size = int(run(["git", "cat-file", "-s", sha]).strip())
    except Exception:
        continue
    rows.append((size, sha, path))

for size, sha, path in sorted(rows, reverse=True)[:20]:
    print(f"{size:>12} {sha} {path}")
