from __future__ import annotations

import re
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "translategemma_download.log"

PROGRESS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| "
    r"progress=(?P<progress>[^ ]+) downloaded=(?P<downloaded>[0-9.]+) GB speed=(?P<speed>[0-9.]+) MB/s eta=(?P<eta>.+)$"
)


def main() -> None:
    if not LOG_PATH.exists():
        print(f"log_missing: {LOG_PATH}")
        return

    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    points: list[tuple[str, float, float]] = []
    for line in lines:
        match = PROGRESS_RE.match(line)
        if not match:
            continue
        points.append(
            (
                match.group("ts"),
                float(match.group("downloaded")),
                float(match.group("speed")),
            )
        )

    if len(points) < 5:
        print("not_enough_progress_samples")
        return

    dips: list[str] = []
    nonzero_speeds: list[float] = []
    stall_streak = 0

    for idx, (ts, downloaded, speed) in enumerate(points):
        prev_downloaded = points[idx - 1][1] if idx > 0 else downloaded
        delta = downloaded - prev_downloaded

        if speed <= 0.01 and delta <= 0.0001:
            stall_streak += 1
            if stall_streak >= 2:
                dips.append(
                    f"{ts} | stall | speed={speed:.2f} MB/s | downloaded={downloaded:.3f} GB | "
                    "reason=No new bytes for multiple intervals. Likely shard switch, timeout retry, or network stall."
                )
            continue

        stall_streak = 0
        if speed > 0.05:
            nonzero_speeds.append(speed)
            if len(nonzero_speeds) > 12:
                nonzero_speeds = nonzero_speeds[-12:]

        if len(nonzero_speeds) >= 4:
            baseline = sum(nonzero_speeds[-4:]) / 4
            ratio = speed / baseline if baseline > 0 else 1.0
            if baseline >= 0.5 and speed > 0.01 and ratio < 0.35:
                dips.append(
                    f"{ts} | throughput_drop | speed={speed:.2f} MB/s baseline={baseline:.2f} MB/s ratio={ratio:.2f} | "
                    "reason=Sharp drop versus recent baseline. Likely congestion, route switch, packet loss, or worker contention."
                )

    if not dips:
        print("no_major_dips_detected")
        return

    print("detected_dips:")
    for item in dips:
        print(item)


if __name__ == "__main__":
    main()
