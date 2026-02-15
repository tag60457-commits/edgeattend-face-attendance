from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime

import cv2

from .recognize import FaceRecognizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--cooldown_s", type=int, default=30, help="Min seconds between re-logging same name")
    args = ap.parse_args()

    rec = FaceRecognizer()

    os.makedirs("data/logs", exist_ok=True)
    out_path = os.path.join("data/logs", f"attendance-{datetime.now().date().isoformat()}.csv")
    seen_at = {}

    file_exists = os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if not file_exists:
        w.writerow(["timestamp", "name", "score"])

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    print("Running attendance. Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        emb = rec.embed_bgr(frame)
        label = "No face"
        if emb is not None:
            m = rec.best_match(emb, threshold=args.threshold)
            if m is None:
                label = "Unknown"
            else:
                label = f"{m.name} ({m.score:.2f})"
                now = time.time()
                last = seen_at.get(m.name, 0)
                if now - last >= args.cooldown_s:
                    seen_at[m.name] = now
                    ts = datetime.now().isoformat(timespec="seconds")
                    w.writerow([ts, m.name, f"{m.score:.4f}"])
                    f.flush()

        cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("EdgeAttend", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    f.close()
    cv2.destroyAllWindows()
    print(f"Saved log: {out_path}")


if __name__ == "__main__":
    main()
