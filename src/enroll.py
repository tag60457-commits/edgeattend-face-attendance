from __future__ import annotations

import argparse
import time

import cv2

from .recognize import FaceRecognizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Person name to enroll")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--shots", type=int, default=3)
    args = ap.parse_args()

    rec = FaceRecognizer()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    print("Press SPACE to capture a face. Press Q to quit.")
    captured = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow("Enroll", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key == 32:  # space
            emb = rec.embed_bgr(frame)
            if emb is None:
                print("No face detected. Try again.")
                continue
            tag = f"{int(time.time())}-{captured}"
            path = rec.db.add(args.name, emb, tag=tag)
            captured += 1
            print(f"Saved embedding: {path}")
            if captured >= args.shots:
                print("Enrollment complete.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
