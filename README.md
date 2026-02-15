# EdgeAttend — Face Verification Attendance (Offline)

Offline face verification + attendance logger:
- Enroll a person (captures embeddings)
- Run attendance: recognizes faces via cosine similarity and writes CSV logs

## Setup

```bash
cd edgeattend-face-attendance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Enroll

```bash
python -m src.enroll --name Anand --shots 3
```

Press **SPACE** to capture a face embedding. Do 3 shots.

## Attendance

```bash
python -m src.attendance --threshold 0.6 --cooldown_s 30
```

Logs are written to:
- `data/logs/attendance-YYYY-MM-DD.csv`

## Notes
- Threshold depends on lighting/camera. Try 0.55–0.70.
- This is a demo; for real production use, add liveness/anti-spoofing.
