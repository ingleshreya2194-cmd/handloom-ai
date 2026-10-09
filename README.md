# Dhaaga: AI Handloom & Craft Heritage Preserver
Flask + scikit-learn web app. Identifies 6 Indian crafts from an image, shows heritage info, saves pieces to an archive.

## Run locally
    pip install -r requirements.txt
    python ml.py          # trains and saves model.joblib
    python app.py         # http://localhost:5000

## Deploy on Render
1. Push this folder to a GitHub repo.
2. Render > New > Web Service > connect the repo (or New > Blueprint to use render.yaml).
3. Settings if asked: Build `pip install -r requirements.txt && python -c "import ml; ml.get_model()"`, Start `gunicorn app:app --workers 1 --threads 4 --timeout 120`, env var `PYTHON_VERSION=3.11.9`.
4. Deploy. Health check path: `/health`.

## Improve accuracy
Add real photos as `dataset/warli/*.jpg`, `dataset/ikat/*.jpg` etc., delete `model.joblib`, rebuild.
Free Render disks are temporary, so the archive resets on restart.
