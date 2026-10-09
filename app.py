import os, io, json, time, threading
from flask import Flask, jsonify, request, render_template, send_file, abort
from PIL import Image, UnidentifiedImageError
import ml

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
DATA = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
os.makedirs(DATA, exist_ok=True)
ARCHIVE = os.path.join(DATA, "archive.json"); LOCK = threading.Lock()

CRAFTS = {
 "warli": dict(name="Warli Painting", state="Maharashtra", region="Palghar & Dahanu belt", tech="Rice-paste white strokes on mud-and-cow-dung walls",
   motifs="Circles, triangles, squares: the sun, a tarpa dance, farming life", story="A tribal wall art of the Warli community, painted for weddings and harvests. Its geometry turns daily village life into a visual language.",
   care="Keep canvas pieces away from damp walls; clean with a dry soft cloth.", risk="Fewer young artists are learning the ritual form; most work now moves to canvas."),
 "madhubani": dict(name="Madhubani Painting", state="Bihar", region="Mithila region", tech="Natural pigments and bamboo-nib line work, double-line borders",
   motifs="Fish, peacocks, sun, lotus, wedding scenes; no empty space", story="Once painted on freshly plastered walls during weddings, Mithila art passed from mother to daughter for generations.",
   care="Avoid direct sun; frame under glass; dust gently.", risk="Cheap printed copies undercut hand-painted work."),
 "bandhani": dict(name="Bandhani Tie-Dye", state="Gujarat & Rajasthan", region="Kutch, Jamnagar, Jaipur", tech="Cloth is pinched and tied into tiny knots before dyeing; knots resist the colour",
   motifs="Dots in clusters: bindi, leheriya waves, diamond grids", story="One of India's oldest resist-dye crafts. Thousands of hand-tied knots make a single odhni or saree.",
   care="Dry clean or cold hand-wash separately; dry in shade to keep colours bright.", risk="Machine-printed 'bandhani' is replacing hand-tied pieces."),
 "ikat": dict(name="Ikat / Pochampally", state="Telangana, Odisha, Gujarat", region="Pochampally, Sambalpur, Patan", tech="Yarn is resist-dyed before weaving, giving soft feathered edges",
   motifs="Diamonds, zigzags, stripes, temple borders", story="The pattern is planned in the yarn itself, so weavers must align threads on the loom with great precision.",
   care="Wash gently in cold water; iron on reverse; store folded in cotton.", risk="Power-loom imitations and falling weaver incomes."),
 "kalamkari": dict(name="Kalamkari", state="Andhra Pradesh", region="Srikalahasti & Machilipatnam", tech="Hand-drawn with a bamboo pen or block-printed using natural dyes (indigo, madder, myrobalan)",
   motifs="Vines, flowers, epic and temple scenes in earthy tones", story="The word means 'pen work'. Storytellers once carried painted cloth scrolls from village to village.",
   care="Natural dyes bleed; dry clean or wash by hand in cold water.", risk="Synthetic dyes and digital prints are cutting demand for natural-dye work."),
 "banarasi": dict(name="Banarasi Brocade", state="Uttar Pradesh", region="Varanasi", tech="Silk woven with gold and silver zari threads on jacquard or handloom",
   motifs="Floral jaal, kalga/bel, paisley, dense gold borders", story="Banarasi weaving flourished under Mughal patronage, blending Persian and Indian design into heavy silk brocade.",
   care="Dry clean only; wrap in muslin; refold every few months.", risk="Power-loom 'Banarasi' sold at low price squeezes handloom weavers."),
}

def load():
    try:
        with open(ARCHIVE) as f: return json.load(f)
    except Exception: return []

@app.route("/")
def home(): return render_template("index.html")

@app.route("/health")
def health(): return "ok"

@app.route("/api/crafts")
def crafts(): return jsonify(CRAFTS)

@app.route("/sample/<key>/<int:i>.png")
def sample(key, i):
    if key not in ml.GEN or i > 50: abort(404)
    b = io.BytesIO(); ml.sample_image(key, i).save(b, "PNG"); b.seek(0)
    return send_file(b, mimetype="image/png", max_age=86400)

@app.route("/api/classify", methods=["POST"])
def classify():
    f = request.files.get("image")
    if not f: return jsonify(error="Choose an image first."), 400
    try:
        im = Image.open(f.stream); im.load(); im = im.convert("RGB"); im.thumbnail((800, 800))
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error="That file isn't a readable image. Try a JPG or PNG."), 400
    return jsonify(ml.analyze(im))

@app.route("/api/archive", methods=["GET", "POST"])
def archive():
    if request.method == "GET": return jsonify(load()[::-1][:60])
    d = request.get_json(silent=True) or {}
    if d.get("key") not in CRAFTS: return jsonify(error="Unknown craft."), 400
    thumb = str(d.get("thumb", ""))
    if not thumb.startswith("data:image/") or len(thumb) > 60000: thumb = ""
    rec = dict(key=d["key"], conf=round(float(d.get("conf", 0)), 3), place=str(d.get("place", ""))[:80],
               note=str(d.get("note", ""))[:300], thumb=thumb, t=int(time.time()))
    with LOCK:
        rows = load(); rows.append(rec)
        with open(ARCHIVE, "w") as fh: json.dump(rows[-300:], fh)
    return jsonify(ok=True)

@app.errorhandler(413)
def big(e): return jsonify(error="Image is over 8 MB. Choose a smaller one."), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
