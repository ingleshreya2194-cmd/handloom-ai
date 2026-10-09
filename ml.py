"""Feature extraction + synthetic pattern generators + Random Forest classifier."""
import os, random, joblib, numpy as np
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

S = 128
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.joblib")
KEYS = ["warli", "madhubani", "bandhani", "ikat", "kalamkari", "banarasi"]

def J(c, r, v=18):
    return tuple(int(min(255, max(0, x + r.randint(-v, v)))) for x in c)

def warli(r):
    im = Image.new("RGB", (S, S), J((150, 72, 44), r)); d = ImageDraw.Draw(im); w = J((245, 240, 230), r, 8)
    for _ in range(r.randint(6, 10)):
        x, y, s = r.randint(14, S - 14), r.randint(12, S - 34), r.randint(5, 9)
        d.ellipse([x - 3, y - 7, x + 3, y - 1], outline=w)
        d.polygon([(x - s, y), (x + s, y), (x, y + s)], fill=w)
        d.polygon([(x - s, y + 2 * s), (x + s, y + 2 * s), (x, y + s)], fill=w)
        d.line([(x - s, y + 2 * s), (x - s - 2, y + 3 * s)], fill=w); d.line([(x + s, y + 2 * s), (x + s + 2, y + 3 * s)], fill=w)
    for x in range(0, S, 8): d.line([(x, 2), (x + 4, 7), (x + 8, 2)], fill=w)
    return im

def madhubani(r):
    im = Image.new("RGB", (S, S), J((245, 225, 180), r, 10)); d = ImageDraw.Draw(im)
    pal = [(220, 40, 50), (250, 190, 30), (40, 150, 70), (30, 90, 190), (240, 110, 30), (200, 60, 150)]
    for _ in range(r.randint(14, 22)):
        x, y, a, b = r.randint(8, S - 24), r.randint(8, S - 18), r.randint(10, 24), r.randint(6, 12)
        d.ellipse([x, y, x + a, y + b], fill=J(r.choice(pal), r, 12), outline=(0, 0, 0), width=2)
        d.ellipse([x + 3, y + b // 2 - 1, x + 6, y + b // 2 + 2], fill=(0, 0, 0))
    d.rectangle([2, 2, S - 3, S - 3], outline=(0, 0, 0), width=3); d.rectangle([8, 8, S - 9, S - 9], outline=(0, 0, 0), width=1)
    return im

def bandhani(r):
    im = Image.new("RGB", (S, S), J(r.choice([(190, 25, 45), (235, 180, 30), (30, 130, 70), (220, 70, 140)]), r)); d = ImageDraw.Draw(im)
    c = J((250, 245, 225), r, 8); o = r.randint(0, 4)
    for gy in range(6 + o, S, 12):
        for gx in range(6 + o, S, 12):
            for dx, dy in ((0, 0), (3, 0), (-3, 0), (0, 3), (0, -3)):
                d.ellipse([gx + dx - 1, gy + dy - 1, gx + dx + 1, gy + dy + 1], fill=c)
    return im

def ikat(r):
    base = r.choice([(30, 40, 110), (150, 30, 40)]); im = Image.new("RGB", (S, S), J(base, r)); d = ImageDraw.Draw(im)
    c = r.choice([(240, 235, 220), (220, 160, 40)]); st = r.randint(14, 22)
    for x in range(0, S, st):
        pts = [(x + (6 if (y // 8) % 2 else -6), y) for y in range(0, S + 8, 8)]
        d.line(pts, fill=c, width=r.randint(3, 6))
    a = np.asarray(im, float); a = sum(np.roll(a, k, axis=1) for k in range(-3, 4)) / 7
    return Image.fromarray(a.astype("uint8"))

def kalamkari(r):
    im = Image.new("RGB", (S, S), J((225, 205, 165), r, 10)); d = ImageDraw.Draw(im)
    for k in range(3):
        y0 = 25 + k * 40
        d.line([(x, y0 + 12 * np.sin(x / 9 + k)) for x in range(0, S, 3)], fill=(70, 45, 30), width=2)
    pal = [(40, 60, 120), (160, 50, 40), (200, 150, 40), (70, 100, 60)]
    for _ in range(r.randint(8, 13)):
        x, y, s = r.randint(8, S - 22), r.randint(8, S - 22), r.randint(9, 16)
        d.ellipse([x, y, x + s, y + s], fill=J(r.choice(pal), r, 10), outline=(60, 40, 30), width=1)
        d.ellipse([x + s // 3, y + s // 3, x + 2 * s // 3, y + 2 * s // 3], fill=(235, 220, 185))
    return im

def banarasi(r):
    bg = r.choice([(110, 15, 40), (15, 70, 50), (20, 30, 80)]); im = Image.new("RGB", (S, S), J(bg, r, 10)); d = ImageDraw.Draw(im)
    g = J((215, 175, 60), r, 10); d.rectangle([0, 0, S, 9], fill=g); d.rectangle([0, S - 10, S, S], fill=g)
    for y in range(26, S - 14, 18):
        for x in range(10, S, 18):
            d.polygon([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)], outline=g, fill=g if (x + y) % 36 else None)
            d.ellipse([x + 7, y - 1, x + 10, y + 2], fill=g)
    return im

GEN = dict(zip(KEYS, [warli, madhubani, bandhani, ikat, kalamkari, banarasi]))

def sample_image(key, i):
    return GEN[key](random.Random(1000 + i)).resize((256, 256), Image.NEAREST)

def feats(im):
    im = im.convert("RGB").resize((96, 96)); a = np.asarray(im, float) / 255; h = np.asarray(im.convert("HSV"), float) / 255
    H, Sa, V = h[..., 0], h[..., 1], h[..., 2]; m = Sa > .2
    hist = np.histogram(H[m], bins=8, range=(0, 1))[0] / H.size if m.any() else np.zeros(8)
    g = a.mean(2); gx = np.abs(np.diff(g, axis=1)); gy = np.abs(np.diff(g, axis=0))
    F = np.abs(np.fft.fft2(g - g.mean())); F[0, 0] = 0
    gold = ((a[..., 0] > .65) & (a[..., 1] > .5) & (a[..., 2] < .45)).mean()
    white = ((V > .85) & (Sa < .2)).mean()
    return np.r_[hist, a.mean((0, 1)), Sa.mean(), Sa.std(), V.mean(), V.std(), white, (V < .2).mean(), gold,
                 gx.mean(), gy.mean(), (gx > .15).mean(), (gy > .15).mean(), gx.mean() / (gy.mean() + 1e-6), F.max() / (F.sum() + 1e-9)]

def dataset(n=90):
    X, y = [], []
    for i, k in enumerate(KEYS):
        r = random.Random(i)
        for _ in range(n): X.append(feats(GEN[k](r))); y.append(k)
        p = os.path.join(HERE, "dataset", k)  # optional: put real photos in dataset/<craft>/
        if os.path.isdir(p):
            for f in os.listdir(p):
                try:
                    im = Image.open(os.path.join(p, f)).convert("RGB")
                    for v in (im, im.transpose(Image.FLIP_LEFT_RIGHT)):
                        for _ in range(4): X.append(feats(v)); y.append(k)
                except Exception: pass
    return np.array(X), np.array(y)

def get_model():
    if os.path.exists(MODEL):
        try: return joblib.load(MODEL)
        except Exception: pass
    X, y = dataset()
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=.2, random_state=0, stratify=y)
    clf = RandomForestClassifier(200, random_state=0).fit(Xt, yt)
    print("holdout accuracy: %.2f" % clf.score(Xv, yv))
    clf = RandomForestClassifier(200, random_state=0).fit(X, y)
    joblib.dump(clf, MODEL); return clf

_clf = None
def analyze(im):
    global _clf
    _clf = _clf or get_model()
    f = feats(im); p = _clf.predict_proba([f])[0]; order = np.argsort(p)[::-1]
    px = np.asarray(im.convert("RGB").resize((48, 48)), float).reshape(-1, 3)
    km = KMeans(4, n_init=3, random_state=0).fit(px); cnt = np.bincount(km.labels_, minlength=4)
    pal = ["#%02x%02x%02x" % tuple(int(v) for v in km.cluster_centers_[i]) for i in np.argsort(cnt)[::-1]]
    sig = [("Zari / gold shine", min(1, f[19] * 8)), ("Pattern repetition", min(1, f[-1] * 60)),
           ("Line & edge detail", min(1, f[22] * 3)), ("Colour richness", min(1, f[11] * 1.6))]
    cls = list(_clf.classes_)
    return dict(key=str(cls[order[0]]), confidence=float(p[order[0]]), uncertain=bool(p[order[0]] < .5),
                top=[dict(key=str(cls[i]), p=float(p[i])) for i in order[:3]], palette=pal,
                signals=[dict(label=a, v=float(b)) for a, b in sig])

if __name__ == "__main__":
    get_model()
