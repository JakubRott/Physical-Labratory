#!/usr/bin/env python3
"""
track_dots.py – sledování dvou svítících teček ve videu.

Pro každé video vznikne složka <jméno_videa>/ a v ní:
  <jméno>_tracks.csv                    – polohy, natočení θ, rychlost, ω, α, moment
  <jméno>_01_draha_xy.png               – dráha středu + činky (barva = natočení θ)
  <jméno>_02_natoceni_cas.png           – natočení θ v čase (stupně + otáčky)
  <jméno>_03_rychlost_cas.png           – velikost rychlosti v čase
  <jméno>_04_rychlost_uhel.png          – velikost rychlosti v závislosti na θ
  <jméno>_05_moment_rychlost.png        – moment síly v závislosti na velikosti rychlosti,
                                          jen jedna perioda natočení θ = 0–360° (otáčky složeny)
  <jméno>_06_moment_uhel.png            – moment síly v závislosti na θ
  <jméno>_07_uhlova_rychlost_cas.png    – úhlová rychlost v čase
  <jméno>_08_uhlove_zrychleni_cas.png   – úhlové zrychlení v čase
  <jméno>_09_klic_natoceni.png          – KLÍČ: jak vypadá činka při θ = 0°, 30°, … 330°
  <jméno>_10_cinka_v_case.png           – činka v několika okamžicích + hodnoty veličin
  <jméno>_11_moment_na_ramenech.png     – činka v sérii okamžiků, šipky momentu na obou ramenech
  <jméno>_12_proc_se_toci.png           – kdy moment roztáčí / brzdí, ω, energetická bilance otáčky
  <jméno>_13_presnost_momentu.png       – jak moc se dá momentu věřit (šum, vyhlazení, opakovatelnost)
  volitelně <jméno>_debug.mp4           – zesvětlené video s trackem, šipkou rychlosti středu
                                          a panelem s číselnými hodnotami všech veličin
  volitelně <jméno>_moment_ramena.mp4   – video se šipkami momentu síly na obou ramenech
  (v obou videích je bílá šipka proudění vzduchu: vychází ze středu činky a míří PROTI její rychlosti)

Pokud složka už existuje a obsahuje soubory se stejným názvem, ke všem souborům
tohoto běhu se přidá společný index (_2, _3, …), nic se nepřepisuje.

Definice natočení θ
  * θ = úhel označeného konce činky měřený od VODOROVNÉ osy x obrazu:
    θ = 0° ⇒ označený konec míří doprava, 90° ⇒ nahoru, 180° ⇒ doleva, 270° ⇒ dolů.
  * Jeden konec činky je „označený“ (velký plný puntík). Tečky ve videu nelze rozlišit,
    proto je volba označeného konce konvence: v 1. platném snímku je to ten konec,
    který leží v horní polorovině (θ na startu je tedy 0–180°). Protože se činka otáčí
    plynule, lze označený konec dál sledovat a θ určit v plném rozsahu 360° (a dál
    720°, …), ne jen modulo 180°. Fyzicky je poloha θ a θ + 180° stejná.
  * Kladný směr θ: proti směru hodinových ručiček (--positive ccw, výchozí),
    nebo po směru (--positive cw).

Použití:
  python track_dots.py video.MOV
  python track_dots.py video.MOV --thr 15 --dot-distance-mm 20 --inertia 1.2e-6 --debug-video
  python track_dots.py video.MOV --debug-video --torque-video   # v obou videích šipka proudění vzduchu

Princip:
  1) pozadí = medián z každého N-tého snímku (odstraní statické světlé body)
  2) kandidáti = světlé skvrny v (snímek − pozadí), po lehkém rozostření
  3) dvojice se vybírá podle známé (téměř konstantní) vzdálenosti teček L
  4) střed = (A + B) / 2, natočení = úhel spojnice A–B, odvinuté po 180° (viz výše)
  5) rychlost, ω a α = derivace vyhlazené Savitzkého–Golayovým filtrem po souvislých úsecích
  6) moment síly M = I·α (I = moment setrvačnosti kolem středu; zadej --inertia).
     Bez --inertia se vykreslí M/I [s⁻²], tj. tvarově totéž co α.

Potřebné balíčky: opencv-python, numpy, matplotlib, scipy
"""
import argparse
import sys
import textwrap
import warnings
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.transforms import blended_transform_factory, ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import savgol_filter

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 100,
    "savefig.dpi": 200,
})

PHASE_CMAP = plt.get_cmap("twilight_shifted")      # cyklická mapa: 0° a 360° mají stejnou barvu
PHASE_NORM = Normalize(0, 360)
DEG_FMT = FuncFormatter(lambda v, _: f"{v:.0f}°")


# ----------------------------------------------------------------------------
# 0) Výstupní složka a názvy souborů
# ----------------------------------------------------------------------------
OUTPUTS = {
    "csv": "tracks.csv",
    "xy": "01_draha_xy.png",
    "phi_t": "02_natoceni_cas.png",
    "v_t": "03_rychlost_cas.png",
    "v_phi": "04_rychlost_uhel.png",
    "M_v": "05_moment_rychlost.png",
    "M_phi": "06_moment_uhel.png",
    "w_t": "07_uhlova_rychlost_cas.png",
    "a_t": "08_uhlove_zrychleni_cas.png",
    "key": "09_klic_natoceni.png",
    "snap": "10_cinka_v_case.png",
    "arms": "11_moment_na_ramenech.png",
    "mech": "12_proc_se_toci.png",
    "acc": "13_presnost_momentu.png",
    "debug": "debug.mp4",
    "arms_video": "moment_ramena.mp4",
}


def prepare_outputs(video, outdir=None):
    """Vytvoří složku <outdir>/<jméno videa> a vrátí cesty k výstupům.
    Když by se některý soubor přepsal, přidá společný index (_2, _3, …)."""
    stem = Path(video).stem
    folder = Path(outdir or ".") / stem
    folder.mkdir(parents=True, exist_ok=True)

    def build(sfx):
        out = {}
        for key, name in OUTPUTS.items():
            base, ext = name.rsplit(".", 1)
            out[key] = folder / f"{stem}_{base}{sfx}.{ext}"
        return out

    idx = 1
    while True:
        sfx = "" if idx == 1 else f"_{idx}"
        paths = build(sfx)
        if not any(p.exists() for p in paths.values()):
            return folder, paths
        idx += 1


# ----------------------------------------------------------------------------
# 1) Pozadí
# ----------------------------------------------------------------------------
def estimate_background(path, step=12):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Nelze otevřít video: {path}")
    samples, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            samples.append(f[::2, ::2])  # zmenšeno jen kvůli paměti
        i += 1
    cap.release()
    h, w = samples[0].shape[:2]
    bg = np.median(np.array(samples), axis=0).astype(np.float32)
    bg = cv2.resize(bg, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    return bg, i


# ----------------------------------------------------------------------------
# 2) Detekce skvrn v každém snímku
# ----------------------------------------------------------------------------
def detect_blobs(frame, bg, thr, min_area):
    diff = (frame.astype(np.float32) - bg).max(axis=2)
    diff = cv2.GaussianBlur(diff, (0, 0), 1.0)
    mask = (diff > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x0, y0, w, h = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                        stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
        sub = diff[y0:y0 + h, x0:x0 + w] * (lab[y0:y0 + h, x0:x0 + w] == k)
        wsum = sub.sum()
        if wsum <= 0:
            continue
        ys, xs = np.mgrid[y0:y0 + h, x0:x0 + w]
        cx = (sub * xs).sum() / wsum          # centroid vážený jasem (sub-pixel)
        cy = (sub * ys).sum() / wsum
        blobs.append((cx, cy, float(wsum)))
    return blobs


def collect_blobs(path, bg, thr, min_area):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    all_blobs = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        all_blobs.append(detect_blobs(f, bg, thr, min_area))
    cap.release()
    return all_blobs, fps


# ----------------------------------------------------------------------------
# 3) Výběr dvojice teček
# ----------------------------------------------------------------------------
def estimate_distance(all_blobs):
    d = []
    for b in all_blobs:
        if len(b) == 2:
            d.append(np.hypot(b[0][0] - b[1][0], b[0][1] - b[1][1]))
    if len(d) < 5:
        sys.exit("Našlo se příliš málo snímků se dvěma tečkami – zkus snížit --thr.")
    d = np.array(d)
    med = np.median(d)
    keep = d[np.abs(d - med) < 0.3 * med]
    return float(np.median(keep))


def pick_pairs(all_blobs, L, tol):
    n = len(all_blobs)
    A = np.full((n, 2), np.nan)
    B = np.full((n, 2), np.nan)
    for i, b in enumerate(all_blobs):
        best, best_score = None, -1.0
        for p in range(len(b)):
            for q in range(p + 1, len(b)):
                dist = np.hypot(b[p][0] - b[q][0], b[p][1] - b[q][1])
                if abs(dist - L) > tol * L:
                    continue
                score = b[p][2] + b[q][2]
                if score > best_score:
                    best, best_score = (p, q), score
        if best is not None:
            p, q = best
            A[i] = b[p][:2]
            B[i] = b[q][:2]
    return A, B


# ----------------------------------------------------------------------------
# 4) Střed a natočení
# ----------------------------------------------------------------------------
def compute_pose(A, B, t, gap_max=3, fit_n=20):
    """Vrací střed (n,2), φ [rad] (odvinuté, počátek = první platný snímek = 0),
    platné snímky a raw0 = úhel spojnice teček v prvním platném snímku
    (v souřadnicích grafu, osa y nahoru; rozsah 0–180°).

    Spojnice teček je neorientovaná (tečky nerozlišíme), takže v jednom snímku
    známe úhel jen modulo 180°. Protože se ale činka otáčí plynule, lze úhel
    odvinout a sledovat označený konec činky v plném rozsahu 360° a dál:
      * uvnitř souvislého úseku (mezera <= gap_max snímků): nejbližší větev
      * přes delší mezeru: větev nejbližší extrapolaci z lineárního fitu
        posledních `fit_n` bodů předchozího úseku (tj. předpoklad, že se
        úhlová rychlost přes mezeru výrazně nezmění).
    """
    valid = ~np.isnan(A[:, 0])
    com = (A + B) / 2.0
    com[:, 1] *= -1
    dx = B[:, 0] - A[:, 0]
    dy = -(B[:, 1] - A[:, 1])
    raw = np.arctan2(dy, dx)                       # (-π, π]
    raw = np.mod(raw, np.pi)                       # modulo 180°

    idx = np.where(valid)[0]
    phi = np.full(len(A), np.nan)
    if len(idx) == 0:
        return com, phi, valid, 0.0
    raw0 = float(raw[idx[0]])
    phi[idx[0]] = raw[idx[0]]
    hist_i = [idx[0]]
    for prev, cur in zip(idx[:-1], idx[1:]):
        if cur - prev <= gap_max + 1 or len(hist_i) < 5:
            pred = phi[prev]
        else:
            h = np.array(hist_i[-fit_n:])
            slope, icpt = np.polyfit(t[h], phi[h], 1)
            pred = slope * t[cur] + icpt
        k = np.round((pred - raw[cur]) / np.pi)    # nejbližší větev k predikci
        phi[cur] = raw[cur] + k * np.pi
        hist_i.append(cur)
    phi -= phi[idx[0]]
    return com, phi, valid, raw0


def assign_tips(A, B, psi):
    """Určí, který z dvojice detekovaných bodů je označený konec T1 a který druhý konec T2.
    psi = úhel označeného konce vůči středu (rad, proti směru hod. ručiček, y nahoru).
    Vrací T1, T2 v souřadnicích obrazu (y dolů); mimo detekci NaN."""
    T1 = np.full(A.shape, np.nan)
    T2 = np.full(A.shape, np.nan)
    for i in np.where(~np.isnan(A[:, 0]) & np.isfinite(psi))[0]:
        c = (A[i] + B[i]) / 2.0
        d = np.array([np.cos(psi[i]), -np.sin(psi[i])])      # směr v obraze (y dolů)
        if np.dot(A[i] - c, d) >= 0:
            T1[i], T2[i] = A[i], B[i]
        else:
            T1[i], T2[i] = B[i], A[i]
    return T1, T2


def segment_flags(n, segs, half):
    """Číslo souvislého úseku pro každý snímek a příznak 'blízko kraje úseku' (EDGE)."""
    seg_of = np.zeros(n, int)
    edge = np.zeros(n, bool)
    for k, (s0, e0) in enumerate(segs, 1):
        for j in range(s0, e0 + 1):
            seg_of[j] = k
            edge[j] = min(j - s0, e0 - j) < half
    return seg_of, edge


def segments_from_valid(valid, gap_max):
    """Indexy souvislých úseků (mezery ≤ gap_max snímků se přeskočí)."""
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return []
    segs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > gap_max + 1:
            segs.append((start, a))
            start = b
    segs.append((start, idx[-1]))
    return segs


# ----------------------------------------------------------------------------
# 5) Kinematika: rychlost, úhlová rychlost, úhlové zrychlení, moment
# ----------------------------------------------------------------------------
def sg_derivative(y, valid, segs, dt, win, poly, deriv):
    """Derivace signálu y podle času (Savitzky–Golay) po souvislých úsecích.
    Malé mezery uvnitř úseku se pro filtr lineárně dopočítají, ale ve výsledku
    zůstanou prázdné (NaN)."""
    out = np.full(len(y), np.nan)
    for s, e in segs:
        ii = np.arange(s, e + 1)
        vi = ii[valid[ii]]
        yy = np.interp(ii, vi, y[vi])
        w = min(win, len(ii))
        if w % 2 == 0:
            w -= 1
        if w < poly + 2:
            continue                                # úsek je příliš krátký
        out[ii] = savgol_filter(yy, w, poly, deriv=deriv, delta=dt)
    out[~valid] = np.nan
    return out


def compute_kinematics(com_scaled, theta, valid, segs, dt, win, win_alpha, poly, inertia):
    vx = sg_derivative(com_scaled[:, 0], valid, segs, dt, win, poly, 1)
    vy = sg_derivative(com_scaled[:, 1], valid, segs, dt, win, poly, 1)
    speed = np.hypot(vx, vy)
    ax_ = sg_derivative(com_scaled[:, 0], valid, segs, dt, win_alpha, max(poly, 3), 2)
    ay_ = sg_derivative(com_scaled[:, 1], valid, segs, dt, win_alpha, max(poly, 3), 2)
    omega = sg_derivative(theta, valid, segs, dt, win, poly, 1)
    alpha = sg_derivative(theta, valid, segs, dt, win_alpha, max(poly, 3), 2)
    if inertia:
        torque = inertia * alpha
        t_label, t_unit = "M", "N·m"
    else:
        torque = alpha
        t_label, t_unit = "M/I", "s⁻²"
    # vektory rychlosti a zrychlení středu (osa x doprava, y nahoru) vracíme zvlášť
    return speed, omega, alpha, torque, t_label, t_unit, (vx, vy, ax_, ay_)


# ----------------------------------------------------------------------------
# 6) Kreslení činky, klíč natočení, značky úhlu
# ----------------------------------------------------------------------------
def draw_dumbbell(ax, c, ang, half, color="0.15", lw=2.0, ms_big=7.0, ms_small=4.5,
                  zorder=4, ghost=False):
    """Činka se středem c a natočením ang [rad]. Označený konec (ve směru ang) je
    velký plný puntík, druhý konec je malý bílý puntík."""
    d = np.array([np.cos(ang), np.sin(ang)])
    c = np.asarray(c, float)
    pm, po = c + half * d, c - half * d
    if ghost:
        ax.plot([po[0], pm[0]], [po[1], pm[1]], "--", color="0.55", lw=lw * 0.8, zorder=zorder)
        ax.plot(pm[0], pm[1], "o", ms=ms_big, mfc="white", mec="0.45", mew=1.1, zorder=zorder)
        ax.plot(po[0], po[1], "o", ms=ms_small, mfc="white", mec="0.45", mew=1.0, zorder=zorder)
    else:
        ax.plot([po[0], pm[0]], [po[1], pm[1]], "-", color=color, lw=lw,
                solid_capstyle="round", zorder=zorder)
        ax.plot(pm[0], pm[1], "o", ms=ms_big, mfc=color, mec="k", mew=0.9, zorder=zorder + 0.1)
        ax.plot(po[0], po[1], "o", ms=ms_small, mfc="white", mec="k", mew=0.9, zorder=zorder + 0.1)


def pos_text(conv):
    return "proti směru hodinových ručiček (CCW)" if conv.sgn > 0 else "po směru hodinových ručiček (CW)"


def footnote(fig, conv, extra=""):
    txt = ("θ = úhel označeného konce činky od vodorovné osy (0° = označený konec míří doprava), "
           f"kladný směr = {pos_text(conv)}.  Tečky nelze rozlišit (θ a θ + 180° je stejná poloha); "
           "θ se sleduje plynule, přes mezery bez detekce je odvinutí odhad.")
    if extra:
        txt += "\n" + extra
    wrapped = "\n".join(textwrap.fill(p, 175) for p in txt.split("\n"))
    fig.text(0.012, 0.008, wrapped, fontsize=8, color="0.3", ha="left", va="bottom",
             linespacing=1.35)


def draw_key(kax, conv, colored=True):
    """Kruhový klíč: barva = θ mod 360°, uprostřed silná vodorovná činka = poloha 0°.
    Poloha popisků odpovídá skutečnému natočení činky v obraze."""
    kax.set_aspect("equal")
    kax.axis("off")
    kax.set_anchor("N")
    kax.set_xlim(-1.85, 1.85)
    kax.set_ylim(-2.75, 1.85)
    kax.grid(False)
    base = np.degrees(conv.raw0)
    for a in range(0, 360, 4):
        col = PHASE_CMAP(PHASE_NORM(a + 2)) if colored else "0.88"
        c_ang = base + conv.sgn * (a + 2)
        kax.add_patch(Wedge((0, 0), 1.0, c_ang - 2.2, c_ang + 2.2, width=0.27,
                            fc=col, ec=col, lw=0.3))
    for a in (0, 90, 180, 270):
        ang = conv.raw0 + conv.sgn * np.radians(a)
        cs, sn = np.cos(ang), np.sin(ang)
        kax.plot([0.70 * cs, 1.05 * cs], [0.70 * sn, 1.05 * sn], color="k", lw=1.3)
        kax.text(1.30 * cs, 1.30 * sn, f"{a}°", ha="center", va="center", fontsize=10,
                 fontweight="bold" if a == 0 else "normal")
    draw_dumbbell(kax, (0, 0), conv.raw0, 0.55, color="k", lw=3.2, ms_big=9, ms_small=5.5)
    # šipka kladného směru
    th = np.radians(np.linspace(28, 70, 20))
    ang = conv.raw0 + conv.sgn * th
    xs, ys = 1.62 * np.cos(ang), 1.62 * np.sin(ang)
    kax.plot(xs[:-1], ys[:-1], color="0.25", lw=1.6)
    kax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-3], ys[-3]),
                 arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.6, shrinkA=0, shrinkB=0))
    mid = conv.raw0 + conv.sgn * np.radians(49)
    kax.text(1.80 * np.cos(mid), 1.80 * np.sin(mid), "+", ha="center", va="center",
             fontsize=11, fontweight="bold", color="0.25")
    kax.text(0, -1.62, "Klíč natočení θ", ha="center", va="top", fontsize=10.5, fontweight="bold")
    lines = ["silná činka = 0° (vodorovně, vpravo)",
             "velký puntík = označený konec",
             "+ = " + ("proti směru hod. ruček" if conv.sgn > 0 else "po směru hod. ruček")]
    if colored:
        lines.append("barva = θ mod 360°")
    kax.text(0, -1.95, "\n".join(lines), ha="center", va="top", fontsize=8.5, linespacing=1.45)


def pick_angle_step(lo, hi, max_n=12):
    """Nejmenší 'hezký' krok úhlu [°], pro nějž je v <lo,hi> nejvýš max_n násobků."""
    for step in (45, 90, 180, 360, 720, 1440, 2880, 5760):
        n = np.floor(hi / step) - np.ceil(lo / step) + 1
        if n <= max_n:
            return step
    return 11520


def angle_crossings(t, theta_deg, valid, step):
    """Časy, kdy θ poprvé dosáhne násobků `step` (kromě 0°)."""
    idx = np.where(valid)[0]
    res = []
    if len(idx) < 2:
        return res
    tt, aa = t[idx], theta_deg[idx]
    for k in range(int(np.ceil(aa.min() / step)), int(np.floor(aa.max() / step)) + 1):
        a = k * step
        hit = np.where((aa[:-1] - a) * (aa[1:] - a) <= 0)[0]
        if len(hit) == 0:
            continue
        j = hit[0]
        da = aa[j + 1] - aa[j]
        frac = 0.0 if da == 0 else (a - aa[j]) / da
        res.append((a, tt[j] + frac * (tt[j + 1] - tt[j])))
    return res


def add_angle_markers(ax, t, theta_deg, valid):
    """Svislé čáry v okamžicích, kdy θ dosáhne násobku 90° (plná čára = celá otáčka)."""
    idx = np.where(valid)[0]
    step = pick_angle_step(theta_deg[idx].min(), theta_deg[idx].max(), 12)
    tr = ax.get_xaxis_transform()
    for a, tc in angle_crossings(t, theta_deg, valid, step):
        full = abs(a) % 360 < 1e-6
        ax.axvline(tc, color="0.2" if full else "0.45", ls="-" if full else ":",
                   lw=1.2 if full else 0.9, alpha=0.75, zorder=1)
        ax.text(tc, 1.01, "0°" if a == 0 else f"{a:+.0f}°", transform=tr, ha="center",
                va="bottom", fontsize=8.5, fontweight="bold" if full else "normal", color="0.15")
    t0 = t[idx[0]]
    ax.axvline(t0, color="C3", lw=1.2, alpha=0.8, zorder=1)
    ax.text(t0, 1.075, f"start (θ = {theta_deg[idx[0]]:.0f}°)", transform=tr, ha="left",
            va="bottom", fontsize=8.5, fontweight="bold", color="C3")
    return step


def add_pose_icons_top(ax, angles, conv, size=0.44, lift=0.40):
    """Nad horní okraj grafu (v poloze odpovídající ose x = θ) nakreslí malé činky."""
    fig = ax.figure
    tr = (blended_transform_factory(ax.transData, ax.transAxes)
          + ScaledTranslation(0, lift, fig.dpi_scale_trans))
    tr_txt = (blended_transform_factory(ax.transData, ax.transAxes)
              + ScaledTranslation(0, lift + size / 2 + 0.05, fig.dpi_scale_trans))
    for a in angles:
        ia = inset_axes(ax, width=size, height=size, loc="center",
                        bbox_to_anchor=(a, 1.0), bbox_transform=tr, borderpad=0)
        ia.set_xlim(-1.05, 1.05)
        ia.set_ylim(-1.05, 1.05)
        ia.set_aspect("equal")
        ia.axis("off")
        ia.grid(False)
        ang = conv.raw0 + conv.sgn * np.radians(a)
        zero = abs(a) < 1e-9
        draw_dumbbell(ia, (0, 0), ang, 0.85, color="C3" if zero else "0.15", lw=1.8,
                      ms_big=5.5, ms_small=3.5, zorder=3)
        ax.text(a, 1.0, f"{a:.0f}°", transform=tr_txt, ha="center",
                va="bottom", fontsize=8.5, fontweight="bold" if zero else "normal",
                color="C3" if zero else "0.15")


def band_shading(ax, lo, hi, axis="x", label=True):
    """Střídavé podbarvení po úhlových pásech 0–360°, 360–720°, … (volitelně s popisem)."""
    j0, j1 = int(np.floor(lo / 360)), int(np.ceil(hi / 360))
    for j in range(j0, j1):
        a0, a1 = 360 * j, 360 * j + 360
        if j % 2 == 0:
            (ax.axvspan if axis == "x" else ax.axhspan)(a0, a1, color="0.5", alpha=0.08,
                                                       lw=0, zorder=0)
        c0, c1 = max(a0, lo), min(a1, hi)
        if label and c1 - c0 > 60:
            txt = f"{a0:.0f}° … {a1:.0f}°"
            if axis == "x":
                ax.text((c0 + c1) / 2, 0.02, txt, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=9, color="0.35", style="italic")
            else:
                ax.text(0.012, (c0 + c1) / 2, txt, transform=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=9, color="0.35", style="italic")
    for k in range(int(np.ceil(lo / 360)), int(np.floor(hi / 360)) + 1):
        (ax.axvline if axis == "x" else ax.axhline)(360 * k, color="0.25", lw=1.0,
                                                    alpha=0.7, zorder=1)


# ----------------------------------------------------------------------------
# 7) Jednotlivé grafy
# ----------------------------------------------------------------------------
def new_fig_with_key(figsize=(13.2, 6.6), ratios=(4.2, 1.25)):
    return plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": list(ratios)})


def nice_number(x):
    """Největší 'hezké' číslo 1/2/5 · 10^n, které není větší než x."""
    e = 10.0 ** np.floor(np.log10(x))
    m = x / e
    return e * max(c for c in (1, 2, 5) if c <= m * 1.0001)


def plot_xy(com, theta_deg, conv, valid, A, B, L, scale, unit, out_png, gap_max, ticks_every):
    th_mod = np.mod(theta_deg, 360.0)
    fig, (ax, kax) = new_fig_with_key(figsize=(13.4, 7.6), ratios=(4.0, 1.25))
    idx = np.where(valid)[0]
    for s, e in segments_from_valid(valid, gap_max):
        ii = np.arange(s, e + 1)
        ii = ii[valid[ii]]
        if len(ii) < 2:
            continue
        pts = com[ii] * scale
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap=PHASE_CMAP, norm=PHASE_NORM, linewidth=2.5)
        lc.set_array(th_mod[ii][:-1])
        ax.add_collection(lc)
    ax.scatter(com[idx, 0] * scale, com[idx, 1] * scale, c=th_mod[idx],
               cmap=PHASE_CMAP, norm=PHASE_NORM, s=12, zorder=3)

    # „činky“ – skutečná poloha teček; velký puntík = označený konec (sledovaný plynule)
    for k in idx[::max(1, ticks_every)]:
        ang = conv.raw0 + conv.sgn * np.radians(theta_deg[k])
        d = np.array([np.cos(ang), np.sin(ang)])
        a = np.array([A[k, 0], -A[k, 1]])
        b = np.array([B[k, 0], -B[k, 1]])
        c = (a + b) / 2
        pm, po = (a, b) if np.dot(a - c, d) >= 0 else (b, a)
        pm, po = pm * scale, po * scale
        col = PHASE_CMAP(PHASE_NORM(th_mod[k]))
        ax.plot([po[0], pm[0]], [po[1], pm[1]], "-", color=col, lw=1.5, zorder=2)
        ax.plot(pm[0], pm[1], "o", ms=5.5, mfc=col, mec="k", mew=0.7, zorder=4)
        ax.plot(po[0], po[1], "o", ms=3.2, mfc="white", mec="k", mew=0.7, zorder=4)

    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.set_title(f"Dráha středu mezi tečkami (vzdálenost teček ≈ {L * scale:.1f} {unit})")
    ax.annotate("start", com[idx[0]] * scale, xytext=(8, 8), textcoords="offset points",
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax.annotate("konec", com[idx[-1]] * scale, xytext=(8, -14), textcoords="offset points",
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    draw_key(kax, conv)
    footnote(fig, conv, "Činky v grafu mají stejné natočení i barvu jako odpovídající činka v klíči vpravo.")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_png)
    plt.close(fig)


def plot_torque_speed_period(theta_deg, speed, torque, valid, segs, conv, t_label, t_unit,
                             v_unit, out_png, bin_deg=10):
    """M(|v|) pouze v jedné periodě natočení 0–360°.
    Obě veličiny jsou periodické s periodou 360° natočení, proto se všechny otáčky
    složí přes θ mod 360° do jedné periody (tenké čáry = jednotlivé otáčky, body barevně
    podle θ, tlustá černá křivka = průměr po `bin_deg` stupních)."""
    ok = valid & np.isfinite(theta_deg) & np.isfinite(speed) & np.isfinite(torque)
    th = np.mod(theta_deg, 360.0)
    fig, (ax, kax) = new_fig_with_key()

    # tenké čáry po jednotlivých otáčkách (v místě přechodu 360° → 0° se čára přeruší)
    lines = []
    for s, e in segs:
        ii = np.arange(s, e + 1)
        ii = ii[ok[ii]]
        for p, q in zip(ii[:-1], ii[1:]):
            if abs(th[q] - th[p]) < 180.0 and q - p <= 4:
                lines.append([(speed[p], torque[p]), (speed[q], torque[q])])
    if lines:
        ax.add_collection(LineCollection(lines, colors="0.78", linewidths=0.7, zorder=1))
    ax.scatter(speed[ok], torque[ok], c=th[ok], cmap=PHASE_CMAP, norm=PHASE_NORM, s=16,
               alpha=0.85, zorder=3)

    # průměr po úhlových intervalech
    edges = np.arange(0.0, 360.0 + bin_deg, bin_deg)
    mv, mt, mc = [], [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        sel = ok & (th >= lo_) & (th < hi_)
        if sel.sum() >= 2:
            mv.append(np.mean(speed[sel]))
            mt.append(np.mean(torque[sel]))
            mc.append((lo_ + hi_) / 2)
    if len(mv) >= 3:
        mv, mt, mc = np.array(mv), np.array(mt), np.array(mc)
        ax.plot(np.r_[mv, mv[0]], np.r_[mt, mt[0]], "-", color="k", lw=2.4, zorder=5,
                solid_joinstyle="round")
        for j in range(len(mv)):                       # šipky = směr rostoucího θ
            j2 = (j + 1) % len(mv)
            if j % 3 == 0:
                ax.annotate("", xy=(mv[j2], mt[j2]), xytext=(mv[j], mt[j]), zorder=6,
                            arrowprops=dict(arrowstyle="-|>", color="k", lw=0, mutation_scale=16))
        for a in (0, 90, 180, 270):                    # popisky úhlů na průměrné křivce
            j = int(np.argmin(np.abs(mc - (a + bin_deg / 2))))
            ax.plot(mv[j], mt[j], "o", ms=9, mfc="white", mec="k", mew=1.6, zorder=7)
            ax.annotate(f"{a}°", (mv[j], mt[j]), xytext=(9, 9), textcoords="offset points",
                        fontsize=11, fontweight="bold", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", alpha=0.9))
    ax.autoscale_view()
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(f"velikost rychlosti |v| [{v_unit}]")
    ax.set_ylabel(f"moment síly {t_label} [{t_unit}]")
    ax.set_title("Moment síly v závislosti na velikosti rychlosti (jedna perioda, θ = 0–360°)")
    draw_key(kax, conv)
    n_rev = (np.nanmax(theta_deg[ok]) - np.nanmin(theta_deg[ok])) / 360.0 if ok.any() else 0
    footnote(fig, conv,
             f"Všechny otáčky ({n_rev:.1f} ot.) složeny do jedné periody (θ mod 360°). Barva bodu = θ "
             f"(viz klíč), tenké šedé čáry = jednotlivé otáčky, tlustá černá křivka = průměr po "
             f"{bin_deg}° ve směru rostoucího θ, popisky 0°/90°/180°/270° = úhly na průměrné křivce.")
    fig.tight_layout(rect=(0, 0.095, 1, 1))
    fig.savefig(out_png)
    plt.close(fig)


def plot_rotation_time(t, theta_deg, valid, segs, conv, out_png):
    """Natočení θ v čase: hlavní osa ve stupních, druhá osa v otáčkách."""
    th_mod = np.mod(theta_deg, 360.0)
    fig, (ax, kax) = new_fig_with_key()
    idx = np.where(valid)[0]
    lo, hi = np.nanmin(theta_deg[idx]), np.nanmax(theta_deg[idx])
    band_shading(ax, lo, hi, axis="y", label=False)
    for s, e in segs:
        ii = np.arange(s, e + 1)
        ii = ii[valid[ii]]
        ax.plot(t[ii], theta_deg[ii], "-", color="0.6", lw=1.2, zorder=2)
    ax.scatter(t[idx], theta_deg[idx], c=th_mod[idx], cmap=PHASE_CMAP, norm=PHASE_NORM,
               s=16, zorder=3)
    ax.axhline(0, color="k", lw=1.0, alpha=0.6, zorder=1)
    ax.text(0.99, 0, "0° = vodorovná osa (vpravo)", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9, fontweight="bold")
    ax.plot(t[idx[0]], theta_deg[idx[0]], "o", ms=9, mfc="none", mec="C3", mew=1.8, zorder=5)
    ax.annotate(f"start (θ = {theta_deg[idx[0]]:.0f}°)", (t[idx[0]], theta_deg[idx[0]]),
                xytext=(10, 14), textcoords="offset points", color="C3", fontsize=9,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color="C3"))
    ax.yaxis.set_major_locator(MultipleLocator(pick_angle_step(lo, hi, 14)))
    ax.yaxis.set_major_formatter(DEG_FMT)
    sec = ax.secondary_yaxis("right", functions=(lambda x: x / 360.0, lambda x: x * 360.0))
    sec.set_ylabel("natočení [otáčky]")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("natočení θ [°]")
    ax.set_title("Natočení činky v čase (0° = vodorovná osa)")
    draw_key(kax, conv)
    footnote(fig, conv, "Střídavé šedé pruhy = úhlové pásy po 360°; barva bodu = θ mod 360° (viz klíč).")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_png)
    plt.close(fig)


def plot_vs_time_with_rotation(t, y, valid, segs, theta_deg, conv, ylabel, title, out_png,
                               zero_line=False):
    """Časový graf; barva bodů = aktuální θ mod 360° (klíč vpravo), svislé čáry = násobky 90°."""
    th_mod = np.mod(theta_deg, 360.0)
    fig, (ax, kax) = new_fig_with_key()
    for s, e in segs:
        ii = np.arange(s, e + 1)
        ii = ii[valid[ii]]
        ax.plot(t[ii], y[ii], "-", color="0.65", lw=1.0, zorder=2)
    idx = np.where(valid)[0]
    ax.scatter(t[idx], y[idx], c=th_mod[idx], cmap=PHASE_CMAP, norm=PHASE_NORM, s=16, zorder=3)
    if zero_line:
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    add_angle_markers(ax, t, theta_deg, valid)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=26)
    draw_key(kax, conv)
    footnote(fig, conv, "Svislé čáry s popisky = okamžiky, kdy θ dosáhne násobku 90° "
                        "(plná čára = celá otáčka 360°); barva bodu = aktuální θ mod 360° (viz klíč).")
    fig.tight_layout(rect=(0, 0.095, 1, 1))
    fig.savefig(out_png)
    plt.close(fig)


def plot_vs_angle(theta_deg, y, t, valid, segs, conv, ylabel, title, out_png, zero_line=False):
    """y(θ): osa x = θ ve stupních (360° za sebou), nad grafem malé činky v daných úhlech."""
    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ok = valid & np.isfinite(theta_deg) & np.isfinite(y)
    lo, hi = np.nanmin(theta_deg[ok]), np.nanmax(theta_deg[ok])
    pad = 0.04 * max(hi - lo, 90)
    ax.set_xlim(lo - pad, hi + pad)
    band_shading(ax, lo - pad, hi + pad, axis="x", label=False)
    for s, e in segs:
        ii = np.arange(s, e + 1)
        ii = ii[ok[ii]]
        ax.plot(theta_deg[ii], y[ii], "-", color="0.75", lw=0.8, zorder=2)
    sc = ax.scatter(theta_deg[ok], y[ok], c=t[ok], cmap="viridis", s=16, zorder=3)
    if zero_line:
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    step = pick_angle_step(lo - pad, hi + pad, 14)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(DEG_FMT)
    angles = [k * step for k in range(int(np.ceil((lo - pad) / step)),
                                       int(np.floor((hi + pad) / step)) + 1)]
    add_pose_icons_top(ax, angles, conv)
    ax.set_xlabel("natočení θ [°]   (0° = vodorovná osa, označený konec míří doprava; "
                  "svislé čáry = násobky 360°)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=62)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("t [s]")
    footnote(fig, conv, "Malé činky nad grafem ukazují, jak činka vypadá při daném θ "
                        "(velký puntík = označený konec, červená = 0°).")
    with warnings.catch_warnings():          # vložené osy s činkami tight_layout nezná
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=(0, 0.095, 1, 1))
    fig.savefig(out_png)
    plt.close(fig)


def plot_xy_colored_by_time(x, y, t, valid, segs, xlabel, ylabel, title, out_png, zero_lines=False):
    """Závislost y(x) bez osy úhlu – barva = čas."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for s, e in segs:
        ii = np.arange(s, e + 1)
        ii = ii[valid[ii] & np.isfinite(x[ii]) & np.isfinite(y[ii])]
        ax.plot(x[ii], y[ii], "-", color="0.75", lw=0.8, zorder=1)
    ok = valid & np.isfinite(x) & np.isfinite(y)
    sc = ax.scatter(x[ok], y[ok], c=t[ok], cmap="viridis", s=16, zorder=3)
    if zero_lines:
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("t [s]")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_key_figure(conv, out_png, step=30, ncols=4):
    """Klíč: jak vypadá činka při θ = 0°, 30°, … ; 0° (vodorovně) je vždy vyznačeno čárkovaně."""
    angles = np.arange(0, 360, step)
    nrows = int(np.ceil(len(angles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.25 * nrows + 1.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, a in zip(axes.flat, angles):
        col = PHASE_CMAP(PHASE_NORM(a))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.add_patch(Circle((0, 0), 1.15, fill=False, ec="0.85", lw=1))
        # osy obrazu
        ax.annotate("", xy=(-0.95, -1.35), xytext=(-1.42, -1.35),
                    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1))
        ax.annotate("", xy=(-1.42, -0.88), xytext=(-1.42, -1.35),
                    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1))
        ax.text(-0.9, -1.45, "x", fontsize=8, color="0.45", va="center")
        ax.text(-1.5, -0.85, "y", fontsize=8, color="0.45", ha="center")
        ang = conv.raw0 + conv.sgn * np.radians(a)
        if a > 0:
            draw_dumbbell(ax, (0, 0), conv.raw0, 1.0, ghost=True, zorder=2)
            th = np.radians(np.linspace(0, a, 60))
            aa = conv.raw0 + conv.sgn * th
            xs, ys = 0.55 * np.cos(aa), 0.55 * np.sin(aa)
            ax.plot(xs[:-1], ys[:-1], color="0.35", lw=1.2, zorder=3)
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-3], ys[-3]),
                        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.2,
                                        shrinkA=0, shrinkB=0))
        draw_dumbbell(ax, (0, 0), ang, 1.0, color=col, lw=3.5, ms_big=11, ms_small=7, zorder=4)
        if a == 0:
            ax.set_title("θ = 0°   (vodorovně, vpravo)", fontsize=12, fontweight="bold")
        else:
            eq = f"   (= −{360 - a}°)" if a > 180 else ""
            ax.set_title(f"θ = {a}°{eq}", fontsize=12, fontweight="bold")
    fig.suptitle("Klíč natočení: jaká poloha činky odpovídá jakému úhlu θ", fontsize=15,
                 fontweight="bold", y=0.995)
    fig.text(0.5, 0.012,
             "Barevná plná činka = poloha při daném θ; šedá čárkovaná = referenční poloha 0° "
             "(vodorovně, označený konec vpravo); šedý oblouk = od 0° k θ.\n"
             f"Velký puntík = označený konec činky.  Kladný směr θ: {pos_text(conv)}.  "
             "Činky jsou kresleny ve stejné orientaci jako v obraze videa (x doprava, y nahoru).\n"
             "Barva odpovídá barvě v grafech.  Úhel θ a θ + 360° je stejná poloha (např. 330° = −30°).",
             ha="center", va="bottom", fontsize=9, color="0.25", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97), h_pad=3.5)
    fig.savefig(out_png)
    plt.close(fig)


def plot_snapshots(t, theta_deg, speed, omega, alpha, torque, valid, conv, v_unit,
                   t_label, t_unit, n, out_png):
    """Činka v n okamžicích rovnoměrně rozložených v čase + hodnoty veličin v těch okamžicích."""
    good = np.where(valid & np.isfinite(speed) & np.isfinite(omega) &
                    np.isfinite(alpha) & np.isfinite(torque))[0]
    if len(good) == 0:
        return False
    n = min(n, len(good))
    sel = good[np.unique(np.round(np.linspace(0, len(good) - 1, n)).astype(int))]
    ncols = 4
    nrows = int(np.ceil(len(sel) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 4.0 * nrows + 1.0), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, k in zip(axes.flat, sel):
        th = theta_deg[k]
        thm = th % 360.0
        col = PHASE_CMAP(PHASE_NORM(thm))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-2.85, 1.5)
        ax.set_aspect("equal")
        ax.add_patch(Circle((0, 0), 1.15, fill=False, ec="0.85", lw=1))
        draw_dumbbell(ax, (0, 0), conv.raw0, 1.0, ghost=True, zorder=2)
        ang = conv.raw0 + conv.sgn * np.radians(th)
        draw_dumbbell(ax, (0, 0), ang, 1.0, color=col, lw=3.5, ms_big=11, ms_small=7, zorder=4)
        ax.set_title(f"t = {t[k]:.2f} s", fontsize=12, fontweight="bold")
        txt = (f"θ = {th:+.0f}°   (≡ {thm:.0f}°,  {th / 360:+.2f} ot.)\n"
               f"ω = {omega[k]:+.2f} rad/s\n"
               f"α = {alpha[k]:+.2f} rad/s²\n"
               f"|v| = {speed[k]:.1f} {v_unit}\n"
               f"{t_label} = {torque[k]:+.3g} {t_unit}")
        ax.text(0, -1.4, txt, ha="center", va="top", fontsize=9.5, linespacing=1.55)
    fig.suptitle("Činka v jednotlivých okamžicích a odpovídající hodnoty", fontsize=15,
                 fontweight="bold", y=0.995)
    fig.text(0.5, 0.012,
             "Barevná činka = skutečné natočení v daném čase; šedá čárkovaná = poloha 0° (vodorovně, vpravo); "
             f"velký puntík = označený konec.  Kladný směr θ: {pos_text(conv)}.\n"
             "θ = úhel od vodorovné osy (plynule sledovaný), v závorce stejná poloha v rozsahu 0–360° "
             "a θ v otáčkách.",
             ha="center", va="bottom", fontsize=9, color="0.25", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_torque_arms_series(t, theta_deg, omega, alpha, torque, valid, conv, t_label, t_unit,
                            n, out_png, force=None):
    """Činka v sérii okamžiků (v čase zleva doprava, shora dolů). Na obou ramenech je šipka
    momentu: tečná (kolmo na rameno), na protilehlých koncích opačná (dvojice sil), délka
    úměrná |M| se stejným měřítkem ve všech panelech. Označený konec 1 = velký plný puntík."""
    good = np.where(valid & np.isfinite(theta_deg) & np.isfinite(torque) &
                    np.isfinite(omega) & np.isfinite(alpha))[0]
    if len(good) == 0:
        return False
    p95 = float(np.percentile(np.abs(torque[good]), 95))
    if p95 <= 0:
        return False
    n = min(n, len(good))
    sel = good[np.unique(np.round(np.linspace(0, len(good) - 1, n)).astype(int))]
    ncols = 4
    nrows = int(np.ceil(len(sel) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 4.6 * nrows + 1.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    C_POS, C_NEG = "#d95f02", "#1f78b4"
    for ax, k in zip(axes.flat, sel):
        psi = conv.sgn * np.radians(theta_deg[k])               # úhel konce 1 (CCW, y nahoru)
        r = np.array([np.cos(psi), np.sin(psi)])
        tan = np.array([-np.sin(psi), np.cos(psi)])             # tečna ve směru CCW
        tau = conv.sgn * torque[k] / p95                        # >0 = moment proti směru hod. ručiček
        col = C_POS if torque[k] >= 0 else C_NEG
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(-3.9, 2.05)
        ax.set_aspect("equal")
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="0.8", lw=1, ls="--"))
        ax.plot([-r[0], r[0]], [-r[1], r[1]], "-", color="0.15", lw=2.6, solid_capstyle="round",
                zorder=3)
        ax.plot(r[0], r[1], "o", ms=11, mfc="0.15", mec="k", zorder=4)
        ax.plot(-r[0], -r[1], "o", ms=8, mfc="white", mec="k", zorder=4)
        ax.plot(0, 0, "+", color="0.3", ms=8, zorder=4)
        ax.text(*(1.2 * r), "1", ha="center", va="center", fontsize=9, color="0.35")
        ax.text(*(-1.2 * r), "2", ha="center", va="center", fontsize=9, color="0.35")
        ln = float(np.clip(tau, -1.6, 1.6))
        if abs(ln) > 0.04:
            for side, tip in ((1.0, r), (-1.0, -r)):
                ax.annotate("", xy=tip + ln * side * tan, xytext=tip, zorder=6,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=3.4, shrinkA=0,
                                            shrinkB=0, mutation_scale=22))
        ccw = conv.sgn * torque[k] >= 0
        lines = [f"θ = {theta_deg[k]:+.0f}°      ω = {omega[k]:+.2f} rad/s",
                 f"α = {alpha[k]:+.2f} rad/s²",
                 f"{t_label} = {torque[k]:+.3g} {t_unit}   (na rameno {torque[k] / 2:+.3g})"]
        if force is not None and np.isfinite(force[k]):
            lines.append(f"síla na konec F = {force[k]:+.3g} N")
        lines.append("moment otáčí " + ("proti směru hod. ručiček" if ccw else "po směru hod. ručiček"))
        ax.text(0, -2.2, "\n".join(lines), ha="center", va="top", fontsize=9.3, linespacing=1.6,
                color="k")
        ax.set_title(f"t = {t[k]:.2f} s", fontsize=12, fontweight="bold")
    fig.suptitle("Moment síly na ramenech papírku v sérii okamžiků", fontsize=15,
                 fontweight="bold", y=0.995)
    handles = [Line2D([0], [0], color=C_POS, lw=3.4, label=f"{t_label} > 0 (moment ve směru rostoucího θ)"),
               Line2D([0], [0], color=C_NEG, lw=3.4, label=f"{t_label} < 0 (moment proti směru rostoucího θ)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.055))
    fig.text(0.5, 0.012,
             "Šipky na koncích ramen jsou tečné (kolmé na rameno) a na protilehlých koncích opačné – "
             "dvojice sil, která tvoří moment M = I·α. Délka šipky je úměrná |M| se stejným měřítkem ve "
             f"všech panelech: šipka dlouhá jako rameno = {p95:.3g} {t_unit}.\n"
             "Čistý moment se mezi ramena dělí rovným dílem (M/2 na rameno, symetrický papírek). "
             f"Velký plný puntík = označený konec 1.  Kladný směr θ: {pos_text(conv)}.",
             ha="center", va="bottom", fontsize=8.6, color="0.25", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    fig.savefig(out_png)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------
# 7b) Přesnost momentu a "proč se papírek točí"
# ----------------------------------------------------------------------------
def estimate_theta_noise(theta, valid):
    """Šum měření úhlu θ [rad] z robustní směrodatné odchylky třetích diferencí
    (bílý šum σ dává rozptyl 20 σ², hladký signál se třetí diferencí téměř neprojeví)."""
    ok = valid[:-3] & valid[1:-2] & valid[2:-1] & valid[3:]
    i = np.where(ok)[0]
    if len(i) < 15:
        return float("nan")
    d3 = theta[i + 3] - 3 * theta[i + 2] + 3 * theta[i + 1] - theta[i]
    d3 = d3[np.isfinite(d3)]
    if len(d3) < 15:
        return float("nan")
    mad = np.median(np.abs(d3 - np.median(d3)))
    return float(1.4826 * mad / np.sqrt(20.0))


def mc_alpha_std(theta, valid, segs, dt, win, poly, sigma, n_mc=40, seed=1):
    """Směrodatná odchylka úhlového zrychlení vlivem šumu v θ (Monte Carlo přes celý výpočet,
    včetně okrajů úseků a dopočítaných mezer)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_mc, len(theta)))
    for k in range(n_mc):
        th = theta + np.where(valid, rng.normal(0.0, sigma, len(theta)), 0.0)
        out[k] = sg_derivative(th, valid, segs, dt, win, poly, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanstd(out, axis=0)


def fold_stats(theta_deg, y, valid, nb):
    """Složí všechny otáčky do jedné periody θ mod 360° (nb intervalů).
    Vrací středy intervalů, průměr přes všechny otáčky, matici R[otáčka, interval] a čísla otáček."""
    ok = valid & np.isfinite(theta_deg) & np.isfinite(y)
    th = np.mod(theta_deg[ok], 360.0)
    rev = np.floor(theta_deg[ok] / 360.0).astype(int)
    yy = y[ok]
    b = np.minimum((th / (360.0 / nb)).astype(int), nb - 1)
    revs = np.unique(rev)
    R = np.full((len(revs), nb), np.nan)
    for ri, r in enumerate(revs):
        for bi in range(nb):
            m = (rev == r) & (b == bi)
            if m.any():
                R[ri, bi] = yy[m].mean()
    cnt = np.bincount(b, minlength=nb)
    mean = np.bincount(b, weights=yy, minlength=nb) / np.maximum(cnt, 1)
    mean[cnt == 0] = np.nan
    centers = (np.arange(nb) + 0.5) * 360.0 / nb
    return centers, mean, R, revs


def revolution_spread(theta_deg, y, valid, nb=12):
    """Relativní rozptyl mezi otáčkami: RMS (směrodatná odchylka mezi otáčkami při stejném θ)
    / RMS průměrné křivky. Menší = měření se z otáčky na otáčku lépe opakuje."""
    _, _, R, _ = fold_stats(theta_deg, y, valid, nb)
    R = R[np.sum(np.isfinite(R), axis=1) >= nb // 2]      # jen téměř úplné otáčky
    nrev = R.shape[0]
    if nrev < 2:
        return float("nan"), nrev
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmean(R, axis=0)
        cnt = np.sum(np.isfinite(R), axis=0)
        sd = np.where(cnt >= 2, np.nanstd(R, axis=0, ddof=1), np.nan)
    good = (cnt >= 2) & np.isfinite(m)
    if good.sum() < 3:
        return float("nan"), nrev
    amp = np.sqrt(np.mean((m[good] - m[good].mean()) ** 2))
    if not amp > 0:
        return float("nan"), nrev
    return float(np.sqrt(np.mean(sd[good] ** 2)) / amp), nrev


def compute_accuracy(theta_deg, valid, segs, dt, win_alpha, poly, speed, torque):
    """Odhad přesnosti momentu: šum, vliv vyhlazení a opakovatelnost mezi otáčkami."""
    n = len(theta_deg)
    theta = np.radians(theta_deg)
    poly_a = max(poly, 3)

    def odd(w):
        w = int(round(w))
        return max(5, w if w % 2 else w - 1)

    w0 = odd(win_alpha)
    sig = estimate_theta_noise(theta, valid)
    sig_alpha = (mc_alpha_std(theta, valid, segs, dt, w0, poly_a, sig)
                 if np.isfinite(sig) else np.full(n, np.nan))
    wins = {w: sg_derivative(theta, valid, segs, dt, w, poly_a, 2)
            for w in sorted({odd(w0 * 0.5), w0, odd(w0 * 2.0)})}
    _, edge = segment_flags(n, segs, max(wins) // 2)
    use = valid & ~edge
    p95 = {}
    for w, a in wins.items():
        m = use & np.isfinite(a)
        if m.sum() > 10:
            p95[w] = float(np.percentile(np.abs(a[m]), 95))

    sp = speed.copy()
    oks = valid & np.isfinite(sp)
    thr = 0.2 * np.nanmedian(sp[oks]) if oks.any() else 0.0
    sp[~(sp > thr)] = np.nan
    spread = {}
    for name, y in (("M", torque), ("M/|v|", torque / sp), ("M/|v|²", torque / sp ** 2)):
        spread[name] = revolution_spread(theta_deg, y, valid)
    return SimpleNamespace(sigma_theta=sig, sigma_alpha=sig_alpha, w0=w0, wins=wins, p95=p95,
                           spread=spread, dt=dt)


def plot_accuracy(t, torque, valid, segs, acc, fac, t_label, t_unit, out_png):
    """Obrázek 13: jak moc se dá momentu věřit."""
    dt = acc.dt
    fig = plt.figure(figsize=(15, 9.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], width_ratios=[1.0, 1.0, 1.25])
    ax = fig.add_subplot(gs[0, :])
    first = True
    for s, e in segs:
        ii = np.arange(s, e + 1)
        ii = ii[valid[ii] & np.isfinite(torque[ii])]
        if len(ii) < 2:
            continue
        band = fac * acc.sigma_alpha[ii]
        ax.fill_between(t[ii], torque[ii] - band, torque[ii] + band, color="C0", alpha=0.28, lw=0,
                        label="± 1 σ vlivem šumu měření" if first else None, zorder=2)
        ax.plot(t[ii], torque[ii], "-", color="C0", lw=1.7,
                label=f"použité vyhlazení (okno {acc.w0} snímků = {acc.w0 * dt:.2f} s)" if first else None,
                zorder=3)
        first = False
    others = [w for w in acc.wins if w != acc.w0]
    for w, col in zip(others, ("#d95f02", "#1b9e77")):
        first = True
        for s, e in segs:
            ii = np.arange(s, e + 1)
            ii = ii[valid[ii] & np.isfinite(acc.wins[w][ii])]
            if len(ii) < 2:
                continue
            ax.plot(t[ii], fac * acc.wins[w][ii], "--", color=col, lw=1.1,
                    label=f"okno {w} snímků = {w * dt:.2f} s" if first else None, zorder=2)
            first = False
    half = acc.w0 // 2
    lab = True
    for s, e in segs:
        for a_, b_ in ((s, min(s + half, e)), (max(e - half, s), e)):
            ax.axvspan(t[a_], t[b_], color="#d73027", alpha=0.08, lw=0,
                       label="okraj úseku (nejméně spolehlivé)" if lab else None)
            lab = False
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(f"moment síly {t_label} [{t_unit}]")
    ax.set_title("Moment síly v čase: pás šumu a vliv vyhlazení")
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.92)

    # (b) výška peaků podle okna
    axb = fig.add_subplot(gs[1, 0])
    ws = sorted(acc.p95)
    vals = [fac * acc.p95[w] for w in ws]
    axb.bar(range(len(ws)), vals, color=["C0" if w == acc.w0 else "0.72" for w in ws])
    for i_, v in enumerate(vals):
        axb.text(i_, v, f"{v:.3g}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axb.set_xticks(range(len(ws)))
    axb.set_xticklabels([f"okno {w}\n({w * dt:.2f} s)" for w in ws])
    axb.set_ylabel(f"typická výška peaků  p95 |{t_label}| [{t_unit}]")
    axb.set_title("Výška peaků závisí na vyhlazení", fontsize=11.5)
    axb.grid(axis="x", alpha=0)

    # (c) opakovatelnost mezi otáčkami
    axc = fig.add_subplot(gs[1, 1])
    names = list(acc.spread)
    vv = [acc.spread[k][0] for k in names]
    nrev = max(acc.spread[k][1] for k in names)
    if np.any(np.isfinite(vv)):
        best = int(np.nanargmin(vv))
        axc.bar(range(len(names)), [0 if not np.isfinite(v) else 100 * v for v in vv],
                color=["#1a9850" if i_ == best else "0.72" for i_ in range(len(names))])
        for i_, v in enumerate(vv):
            if np.isfinite(v):
                axc.text(i_, 100 * v, f"{100 * v:.0f} %", ha="center", va="bottom", fontsize=10,
                         fontweight="bold")
        axc.set_xticks(range(len(names)))
        axc.set_xticklabels(names)
        axc.set_ylabel("rozptyl mezi otáčkami [% amplitudy]")
        axc.set_title("Opakovatelnost z otáčky na otáčku\n(menší = lepší; zelená = nejlepší)", fontsize=11.5)
    else:
        axc.axis("off")
        axc.text(0.5, 0.5, f"Málo úplných otáček ({nrev}) –\nopakovatelnost nelze určit.\n(potřeba ≥ 2)",
                 ha="center", va="center", fontsize=11, transform=axc.transAxes)
    axc.grid(axis="x", alpha=0)

    # (d) shrnutí
    axd = fig.add_subplot(gs[1, 2])
    axd.axis("off")
    ok_m = valid & np.isfinite(torque)
    amp = float(np.percentile(np.abs(fac * np.where(ok_m, torque / fac, np.nan)[ok_m]), 95)) if ok_m.any() else float("nan")
    sig_med = float(np.nanmedian(fac * acc.sigma_alpha[valid])) if np.any(np.isfinite(acc.sigma_alpha[valid])) else float("nan")
    L = []
    L.append(("Jak moc se dá momentu věřit", "h"))
    if np.isfinite(acc.sigma_theta):
        L.append((f"Šum měření úhlu:  σθ ≈ {np.degrees(acc.sigma_theta):.2f}°", ""))
    if np.isfinite(sig_med) and amp > 0:
        L.append((f"Šum v momentu:  σ ≈ {sig_med:.2g} {t_unit}  =  {100 * sig_med / amp:.0f} % typického peaku", ""))
    if len(acc.p95) >= 2:
        pv = [acc.p95[w] for w in ws]
        ref = acc.p95.get(acc.w0, np.mean(pv))
        L.append((f"Vyhlazení: peaky se mezi okny {ws[0]}–{ws[-1]} mění o {100 * (max(pv) - min(pv)) / ref:.0f} %", ""))
    if np.isfinite(vv[0]):
        L.append((f"Opakovatelnost mezi otáčkami: {100 * vv[0]:.0f} % ({acc.spread['M'][1]} otáček)", ""))
        if np.all(np.isfinite(vv)) and min(vv[1:]) < 0.9 * vv[0]:
            L.append((f"→ po normování na {names[1 + int(np.argmin(vv[1:]))]} je rozptyl menší: "
                      f"{100 * min(vv[1:]):.0f} %", ""))
        elif np.all(np.isfinite(vv)):
            L.append(("→ normování na rychlost rozptyl výrazně nesnižuje", ""))
    L.append(("Nezahrnuto (nejde zjistit z videa):", "h2"))
    L.append(("• M = I·α: chyba I jde přímo do M (±10 % v I = ±10 % v M)", ""))
    L.append(("• geometrie kamery, perspektiva, rolling shutter telefonu", ""))
    L.append(("• předpoklad: papírek tuhý, střed = polovina mezi tečkami", ""))
    y = 0.98
    for txt, kind in L:
        wrapped = textwrap.fill(txt, 52 if kind == "" else 60)
        nl = wrapped.count("\n") + 1
        axd.text(0.0, y, wrapped, va="top", ha="left", fontsize=12 if kind == "h" else 10.2,
                 fontweight="bold" if kind in ("h", "h2") else "normal", transform=axd.transAxes,
                 linespacing=1.35)
        y -= (0.075 if kind == "h" else 0.058) * nl + 0.03
    fig.text(0.012, 0.008,
             "Šum je odhadnut z vlastního měření (třetí diference úhlu) a přenesen přes celé vyhlazení "
             "metodou Monte Carlo. Červeně podbarvené okraje souvislých úseků detekce jsou nejméně spolehlivé.",
             fontsize=8.5, color="0.3", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_mechanism(theta_deg, omega, alpha, torque, valid, conv, t_label, t_unit, inertia,
                   out_png, nb=36):
    """Obrázek 12: proč se papírek točí dál. Všechny otáčky složené do jedné; moment a úhlová
    rychlost ve směru otáčení, zóny roztáčení / brzdění a energetická bilance za otáčku."""
    ok = valid & np.isfinite(theta_deg) & np.isfinite(omega) & np.isfinite(alpha) & np.isfinite(torque)
    if ok.sum() < 30:
        return False
    rot = 1.0 if np.median(omega[ok]) >= 0 else -1.0
    GREEN, RED = "#1a9850", "#d73027"
    cen, Mm, RM, _ = fold_stats(theta_deg, rot * torque, ok, nb)
    _, Wm_, RW, _ = fold_stats(theta_deg, rot * omega, ok, nb)
    _, Am, _, _ = fold_stats(theta_deg, rot * alpha, ok, nb)
    rowsM = np.sum(np.isfinite(RM), axis=1) >= nb // 2
    rowsW = np.sum(np.isfinite(RW), axis=1) >= nb // 2

    fig = plt.figure(figsize=(15.5, 9.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.1, 1.0], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    # --- moment ve směru otáčení ----------------------------------------------------
    for r in np.where(rowsM)[0]:
        ax1.plot(cen, RM[r], "-", color="0.7", lw=1.0, zorder=2)
    ax1.fill_between(cen, Mm, 0, where=np.nan_to_num(Mm) > 0, color=GREEN, alpha=0.35,
                     interpolate=True, lw=0, zorder=1)
    ax1.fill_between(cen, Mm, 0, where=np.nan_to_num(Mm) < 0, color=RED, alpha=0.30,
                     interpolate=True, lw=0, zorder=1)
    ax1.plot(cen, Mm, "-", color="k", lw=2.6, zorder=5)
    ax1.axhline(0, color="k", lw=0.9, alpha=0.6)
    top = np.nanmax(np.abs(Mm)) * 1.0 if np.any(np.isfinite(Mm)) else 1.0
    ax1.text(0.012, 0.93, "ROZTÁČÍ  (moment ve směru otáčení)", transform=ax1.transAxes, color=GREEN,
             fontsize=11, fontweight="bold", va="top")
    ax1.text(0.012, 0.07, "BRZDÍ  (moment proti směru otáčení)", transform=ax1.transAxes, color=RED,
             fontsize=11, fontweight="bold", va="bottom")
    ax1.set_ylabel(f"moment síly {t_label} [{t_unit}]\n(kladný = ve směru otáčení)")
    ax1.set_xlim(0, 360)
    ax1.xaxis.set_major_locator(MultipleLocator(45))
    ax1.xaxis.set_major_formatter(DEG_FMT)
    add_pose_icons_top(ax1, list(range(0, 360, 45)), conv)
    ax1.set_title("Kdy moment papírek roztáčí a kdy brzdí  (všechny otáčky složeny do jedné)", pad=62)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # --- úhlová rychlost ve směru otáčení ---------------------------------------------
    for r in np.where(rowsW)[0]:
        ax2.plot(cen, RW[r], "-", color="0.7", lw=1.0, zorder=2)
    ax2.plot(cen, Wm_, "-", color="k", lw=2.6, zorder=5)
    wmax = float(np.nanmax(Wm_)) if np.any(np.isfinite(Wm_)) else 1.0
    wmin = float(np.nanmin(Wm_)) if np.any(np.isfinite(Wm_)) else 0.0
    ax2.axhline(0, color=RED, lw=1.8, ls="--", zorder=3)
    ax2.text(0.99, 0.02, "ω = 0  (papírek by se zastavil)", transform=ax2.transAxes, ha="right",
             va="bottom", color=RED, fontsize=10, fontweight="bold")
    imin, imax = int(np.nanargmin(Wm_)), int(np.nanargmax(Wm_))
    ax2.plot(cen[imin], wmin, "o", ms=9, mfc="white", mec="k", mew=1.8, zorder=6)
    ax2.annotate(f"nejpomalejší: {wmin:.2f} rad/s", (cen[imin], wmin),
                 xytext=(-12 if cen[imin] > 250 else 12, 16), textcoords="offset points",
                 ha="right" if cen[imin] > 250 else "left", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5"))
    ax2.plot(cen[imax], wmax, "o", ms=9, mfc="white", mec="k", mew=1.8, zorder=6)
    ax2.annotate(f"nejrychlejší: {wmax:.2f} rad/s", (cen[imax], wmax),
                 xytext=(-12 if cen[imax] > 250 else 12, 18), textcoords="offset points",
                 ha="right" if cen[imax] > 250 else "left", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5"))
    ax2.set_ylim(min(-0.12 * wmax, wmin * 1.2), wmax * 1.22)
    ax2.set_ylabel("úhlová rychlost ve směru otáčení [rad/s]")
    ax2.set_xlabel(f"natočení θ [°]     (papírek se otáčí ve směru "
                   f"{'rostoucího' if rot > 0 else 'klesajícího'} θ)")
    ax2.xaxis.set_major_formatter(DEG_FMT)

    # --- bilance energie za otáčku -------------------------------------------------------
    dphi = np.radians(360.0 / nb)
    cov = float(np.mean(np.isfinite(Am)))
    Wp = float(np.nansum(np.clip(Am, 0, None)) * dphi)
    Wn = float(np.nansum(np.clip(Am, None, 0)) * dphi)
    net = Wp + Wn
    k_en, u_en = (inertia * 1000.0, "mJ") if inertia else (1.0, "rad²/s²  (= energie / I)")
    sub = gs[:, 1].subgridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.28)
    ax3 = fig.add_subplot(sub[0])
    vals = [k_en * Wp, k_en * Wn, k_en * net]
    ax3.bar(range(3), vals, color=[GREEN, RED, "0.45"])
    for i_, v in enumerate(vals):
        ax3.text(i_, v, f"{v:+.3g}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10.5,
                 fontweight="bold")
    ax3.axhline(0, color="k", lw=0.9)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(["roztáčení", "brzdění", "čistě\nza otáčku"])
    ax3.set_ylabel(f"práce za otáčku [{u_en}]")
    ax3.set_title("Energetická bilance jedné otáčky", fontsize=11.5)
    ax3.grid(axis="x", alpha=0)
    lim = max(abs(v) for v in vals) * 1.35 or 1.0
    ax3.set_ylim(-lim, lim)

    ax4 = fig.add_subplot(sub[1])
    ax4.axis("off")
    total = Wp - Wn
    ratio = net / total if total > 0 else float("nan")
    frac_up = float(np.mean(Am[np.isfinite(Am)] > 0)) if np.any(np.isfinite(Am)) else float("nan")
    L = [f"Roztáčí na {100 * frac_up:.0f} % otáčky, brzdí na {100 * (1 - frac_up):.0f} %."]
    if wmin > 0:
        L.append(f"Úhlová rychlost nikdy neklesne k nule (min {wmin:.2f} rad/s): brzdění papírek "
                 f"zpomalí, ale setrvačnost ho přenese přes celé brzdné místo.")
    else:
        L.append("Úhlová rychlost v některé fázi klesá k nule – papírek se tam zastavuje.")
    if np.isfinite(ratio):
        if abs(ratio) < 0.15:
            L.append(f"Roztáčení a brzdění se za otáčku vyrovnají (zbytek {100 * ratio:+.0f} % z celkové "
                     f"práce) → rychlost otáčení se v průměru nemění, papírek se točí dál.")
        elif ratio > 0:
            L.append(f"Roztáčení převažuje ({100 * ratio:+.0f} %) → papírek se v průběhu měření roztáčí.")
        else:
            L.append(f"Brzdění převažuje ({100 * ratio:+.0f} %) → papírek v průběhu měření zpomaluje.")
    if cov < 0.9:
        L.append(f"Pozor: pokryto jen {100 * cov:.0f} % úhlů otáčky.")
    y = 0.99
    for txt in L:
        wrapped = textwrap.fill(txt, 40)
        ax4.text(0.0, y, wrapped, va="top", ha="left", fontsize=10.2, transform=ax4.transAxes,
                 linespacing=1.4)
        y -= 0.062 * (wrapped.count("\n") + 1) + 0.035

    fig.text(0.012, 0.008,
             "Šedé čáry = jednotlivé otáčky, černá = průměr přes otáčky (intervaly po "
             f"{360 // nb}°). Moment i ω jsou brány ve směru otáčení papírku. Činky nad grafem ukazují, "
             "jak papírek při daném θ vypadá. Jak moc se momentu dá věřit: viz obrázek 13.",
             fontsize=8.6, color="0.3", ha="left", va="bottom")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_png)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------
# 8) Kontrolní video
# ----------------------------------------------------------------------------
def draw_panel(img, lines, fs, thick):
    """Ztmavený panel s textem v levém horním rohu (lines = [(text, BGR barva), …])."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(tx, font, fs, thick)[0] for tx, _ in lines]
    lh = int(max(sz[1] for sz in sizes) * 1.75)
    pw = max(sz[0] for sz in sizes) + 16
    ph = lh * len(lines) + 10
    roi = img[6:6 + ph, 6:6 + pw]
    roi[:] = (roi.astype(np.float32) * 0.45).astype(np.uint8)       # ztmavení pod textem
    for j, (tx, col) in enumerate(lines):
        cv2.putText(img, tx, (14, 6 + 5 + lh * (j + 1) - int(lh * 0.28)), font, fs, col,
                    thick, cv2.LINE_AA)


def draw_air_arrow(img, c, vx, vy, k, s, font):
    """Bílá šipka proudění vzduchu vůči činky: vychází ze středu `c` a míří PROTI rychlosti
    středu (vx, vy; y nahoru). Při volném pádu papíru tedy míří nahoru. Délka je úměrná |v|
    (stejné měřítko `k` jako u šipky rychlosti), takže se mění s rychlostí pádu."""
    if k <= 0 or not (np.isfinite(vx) and np.isfinite(vy)):
        return
    ci = (int(round(c[0])), int(round(c[1])))
    tip = (int(round(ci[0] - vx * k)), int(round(ci[1] + vy * k)))   # opačně než pohyb; y obrazu je dolů
    if np.hypot(tip[0] - ci[0], tip[1] - ci[1]) < 3:
        return
    th = max(3, int(round(3.5 * s)))
    cv2.arrowedLine(img, ci, tip, (0, 0, 0), th + 3, cv2.LINE_AA, tipLength=0.22)   # obrys
    cv2.arrowedLine(img, ci, tip, (255, 255, 255), th, cv2.LINE_AA, tipLength=0.22)
    tx = tip[0] + int(8 * s) if tip[0] >= ci[0] else tip[0] - int(125 * s)
    ty = tip[1] - int(8 * s) if tip[1] <= ci[1] else tip[1] + int(18 * s)
    tk = max(1, int(round(1.4 * s)))
    for col, t_ in (((0, 0, 0), tk + 2), ((255, 255, 255), tk)):
        cv2.putText(img, "AIR FLOW", (tx, ty), font, 0.6 * s, col, t_, cv2.LINE_AA)


def write_debug_video(path, out_path, A, B, valid, kin, segs, half, gain=6.0):
    """Zesvětlené video s trackem. Ze středu činky vychází purpurová šipka rychlosti (stejné
    měřítko v celém videu, takže se plynule mění; měřítko v levém dolním rohu). Všechny ostatní
    veličiny se vypisují číselně do panelu v levém horním rohu (OpenCV umí jen ASCII).
    `half` = polovina okna vyhlazení úhlového zrychlení; snímky blíž ke kraji souvislého úseku
    jsou označeny EDGE (tam je druhá derivace nejméně spolehlivá)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = len(valid)
    s = max(0.7, h / 1000.0)                       # měřítko písma a čar podle rozlišení
    fs = 0.55 * s                                  # písmo panelu
    thick = max(1, int(round(1.4 * s)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    WHITE, GREY, C_V = (255, 255, 255), (170, 170, 170), (255, 0, 255)   # BGR, C_V = purpurová
    u = kin.unit
    t_unit = kin.t_unit.replace("·", ".").replace("s⁻²", "1/s^2")

    seg_of, edge = segment_flags(n, segs, half)

    xs_ = kin.speed[valid & np.isfinite(kin.speed)]
    pv = float(np.percentile(xs_, 95)) if len(xs_) else 0.0
    kv = 0.18 * min(w, h) / pv if pv > 0 else 0.0  # px na jednotku (unit/s)
    refv = nice_number(pv) if pv > 0 else 0.0

    def fmt(v, spec):
        return "n/a" if not np.isfinite(v) else format(v, spec)

    def arrow(img, c, vx, vy, k, color):
        if k <= 0 or not (np.isfinite(vx) and np.isfinite(vy)):
            return
        tip = (int(round(c[0] + vx * k)), int(round(c[1] - vy * k)))   # y obrazu je dolů
        if np.hypot(tip[0] - c[0], tip[1] - c[1]) >= 3:
            cv2.arrowedLine(img, c, tip, color, max(2, int(round(2.5 * s))), cv2.LINE_AA,
                            tipLength=0.22)

    trail, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        have = i < n and valid[i]
        if have:
            a = tuple(np.round(kin.T1[i]).astype(int))        # konec 1 = označený (červený)
            b = tuple(np.round(kin.T2[i]).astype(int))        # konec 2 (zelený)
            c = (int(round((a[0] + b[0]) / 2)), int(round((a[1] + b[1]) / 2)))
            trail.append(c)
            cv2.circle(f, a, 9, (0, 0, 255), 2)
            cv2.circle(f, b, 8, (0, 255, 0), 1)
            cv2.putText(f, "1", (a[0] + 12, a[1] - 10), font, 0.6 * s, (0, 0, 255), thick, cv2.LINE_AA)
            cv2.putText(f, "2", (b[0] + 12, b[1] - 10), font, 0.6 * s, (0, 255, 0), thick, cv2.LINE_AA)
            cv2.line(f, a, b, (255, 255, 0), 1)
            cv2.circle(f, c, 3, (0, 255, 255), -1)
        for p, q in zip(trail[:-1], trail[1:]):
            cv2.line(f, p, q, (0, 200, 255), 1)
        if have:
            arrow(f, c, kin.vx[i], kin.vy[i], kv, C_V)
            draw_air_arrow(f, c, kin.vx[i], kin.vy[i], kv, s, font)

        # panel s číselnými hodnotami -------------------------------------------------
        lines = [(f"frame {i}/{n - 1}    t = {kin.t[i]:.3f} s" if i < n else f"frame {i}", WHITE)]
        if have:
            lines.append((f"detection OK   segment {seg_of[i]}/{len(segs)}"
                          + ("   EDGE (alpha, M less reliable)" if edge[i] else ""),
                          (80, 200, 255) if edge[i] else WHITE))
            dist = float(np.hypot(B[i, 0] - A[i, 0], B[i, 1] - A[i, 1]))
            lines += [
                ("-- measured in video (px, image coords) --", GREY),
                (f"end 1 (marked) = ({kin.T1[i, 0]:.1f}, {kin.T1[i, 1]:.1f})", WHITE),
                (f"end 2          = ({kin.T2[i, 0]:.1f}, {kin.T2[i, 1]:.1f})", WHITE),
                (f"distance of ends = {dist:.1f} px", WHITE),
                (f"centre = ({kin.com[i, 0]:.2f}, {kin.com[i, 1]:.2f}) {u}  (y up)", WHITE),
                ("-- rotation (angle of end 1 around the centre) --", GREY),
                (f"theta = {fmt(kin.theta[i], '+.1f')} deg = {fmt(kin.turns[i], '+.3f')} rev"
                 f"   (mod 360: {fmt(kin.theta[i] % 360, '.1f')})", WHITE),
                (f"omega = {fmt(kin.omega[i], '+.3f')} rad/s", WHITE),
                (f"alpha = {fmt(kin.alpha[i], '+.3f')} rad/s^2", WHITE),
                (f"{kin.t_label} = {fmt(kin.torque[i], '+.4g')} {t_unit}", WHITE),
                ("-- centre motion (derived from centre position) --", GREY),
                (f"|v| = {fmt(kin.speed[i], '.2f')} {u}/s   dir = {fmt(kin.v_dir[i], '.0f')} deg", C_V),
                (f"vx = {fmt(kin.vx[i], '+.2f')}   vy = {fmt(kin.vy[i], '+.2f')} {u}/s", C_V),
                (f"|a| = {fmt(kin.a_mag[i], '.2f')} {u}/s^2   dir = {fmt(kin.a_dir[i], '.0f')} deg", WHITE),
                (f"ax = {fmt(kin.ax[i], '+.2f')}   ay = {fmt(kin.ay[i], '+.2f')} {u}/s^2", WHITE),
            ]
            lines.append((f"air flow dir = {fmt((kin.v_dir[i] + 180.0) % 360.0, '.0f')} deg"
                          "  (opposite to v, white arrow)", WHITE))
        else:
            lines.append(("NO DETECTION (gap)", (200, 200, 200)))
        draw_panel(f, lines, fs, thick)

        # měřítko šipky rychlosti (levý dolní roh)
        if kv > 0:
            yy = h - int(24 * s)
            L_ = int(round(refv * kv))
            cv2.arrowedLine(f, (14, yy), (14 + L_, yy), C_V, max(2, int(round(2.5 * s))),
                            cv2.LINE_AA, tipLength=0.22)
            cv2.putText(f, f"= {refv:g} {u}/s  (velocity arrow)", (24 + L_, yy + int(6 * s)),
                        font, 0.6 * s, C_V, thick, cv2.LINE_AA)
        vw.write(f)
        i += 1
    cap.release()
    vw.release()


def write_torque_video(path, out_path, A, B, valid, kin, segs, half, gain=6.0):
    """Video se šipkami momentu síly na obou ramenech papírku. Šipka je tečná (kolmá na rameno),
    na protilehlých koncích opačná (dvojice sil), délka úměrná |M| (stejné měřítko v celém videu).
    Oranžová = moment ve směru rostoucího θ, modrá = proti. Konec 1 (označený) je červený plný,
    konec 2 zelený. Čistý moment se mezi ramena dělí rovným dílem (M/2 na rameno)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = len(valid)
    s = max(0.7, h / 1000.0)
    fs, thick = 0.55 * s, max(1, int(round(1.4 * s)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    WHITE, GREY = (255, 255, 255), (170, 170, 170)
    C_POS, C_NEG = (0, 128, 255), (255, 140, 30)               # BGR: oranžová, modrá
    t_unit = kin.t_unit.replace("·", ".").replace("s⁻²", "1/s^2")
    seg_of, edge = segment_flags(n, segs, half)

    okt = valid & np.isfinite(kin.torque)
    p95 = float(np.percentile(np.abs(kin.torque[okt]), 95)) if okt.any() else 0.0
    Lref = 0.5 * kin.L_px                                      # šipka při |M| = p95 = délka ramene
    thick_a = max(3, int(round(3.5 * s)))
    xs_ = kin.speed[valid & np.isfinite(kin.speed)]
    pv = float(np.percentile(xs_, 95)) if len(xs_) else 0.0
    kv = 0.18 * min(w, h) / pv if pv > 0 else 0.0                # px na (unit/s), stejně jako v debug videu

    def fmt(v, spec):
        return "n/a" if not np.isfinite(v) else format(v, spec)

    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        have = i < n and valid[i]
        lines = [(f"frame {i}/{n - 1}    t = {kin.t[i]:.3f} s" if i < n else f"frame {i}", WHITE)]
        if have:
            T1, T2 = kin.T1[i], kin.T2[i]
            c = (T1 + T2) / 2.0
            ci = tuple(np.round(c).astype(int))
            cv2.circle(f, ci, int(round(Lref)), (90, 90, 90), 1, cv2.LINE_AA)   # dráha konců
            cv2.line(f, tuple(np.round(T1).astype(int)), tuple(np.round(T2).astype(int)),
                     (255, 255, 0), 1)
            cv2.circle(f, ci, 3, (0, 255, 255), -1)
            draw_air_arrow(f, c, kin.vx[i], kin.vy[i], kv, s, font)
            m = kin.torque[i]
            if p95 > 0 and np.isfinite(m):
                tau = float(np.clip(kin.sgn * m / p95, -1.6, 1.6))   # >0 = proti směru hod. ručiček
                col = C_POS if m >= 0 else C_NEG
                for T in (T1, T2):
                    r = T - c
                    nr = float(np.hypot(*r))
                    if nr < 1e-6:
                        continue
                    r = r / nr
                    vec = tau * Lref * np.array([r[1], -r[0]])         # tečna CCW v obraze (y dolů)
                    if np.hypot(*vec) >= 3:
                        cv2.arrowedLine(f, tuple(np.round(T).astype(int)),
                                        tuple(np.round(T + vec).astype(int)), col, thick_a,
                                        cv2.LINE_AA, tipLength=0.3)
            a_ = tuple(np.round(T1).astype(int))
            b_ = tuple(np.round(T2).astype(int))
            cv2.circle(f, a_, 8, (0, 0, 255), -1)
            cv2.circle(f, b_, 8, (0, 255, 0), 2)
            cv2.putText(f, "1", (a_[0] + 12, a_[1] - 10), font, 0.6 * s, (0, 0, 255), thick, cv2.LINE_AA)
            cv2.putText(f, "2", (b_[0] + 12, b_[1] - 10), font, 0.6 * s, (0, 255, 0), thick, cv2.LINE_AA)

            lines.append((f"detection OK   segment {seg_of[i]}/{len(segs)}"
                          + ("   EDGE (alpha, M less reliable)" if edge[i] else ""),
                          (80, 200, 255) if edge[i] else WHITE))
            lines.append((f"air flow dir = {fmt((kin.v_dir[i] + 180.0) % 360.0, '.0f')} deg"
                          "  (opposite to v, white arrow)", WHITE))
            lines += [
                (f"theta = {fmt(kin.theta[i], '+.1f')} deg   omega = {fmt(kin.omega[i], '+.3f')} rad/s", WHITE),
                (f"alpha = {fmt(kin.alpha[i], '+.3f')} rad/s^2", WHITE),
                (f"{kin.t_label} = {fmt(m, '+.4g')} {t_unit}   (net torque)", C_POS if m >= 0 else C_NEG),
                (f"per arm {kin.t_label}/2 = {fmt(m / 2, '+.4g')} {t_unit}", WHITE),
            ]
            if kin.force is not None:
                lines.append((f"force on each end F = {fmt(kin.force[i], '+.4g')} N", WHITE))
            if np.isfinite(m):
                lines.append(("torque turns "
                              + ("COUNTER-CLOCKWISE" if kin.sgn * m >= 0 else "CLOCKWISE"), WHITE))
        else:
            lines.append(("NO DETECTION (gap)", (200, 200, 200)))
        draw_panel(f, lines, fs, thick)

        # měřítko a barvy (levý dolní roh)
        if p95 > 0:
            yb = h - int(24 * s)
            L_ = int(round(Lref))
            cv2.arrowedLine(f, (14, yb), (14 + L_, yb), WHITE, thick_a, cv2.LINE_AA, tipLength=0.3)
            cv2.putText(f, f"= {p95:.3g} {t_unit}  (|{kin.t_label}|, scale)", (24 + L_, yb + int(6 * s)),
                        font, 0.6 * s, WHITE, thick, cv2.LINE_AA)
            cv2.putText(f, f"orange: {kin.t_label} > 0 (towards +theta)", (14, yb - int(58 * s)),
                        font, 0.6 * s, C_POS, thick, cv2.LINE_AA)
            cv2.putText(f, f"blue: {kin.t_label} < 0 (against +theta)", (14, yb - int(30 * s)),
                        font, 0.6 * s, C_NEG, thick, cv2.LINE_AA)
        vw.write(f)
        i += 1
    cap.release()
    vw.release()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--outdir", default=None,
                    help="kam se vytvoří složka se jménem videa (default: aktuální složka)")
    ap.add_argument("--thr", type=float, default=12.0,
                    help="práh jasu nad pozadím (nižší = citlivější, více šumu) [12]")
    ap.add_argument("--min-area", type=int, default=2, help="min. plocha skvrny v px [2]")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="povolená odchylka vzdálenosti teček od L (relativně) [0.15]")
    ap.add_argument("--gap-max", type=int, default=3,
                    help="max. mezera (snímky), přes kterou se dráha spojí čarou [3]")
    ap.add_argument("--dot-distance-mm", type=float, default=None,
                    help="skutečná vzdálenost teček v mm -> osy v mm místo pixelů")
    ap.add_argument("--ticks-every", type=int, default=8,
                    help="každý n-tý platný snímek vykreslit jako 'činku' v grafu dráhy [8]")
    ap.add_argument("--smooth-win", type=int, default=11,
                    help="okno vyhlazení (snímky, liché) pro rychlost a ω [11]")
    ap.add_argument("--smooth-win-alpha", type=int, default=21,
                    help="okno vyhlazení (snímky, liché) pro úhlové zrychlení [21]")
    ap.add_argument("--smooth-poly", type=int, default=3,
                    help="stupeň polynomu Savitzkého–Golayova filtru [3]")
    ap.add_argument("--inertia", type=float, default=None,
                    help="moment setrvačnosti I [kg·m²] -> moment síly M = I·α v N·m; "
                         "bez něj se kreslí M/I [s⁻²]")
    ap.add_argument("--positive", choices=("ccw", "cw"), default="ccw",
                    help="kladný směr natočení: ccw = proti směru hodinových ručiček [výchozí], "
                         "cw = po směru")
    ap.add_argument("--snapshots", type=int, default=12,
                    help="počet okamžiků v obrázku 10_cinka_v_case [12]")
    ap.add_argument("--debug-video", action="store_true", help="uložit kontrolní video")
    ap.add_argument("--torque-video", action="store_true",
                    help="uložit video se šipkami momentu síly na obou ramenech")
    args = ap.parse_args()

    folder, paths = prepare_outputs(args.video, args.outdir)
    print("Výstupní složka:", folder)

    print("Odhaduji pozadí …")
    bg, n_frames = estimate_background(args.video)
    print(f"Hledám tečky ve {n_frames} snímcích …")
    all_blobs, fps = collect_blobs(args.video, bg, args.thr, args.min_area)

    L = estimate_distance(all_blobs)
    print(f"Odhad vzdálenosti teček: {L:.1f} px")
    A, B = pick_pairs(all_blobs, L, args.tol)
    valid = ~np.isnan(A[:, 0])
    print(f"Platná dvojice nalezena v {valid.sum()} z {len(valid)} snímků "
          f"({valid.sum() / fps:.1f} s z {len(valid) / fps:.1f} s)")

    dt = 1.0 / fps
    t = np.arange(len(A)) * dt
    com, phi, valid, raw0 = compute_pose(A, B, t, gap_max=args.gap_max)
    segs = segments_from_valid(valid, args.gap_max)

    # konvence natočení: θ = úhel označeného konce od vodorovné osy x (0° = doprava),
    # kladný směr dle --positive. Úhel označeného konce v obraze (CCW, y nahoru) = raw0 + φ,
    # kde φ je plynule odvinuté natočení od 1. snímku a raw0 ∈ <0°,180°) úhel v 1. snímku.
    sgn = 1 if args.positive == "ccw" else -1
    conv = SimpleNamespace(raw0=0.0, sgn=sgn)       # raw0 = 0 → nulový úhel = osa x
    # θ se počítá z polohy ZNAČENÉHO KONCE T1 kolem středu (střed = polovina mezi tečkami);
    # φ z compute_pose jen určuje, která z dvou teček je T1 (kontinuita po snímcích)
    psi_guess = raw0 + phi
    cimg = (A + B) / 2.0
    T1, T2 = assign_tips(A, B, psi_guess)
    ang_tip = np.arctan2(-(T1[:, 1] - cimg[:, 1]), T1[:, 0] - cimg[:, 0])   # (-π, π], y nahoru
    psi = ang_tip + 2 * np.pi * np.round((psi_guess - ang_tip) / (2 * np.pi))  # plynulé odvinutí
    theta = sgn * psi

    if args.dot_distance_mm:
        scale, unit = args.dot_distance_mm / L, "mm"
    else:
        scale, unit = 1.0, "px"

    speed, omega, alpha, torque, t_label, t_unit, vec = compute_kinematics(
        com * scale, theta, valid, segs, dt,
        args.smooth_win, args.smooth_win_alpha, args.smooth_poly, args.inertia)
    vx, vy, acx, acy = vec
    a_mag = np.hypot(acx, acy)
    # směr vektorů: úhel od osy x (y nahoru), kladný směr stejný jako u θ (--positive)
    v_dir = np.mod(sgn * np.degrees(np.arctan2(vy, vx)), 360.0)
    a_dir = np.mod(sgn * np.degrees(np.arctan2(acy, acx)), 360.0)
    # síla na každém konci: moment = dvojice sil F, vzdálených L  ⇒  F = M / L (jen s --inertia a --dot-distance-mm)
    force_arm = (torque / (args.dot_distance_mm / 1000.0)
                 if (args.inertia and args.dot_distance_mm) else None)

    theta_deg = np.degrees(theta)
    turns = theta / (2 * np.pi)
    v_unit = f"{unit}/s"
    idx = np.where(valid)[0]

    # CSV --------------------------------------------------------------------
    torque_col = "torque_Nm" if args.inertia else "torque_over_I_s-2"   # čistě ASCII název sloupce
    with open(paths["csv"], "w", encoding="utf-8", newline="") as fh:
        fh.write("frame,t_s,Ax_px,Ay_px,Bx_px,By_px,"
                 f"com_x_{unit},com_y_{unit},theta_total_deg,theta_mod360_deg,turns,"
                 f"speed_{unit}_per_s,omega_rad_s,alpha_rad_s2,"
                 f"{torque_col},"
                 f"vx_{unit}_per_s,vy_{unit}_per_s,v_dir_deg,"
                 f"ax_{unit}_per_s2,ay_{unit}_per_s2,a_mag_{unit}_per_s2,a_dir_deg,"
                 "T1x_px,T1y_px,T2x_px,T2y_px,force_per_end_N\n")
        for i in idx:
            f = lambda v, p: "" if not np.isfinite(v) else f"{v:.{p}f}"
            fh.write(f"{i},{t[i]:.4f},{A[i,0]:.2f},{A[i,1]:.2f},{B[i,0]:.2f},{B[i,1]:.2f},"
                     f"{com[i,0] * scale:.3f},{com[i,1] * scale:.3f},"
                     f"{theta_deg[i]:.2f},{theta_deg[i] % 360:.2f},{turns[i]:.4f},"
                     f"{f(speed[i], 4)},{f(omega[i], 4)},{f(alpha[i], 3)},{f(torque[i], 6)},"
                     f"{f(vx[i], 4)},{f(vy[i], 4)},{f(v_dir[i], 2)},"
                     f"{f(acx[i], 3)},{f(acy[i], 3)},{f(a_mag[i], 3)},{f(a_dir[i], 2)},"
                     f"{T1[i,0]:.2f},{T1[i,1]:.2f},{T2[i,0]:.2f},{T2[i,1]:.2f},"
                     f"{f(force_arm[i], 6) if force_arm is not None else ''}\n")
    print("Uloženo:", paths["csv"])

    # Grafy ------------------------------------------------------------------
    plot_xy(com, theta_deg, conv, valid, A, B, L, scale, unit, paths["xy"],
            args.gap_max, args.ticks_every)

    plot_rotation_time(t, theta_deg, valid, segs, conv, paths["phi_t"])

    plot_vs_time_with_rotation(
        t, speed, valid, segs, theta_deg, conv,
        f"velikost rychlosti |v| [{v_unit}]", "Velikost rychlosti středu v čase",
        paths["v_t"])

    plot_vs_angle(
        theta_deg, speed, t, valid, segs, conv,
        f"velikost rychlosti |v| [{v_unit}]", "Velikost rychlosti v závislosti na natočení",
        paths["v_phi"])

    plot_torque_speed_period(
        theta_deg, speed, torque, valid, segs, conv, t_label, t_unit, v_unit, paths["M_v"])

    plot_vs_angle(
        theta_deg, torque, t, valid, segs, conv,
        f"moment síly {t_label} [{t_unit}]", "Moment síly v závislosti na natočení",
        paths["M_phi"], zero_line=True)

    plot_vs_time_with_rotation(
        t, omega, valid, segs, theta_deg, conv,
        "úhlová rychlost ω [rad/s]", "Úhlová rychlost v čase",
        paths["w_t"], zero_line=True)

    plot_vs_time_with_rotation(
        t, alpha, valid, segs, theta_deg, conv,
        "úhlové zrychlení α [rad/s²]", "Úhlové zrychlení v čase",
        paths["a_t"], zero_line=True)

    plot_key_figure(conv, paths["key"])

    have_snap = plot_snapshots(t, theta_deg, speed, omega, alpha, torque, valid, conv, v_unit,
                               t_label, t_unit, args.snapshots, paths["snap"])

    have_arms = plot_torque_arms_series(t, theta_deg, omega, alpha, torque, valid, conv, t_label,
                                        t_unit, args.snapshots, paths["arms"], force_arm)

    # přesnost momentu a mechanismus otáčení
    acc = compute_accuracy(theta_deg, valid, segs, dt, args.smooth_win_alpha, args.smooth_poly,
                           speed, torque)
    fac = args.inertia or 1.0
    have_acc = plot_accuracy(t, torque, valid, segs, acc, fac, t_label, t_unit, paths["acc"])
    have_mech = plot_mechanism(theta_deg, omega, alpha, torque, valid, conv, t_label, t_unit,
                               args.inertia, paths["mech"])
    if np.isfinite(acc.sigma_theta):
        print(f"Přesnost: šum úhlu σθ ≈ {np.degrees(acc.sigma_theta):.2f}°, "
              f"medián σα ≈ {np.nanmedian(acc.sigma_alpha[valid]):.2g} rad/s²")
    print("Opakovatelnost mezi otáčkami (rozptyl / amplituda): " +
          ", ".join(f"{k}: " + (f"{100 * v:.0f} %" if np.isfinite(v) else "n/a")
                    for k, (v, _) in acc.spread.items()))

    for k in ("xy", "phi_t", "v_t", "v_phi", "M_v", "M_phi", "w_t", "a_t", "key"):
        print("Uloženo:", paths[k])
    if have_snap:
        print("Uloženo:", paths["snap"])
    if have_arms:
        print("Uloženo:", paths["arms"])
    for ok_, k_ in ((have_mech, "mech"), (have_acc, "acc")):
        if ok_:
            print("Uloženo:", paths[k_])

    if args.debug_video or args.torque_video:
        kin = SimpleNamespace(vx=vx, vy=vy, ax=acx, ay=acy, speed=speed, a_mag=a_mag,
                              theta=theta_deg, turns=turns, omega=omega, alpha=alpha,
                              torque=torque, t_label=t_label, t_unit=t_unit,
                              v_dir=v_dir, a_dir=a_dir, t=t, unit=unit, com=com * scale,
                              T1=T1, T2=T2, sgn=sgn, L_px=L, force=force_arm)
        half = args.smooth_win_alpha // 2
        if args.debug_video:
            write_debug_video(args.video, paths["debug"], A, B, valid, kin, segs, half)
            print("Uloženo:", paths["debug"])
        if args.torque_video:
            write_torque_video(args.video, paths["arms_video"], A, B, valid, kin, segs, half)
            print("Uloženo:", paths["arms_video"])

    if not args.inertia:
        print("Pozn.: moment setrvacnosti nebyl zadan (--inertia), kresli se M/I = alfa [1/s^2].")


if __name__ == "__main__":
    main()
