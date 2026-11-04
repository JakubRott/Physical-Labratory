#!/usr/bin/env python3
"""
track_dots.py – sledování dvou svítících teček ve videu.

Pro každé video vznikne složka <jméno_videa>/ a v ní:
  <jméno>_tracks.csv                    – polohy, natočení θ, rychlost, ω, α, moment
  <jméno>_01_draha_xy.png               – dráha středu + činky (barva = natočení θ)
  <jméno>_02_natoceni_cas.png           – natočení θ v čase (stupně + otáčky)
  <jméno>_03_rychlost_cas.png           – velikost rychlosti v čase
  <jméno>_04_rychlost_uhel.png          – velikost rychlosti v závislosti na θ
  <jméno>_05_moment_rychlost.png        – moment síly v závislosti na velikosti rychlosti
  <jméno>_06_moment_uhel.png            – moment síly v závislosti na θ
  <jméno>_07_uhlova_rychlost_cas.png    – úhlová rychlost v čase
  <jméno>_08_uhlove_zrychleni_cas.png   – úhlové zrychlení v čase
  <jméno>_09_klic_natoceni.png          – KLÍČ: jak vypadá činka při θ = 0°, 30°, … 330°
  <jméno>_10_cinka_v_case.png           – činka v několika okamžicích + hodnoty veličin
  volitelně <jméno>_debug.mp4           – zesvětlené video s vyznačeným trackem

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
from matplotlib.patches import Wedge, Circle, Rectangle
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
    "debug": "debug.mp4",
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


def plot_xy(com, theta_deg, conv, valid, A, B, L, scale, unit, out_png, gap_max, ticks_every,
            vec=None):
    """Dráha středu + činky. vec = (vx, vy, ax, ay) ve škálovaných jednotkách (unit/s, unit/s²);
    když je zadáno, u každé nakreslené činky přibude šipka rychlosti (černá) a zrychlení (zelená)."""
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

    # --- šipky rychlosti a zrychlení středu --------------------------------------
    pts_x = [com[idx, 0] * scale, A[idx, 0] * scale, B[idx, 0] * scale]
    pts_y = [com[idx, 1] * scale, -A[idx, 1] * scale, -B[idx, 1] * scale]
    extent = max(np.ptp(np.concatenate(pts_x)), np.ptp(np.concatenate(pts_y)), 1e-9)
    arrows = []                                    # (x, y, u, v, barva, popis, jednotka, ref, k)
    if vec is not None:
        sel = idx[::max(1, ticks_every)]
        for (cx_, cy_), col, name, u_unit in (
                ((vec[0], vec[1]), "k", "rychlost středu", f"{unit}/s"),
                ((vec[2], vec[3]), "#1a9850", "zrychlení středu", f"{unit}/s²")):
            ok = sel[np.isfinite(cx_[sel]) & np.isfinite(cy_[sel])]
            if len(ok) == 0:
                continue
            mag = np.hypot(cx_[ok], cy_[ok])
            p95 = np.percentile(mag, 95)
            if p95 <= 0:
                continue
            k = 0.10 * extent / p95                # šipka o velikosti p95 má 10 % rozsahu grafu
            x0_, y0_ = com[ok, 0] * scale, com[ok, 1] * scale
            u_, v_ = cx_[ok] * k, cy_[ok] * k
            ax.quiver(x0_, y0_, u_, v_, angles="xy", scale_units="xy", scale=1, color=col,
                      width=0.0035, headwidth=4.5, headlength=5.5, zorder=6)
            pts_x.append(x0_ + u_)
            pts_y.append(y0_ + v_)
            arrows.append((col, name, u_unit, nice_number(p95), k))

    xs, ys = np.concatenate(pts_x), np.concatenate(pts_y)
    padx, pady = 0.06 * np.ptp(xs) + 1e-9, 0.06 * np.ptp(ys) + 1e-9
    ax.set_xlim(xs.min() - padx, xs.max() + padx)
    ax.set_ylim(ys.min() - pady, ys.max() + pady)
    ax.set_aspect("equal", adjustable="datalim")
    ax.apply_aspect()

    # --- legenda měřítka šipek (v rohu, kde je nejméně dat) ----------------------
    if arrows:
        X0, X1 = ax.get_xlim()
        Y0, Y1 = ax.get_ylim()
        W, H = X1 - X0, Y1 - Y0
        bw, bh = 0.44 * W, (0.05 + 0.065 * len(arrows)) * H
        corners = [(X0 + 0.01 * W, Y0 + 0.01 * H), (X1 - bw - 0.01 * W, Y0 + 0.01 * H),
                   (X0 + 0.01 * W, Y1 - bh - 0.01 * H), (X1 - bw - 0.01 * W, Y1 - bh - 0.01 * H)]
        def n_inside(c):
            return int(np.sum((xs >= c[0]) & (xs <= c[0] + bw) & (ys >= c[1]) & (ys <= c[1] + bh)))
        bx, by = min(corners, key=n_inside)
        ax.add_patch(Rectangle((bx, by), bw, bh, fc="white", ec="0.65", alpha=0.93, zorder=7))
        for row, (col, name, u_unit, ref, k) in enumerate(arrows):
            yy = by + bh - (0.045 + 0.065 * row + 0.02) * H - 0.012 * H
            ax.quiver([bx + 0.02 * W], [yy], [ref * k], [0], angles="xy", scale_units="xy",
                      scale=1, color=col, width=0.0035, headwidth=4.5, headlength=5.5, zorder=8)
            ax.text(bx + 0.03 * W + ref * k, yy, f"= {ref:g} {u_unit}   ({name})", fontsize=8.5,
                    va="center", ha="left", zorder=8)

    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.set_title(f"Dráha středu mezi tečkami (vzdálenost teček ≈ {L * scale:.1f} {unit})")
    ax.annotate("start", com[idx[0]] * scale, xytext=(8, 8), textcoords="offset points",
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax.annotate("konec", com[idx[-1]] * scale, xytext=(8, -14), textcoords="offset points",
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    draw_key(kax, conv)
    extra = "Činky v grafu mají stejné natočení i barvu jako odpovídající činka v klíči vpravo."
    if arrows:
        extra += ("  Šipky vycházejí ze středu činky: černá = rychlost, zelená = zrychlení "
                  "(směr jako v grafu: x doprava, y nahoru; délka úměrná velikosti, měřítko v rámečku).")
    footnote(fig, conv, extra)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
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


# ----------------------------------------------------------------------------
# 8) Kontrolní video
# ----------------------------------------------------------------------------
def write_debug_video(path, out_path, A, B, valid, gain=6.0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        if valid[i]:
            a = tuple(np.round(A[i]).astype(int))
            b = tuple(np.round(B[i]).astype(int))
            c = (int(round((a[0] + b[0]) / 2)), int(round((a[1] + b[1]) / 2)))
            trail.append(c)
            cv2.circle(f, a, 8, (0, 0, 255), 1)
            cv2.circle(f, b, 8, (0, 255, 0), 1)
            cv2.line(f, a, b, (255, 255, 0), 1)
            cv2.circle(f, c, 3, (0, 255, 255), -1)
        for p, q in zip(trail[:-1], trail[1:]):
            cv2.line(f, p, q, (0, 200, 255), 1)
        cv2.putText(f, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
    theta = sgn * (raw0 + phi)

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
                 f"ax_{unit}_per_s2,ay_{unit}_per_s2,a_mag_{unit}_per_s2,a_dir_deg\n")
        for i in idx:
            f = lambda v, p: "" if not np.isfinite(v) else f"{v:.{p}f}"
            fh.write(f"{i},{t[i]:.4f},{A[i,0]:.2f},{A[i,1]:.2f},{B[i,0]:.2f},{B[i,1]:.2f},"
                     f"{com[i,0] * scale:.3f},{com[i,1] * scale:.3f},"
                     f"{theta_deg[i]:.2f},{theta_deg[i] % 360:.2f},{turns[i]:.4f},"
                     f"{f(speed[i], 4)},{f(omega[i], 4)},{f(alpha[i], 3)},{f(torque[i], 6)},"
                     f"{f(vx[i], 4)},{f(vy[i], 4)},{f(v_dir[i], 2)},"
                     f"{f(acx[i], 3)},{f(acy[i], 3)},{f(a_mag[i], 3)},{f(a_dir[i], 2)}\n")
    print("Uloženo:", paths["csv"])

    # Grafy ------------------------------------------------------------------
    plot_xy(com, theta_deg, conv, valid, A, B, L, scale, unit, paths["xy"],
            args.gap_max, args.ticks_every, vec=(vx, vy, acx, acy))

    plot_rotation_time(t, theta_deg, valid, segs, conv, paths["phi_t"])

    plot_vs_time_with_rotation(
        t, speed, valid, segs, theta_deg, conv,
        f"velikost rychlosti |v| [{v_unit}]", "Velikost rychlosti středu v čase",
        paths["v_t"])

    plot_vs_angle(
        theta_deg, speed, t, valid, segs, conv,
        f"velikost rychlosti |v| [{v_unit}]", "Velikost rychlosti v závislosti na natočení",
        paths["v_phi"])

    plot_xy_colored_by_time(
        speed, torque, t, valid, segs,
        f"velikost rychlosti |v| [{v_unit}]", f"moment síly {t_label} [{t_unit}]",
        "Moment síly v závislosti na velikosti rychlosti", paths["M_v"], zero_lines=True)

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

    for k in ("xy", "phi_t", "v_t", "v_phi", "M_v", "M_phi", "w_t", "a_t", "key"):
        print("Uloženo:", paths[k])
    if have_snap:
        print("Uloženo:", paths["snap"])

    if args.debug_video:
        write_debug_video(args.video, paths["debug"], A, B, valid)
        print("Uloženo:", paths["debug"])

    if not args.inertia:
        print("Pozn.: moment setrvacnosti nebyl zadan (--inertia), kresli se M/I = alfa [1/s^2].")


if __name__ == "__main__":
    main()
