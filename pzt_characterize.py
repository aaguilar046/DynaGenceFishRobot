"""
PZT cylinder impedance sweep and Butterworth-Van Dyke (BVD) fit.

Circuit (sense resistor on the ground side):
    33510B OUT ---+--- PZT cylinder ---+--- RS_OHM --- GND
                  |                    |
               scope CH2            scope CH1
               (drive)              (current: I = V1 / RS_OHM)

The 33510B steps a sine through the frequencies. At each step the scope captures
both channels at once, and a sine fit gives the complex ratio H = V1 / V2. Then
    Z_pzt = RS_OHM * (1 - H) / H

BVD model: C0 (with dielectric loss tan_delta) in parallel with a series R1-L1-C1 branch
    Y = j w C0 (1 - j tan_delta) + 1 / (R1 + j w L1 + 1 / (j w C1))
Only the band FIT_BAND (relative to fs and fp) is fitted, so neighbouring modes
don't pull the result.

Usage:
    python pzt_characterize.py              measure cylinders (prompts for names)
    python pzt_characterize.py --cal-thru   CH1 and CH2 both on the drive node
    python pzt_characterize.py --cal-open   cylinder unplugged, leads still in place
    python pzt_characterize.py --step 100   100 Hz steps across the whole range (no fine sweep)
Calibrations are optional; when their files exist they are applied automatically.
"""

import os
import re
import sys
import time
import numpy as np
import pyvisa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ---------------- user settings ----------------
GEN_ADDR    = "USB0::0x0957::0x2607::MY62004165::INSTR"      # Keysight 33510B
SCOPE_ADDR  = "USB0::0xF4ED::0xEE3A::SDS1EEFC901297::INSTR"  # Siglent SDS1202X-E

RS_OHM      = 11.0              # sense resistor (cylinder to GND); enter its measured value
DRIVE_VPP   = 0.5               # generator amplitude (high-Z setting)
PROBE_ATTN  = 1                 # scope probe attenuation (1 or 10), same on both channels
MEDIUM      = "water"           # tag saved with each result

F_START     = 1e3               # coarse sweep, Hz
F_STOP      = 100e3
F_STEP      = 1e3
ZOOM_SPAN   = 4e3               # fine sweep +/- this around resonance and anti-resonance
ZOOM_STEP   = 100.0
EXTRA_ZOOM  = (57e3, 90e3)      # also fine-sweep +/- ZOOM_SPAN around these, Hz
FIT_NEAR    = None              # e.g. 57e3 to fit the mode nearest this frequency
FIT_BAND    = (0.75, 1.2)       # fit from 0.75*fs to 1.2*fp

N_CYCLES    = 20                # cycles captured per point (at least)
SETTLE_S    = 0.05              # wait after a frequency change
MSIZ        = "7K"              # scope memory depth with two channels on
MAX_RETRIES = 3                 # reconnect attempts per point if the scope's USB hangs

OUT_DIR     = "pzt_results"
SHOW_PLOT   = True              # open each cylinder's plot when it finishes
# -----------------------------------------------

VDIVS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
TDIVS = [m * 10.0 ** e for e in range(-9, 2) for m in (1, 2, 5)]
CAL_THRU = os.path.join(OUT_DIR, "cal_thru.csv")
CAL_OPEN = os.path.join(OUT_DIR, "cal_open.csv")


def num(s):
    return float(re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s).group())


# ---------------- instruments ----------------
def setup_gen(g):
    g.write("*RST")
    g.write("OUTP:LOAD INF")
    g.write("SOUR1:FUNC SIN")
    g.write(f"SOUR1:VOLT {DRIVE_VPP}")
    g.write("SOUR1:VOLT:OFFS 0")
    g.write(f"SOUR1:FREQ {F_START}")
    g.write("OUTP ON")


def setup_scope(s):
    s.write("CHDR OFF")
    for ch in ("C1", "C2"):
        s.write(f"{ch}:TRA ON")
        s.write(f"{ch}:ATTN {PROBE_ATTN}")
        s.write(f"{ch}:CPL D1M")
        s.write(f"BWL {ch},OFF")
        s.write(f"{ch}:OFST 0V")
    s.write(f"MSIZ {MSIZ}")
    s.write("ACQW SAMPLING")
    s.write("TRDL 0S")
    s.write("TRSE EDGE,SR,C2,HT,OFF")
    s.write("C2:TRCP DC")
    s.write("C2:TRSL POS")
    s.write("C2:TRLV 0V")
    s.write("WFSU SP,0,NP,0,FP,0")


# The SDS1202X-E's USB link hangs after enough traffic, so the scope settings are
# tracked here instead of being queried back, and status polling is kept sparse.
VDIV_NOW = {}
TDIV_NOW = {}


def set_vdiv(s, ch, vdiv):
    if VDIV_NOW.get(ch) != vdiv:
        s.write(f"{ch}:VDIV {vdiv:.3E}V")
        VDIV_NOW[ch] = vdiv


def capture(s, tdiv):
    s.write("TRMD SINGLE")
    time.sleep(14 * tdiv + 0.05)
    for _ in range(10):
        if "stop" in s.query("SAST?").lower():
            return
        time.sleep(0.2)
    s.write("FRTR")                                 # sine is continuous; any trigger works
    time.sleep(14 * tdiv + 0.1)


def read_ch(s, ch):
    vdiv = VDIV_NOW[ch]                             # offset is fixed at 0 V
    s.write(f"{ch}:WF? DAT2")
    raw = s.read_raw()
    i = raw.find(b"#9")
    n = int(raw[i + 2:i + 11])
    code = np.frombuffer(raw[i + 11:i + 11 + n], dtype=np.int8).astype(float)
    return code, code * (vdiv / 25.0)


def sine_fit(t, v, f):
    """Complex amplitude of the f component (least-squares cos/sin/DC fit)."""
    w = 2 * np.pi * f
    a = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(a, v, rcond=None)
    return c[0] - 1j * c[1]


def measure_point(g, s, f, vdiv1):
    """Return H = V1/V2 at f and the CH1 V/div that fit the signal."""
    g.write(f"SOUR1:FREQ {f}")
    tdiv = next(td for td in TDIVS if 14 * td >= N_CYCLES / f)
    if TDIV_NOW.get("t") != tdiv:
        s.write(f"TDIV {tdiv:.2E}S")
        TDIV_NOW["t"] = tdiv
    time.sleep(SETTLE_S)
    for _ in range(6):                              # autoscale CH1
        set_vdiv(s, "C1", vdiv1)
        capture(s, tdiv)
        code1, v1 = read_ch(s, "C1")
        peak = np.max(np.abs(code1))
        i = VDIVS.index(vdiv1)
        if peak > 118 and i < len(VDIVS) - 1:
            vdiv1 = VDIVS[i + 1]
        elif peak < 30 and i > 0:
            vdiv1 = VDIVS[i - 1]
        else:
            break
    _, v2 = read_ch(s, "C2")
    t = np.arange(len(v1)) * (14 * tdiv / len(v1))
    return sine_fit(t, v1, f) / sine_fit(t, v2, f), vdiv1


def open_scope(rm):
    VDIV_NOW.clear()
    TDIV_NOW.clear()
    s = rm.open_resource(SCOPE_ADDR)
    s.timeout = 10000
    s.chunk_size = 1 << 20
    setup_scope(s)
    return s


def reconnect(rm, s):
    """The SDS1202X-E's USB link occasionally hangs; drop it and reopen."""
    for _ in range(5):
        try:
            s.close()
        except pyvisa.Error:
            pass
        time.sleep(3)
        try:
            s = open_scope(rm)
            s.query("*IDN?")
            return s
        except pyvisa.Error:
            continue
    raise RuntimeError("Scope stopped responding over USB. Power-cycle it (or unplug and "
                       "replug its USB cable), then run again.")


def sweep(g, s, rm, freqs):
    """Measure H at each frequency. Returns (H array, scope handle)."""
    vdiv2 = next(v for v in VDIVS if 8 * v >= DRIVE_VPP * 1.3)
    vdiv1 = vdiv2
    set_vdiv(s, "C2", vdiv2)
    h = []
    for k, f in enumerate(freqs):
        for attempt in range(MAX_RETRIES + 1):
            try:
                hk, vdiv1 = measure_point(g, s, f, vdiv1)
                break
            except pyvisa.Error as e:
                if attempt == MAX_RETRIES:
                    raise
                print(f"  scope error at {f / 1e3:.2f} kHz ({type(e).__name__}), reconnecting...")
                s = reconnect(rm, s)
                set_vdiv(s, "C2", vdiv2)
        h.append(hk)
        if k % 10 == 0 or k == len(freqs) - 1:
            print(f"  {k + 1:4d}/{len(freqs)}  {f / 1e3:7.2f} kHz  |H| {abs(hk):.3f}")
    return np.array(h), s


# ---------------- impedance + calibration ----------------
def load_cal(path):
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1] + 1j * d[:, 2]


def interp_c(f, cal):
    fc, vc = cal
    return np.interp(f, fc, vc.real) + 1j * np.interp(f, fc, vc.imag)


def impedance(f, h, thru, open_):
    if thru is not None:
        h = h / interp_c(f, thru)                   # channel gain/phase mismatch
    z = RS_OHM * (1 - h) / h
    if open_ is not None:
        y = 1 / z - interp_c(f, open_)              # remove stray C of leads + scope
        z = 1 / y
    return z


# ---------------- BVD fit ----------------
def bvd_y(f, r1, l1, c1, c0, tand=0.0):
    w = 2 * np.pi * f
    return 1j * w * c0 * (1 - 1j * tand) + 1 / (r1 + 1j * w * l1 + 1 / (1j * w * c1))


def fit_bvd(f, z):
    y = 1 / z
    g = y.real
    band = np.ones_like(f, bool) if FIT_NEAR is None else np.abs(f - FIT_NEAR) < 0.2 * FIT_NEAR
    i_s = np.flatnonzero(band)[np.argmax(g[band])]
    fs0 = f[i_s]
    above = (f > fs0) & (f < 1.5 * fs0)
    fp0 = f[above][np.argmax(np.abs(z[above]))] if above.any() else 1.05 * fs0
    low = (f > 0.2 * fs0) & (f < 0.6 * fs0)
    ctot = np.median(y[low].imag / (2 * np.pi * f[low])) if low.any() else \
        y[i_s].imag / (2 * np.pi * fs0)
    tand = np.median(y[low].real / y[low].imag) if low.any() else 0.01
    c1 = max(ctot * (1 - (fs0 / fp0) ** 2), ctot * 1e-3)
    p0 = np.log([1 / g[i_s], 1 / ((2 * np.pi * fs0) ** 2 * c1), c1, max(ctot - c1, ctot * 0.1),
                 min(max(tand, 1e-4), 0.5)])

    sel = (f > FIT_BAND[0] * fs0) & (f < FIT_BAND[1] * fp0)

    def resid(p):
        e = (bvd_y(f[sel], *np.exp(p)) - y[sel]) / np.abs(y[sel])
        return np.concatenate([e.real, e.imag])

    r1, l1, c1, c0, tand = np.exp(least_squares(resid, p0).x)
    fs = 1 / (2 * np.pi * np.sqrt(l1 * c1))
    fp = fs * np.sqrt(1 + c1 / c0)
    zf = 1 / bvd_y(f[sel], r1, l1, c1, c0, tand)
    # read straight from the data well below resonance (no model involved)
    lf = (f > 0.3 * fs) & (f < 0.5 * fs)
    c_lf = np.median(y[lf].imag / (2 * np.pi * f[lf])) if lf.any() else np.nan
    tand_lf = np.median(y[lf].real / y[lf].imag) if lf.any() else np.nan
    return dict(R1_ohm=r1, L1_H=l1, C1_F=c1, C0_F=c0, tan_delta=tand, fs_Hz=fs, fp_Hz=fp,
                C_lowf_F=c_lf, tand_lowf=tand_lf,
                Qm=np.sqrt(l1 / c1) / r1, keff=np.sqrt(1 - (fs / fp) ** 2),
                fit_lo_Hz=f[sel][0], fit_hi_Hz=f[sel][-1],
                fit_err_pct=100 * np.median(np.abs(zf / z[sel] - 1)))


# ---------------- output ----------------
def eng(x, unit):
    for m, p in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"),
                 (1e-6, "u"), (1e-9, "n"), (1e-12, "p")):
        if abs(x) >= m:
            return f"{x / m:.4g} {p}{unit}"
    return f"{x:.4g} {unit}"


def save_results(name, f, z, fit):
    zf = 1 / bvd_y(f, fit["R1_ohm"], fit["L1_H"], fit["C1_F"], fit["C0_F"], fit["tan_delta"])
    base = os.path.join(OUT_DIR, f"{name}_{MEDIUM}")
    np.savetxt(base + ".csv",
               np.column_stack([f, np.abs(z), np.degrees(np.angle(z)), z.real, z.imag,
                                np.abs(zf), np.degrees(np.angle(zf))]),
               delimiter=",", fmt="%.6g",
               header="f_Hz,Zmag_ohm,Zphase_deg,R_ohm,X_ohm,fit_Zmag_ohm,fit_Zphase_deg",
               comments="")

    summary = os.path.join(OUT_DIR, "pzt_summary.csv")
    keys = list(fit)
    rows = []
    if os.path.exists(summary):                     # a re-measured cylinder replaces its row
        with open(summary, encoding="utf-8") as fh:
            rows = [r for r in fh.read().splitlines()[1:]
                    if r and not r.startswith(f"{name},{MEDIUM},")]
    rows.append(f"{name},{MEDIUM},{RS_OHM}," + ",".join(f"{fit[k]:.6g}" for k in keys))
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write("cylinder,medium,RS_ohm," + ",".join(keys) + "\n")
        fh.write("\n".join(rows) + "\n")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    a1.semilogy(f / 1e3, np.abs(z), ".", ms=3, label="measured")
    a1.semilogy(f / 1e3, np.abs(zf), "-", lw=1.2, label="BVD fit")
    a1.set_ylabel("|Z| (ohm)")
    a1.legend()
    a2.plot(f / 1e3, np.degrees(np.angle(z)), ".", ms=3)
    a2.plot(f / 1e3, np.degrees(np.angle(zf)), "-", lw=1.2)
    a2.set_ylabel("phase (deg)")
    a2.set_xlabel("frequency (kHz)")
    for a in (a1, a2):
        a.grid(True, which="both", alpha=0.3)
        for fx in (fit["fs_Hz"], fit["fp_Hz"]):
            a.axvline(fx / 1e3, color="gray", ls=":", lw=1)
        a.axvspan(fit["fit_lo_Hz"] / 1e3, fit["fit_hi_Hz"] / 1e3, color="orange", alpha=0.08)
    a1.set_title(f"{name} ({MEDIUM})   R1 {eng(fit['R1_ohm'], 'ohm')}   "
                 f"L1 {eng(fit['L1_H'], 'H')}   C1 {eng(fit['C1_F'], 'F')}   "
                 f"C0 {eng(fit['C0_F'], 'F')}   tan d {fit['tan_delta']:.3f}   "
                 f"(fit band shaded, median error {fit['fit_err_pct']:.1f}%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)
    return base


def coarse_grid():
    return np.round(np.arange(F_START, F_STOP + F_STEP / 2, F_STEP), 6)


def zoom_grid(f, z):
    y = 1 / z
    i_s = np.argmax(y.real)
    above = f > f[i_s]
    centers = [f[i_s]] + ([f[above][np.argmax(np.abs(z[above]))]] if above.any() else [])
    centers += list(EXTRA_ZOOM)
    fine = [np.arange(c - ZOOM_SPAN, c + ZOOM_SPAN + ZOOM_STEP / 2, ZOOM_STEP) for c in centers]
    fine = np.unique(np.round(np.concatenate(fine), 6))
    return fine[(fine >= F_START) & (fine <= F_STOP)]


# ---------------- main ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rm = pyvisa.ResourceManager()
    g = rm.open_resource(GEN_ADDR)
    g.timeout = 10000
    setup_gen(g)
    s = open_scope(rm)

    try:
        global F_STEP
        args = sys.argv[1:]
        if "--step" in args:
            i = args.index("--step")
            F_STEP = float(args[i + 1])
            del args[i:i + 2]
        mode = args[0] if args else ""
        if mode in ("--cal-thru", "--cal-open"):
            f = coarse_grid()
            print(f"Calibration {mode[6:]}: {len(f)} points")
            h, s = sweep(g, s, rm, f)
            if mode == "--cal-thru":
                path, val = CAL_THRU, h
            else:
                thru = load_cal(CAL_THRU)
                path, val = CAL_OPEN, 1 / impedance(f, h, thru, None)
            np.savetxt(path, np.column_stack([f, val.real, val.imag]), delimiter=",",
                       fmt="%.8g", header="f_Hz,re,im", comments="")
            print(f"Saved {path}")
            return

        thru, open_ = load_cal(CAL_THRU), load_cal(CAL_OPEN)
        print(f"Calibration: thru {'on' if thru else 'off'}, open {'on' if open_ else 'off'}")
        while True:
            # strip the byte-order mark PowerShell adds when piping text in
            name = input("\nCylinder name (blank to quit): ").strip().lstrip("﻿").strip()
            if not name:
                break
            f = coarse_grid()
            print(f"{'Full' if F_STEP <= ZOOM_STEP else 'Coarse'} sweep, {len(f)} points, "
                  f"{F_STEP:g} Hz steps")
            h, s = sweep(g, s, rm, f)
            if F_STEP > ZOOM_STEP:                  # already fine enough otherwise
                fz = zoom_grid(f, impedance(f, h, thru, open_))
                fz = fz[~np.isin(fz, f)]
                print(f"Fine sweep around resonance, {len(fz)} points")
                hz, s = sweep(g, s, rm, fz)
                h = np.concatenate([h, hz])
                f = np.concatenate([f, fz])
            order = np.argsort(f)
            f, h = f[order], h[order]
            z = impedance(f, h, thru, open_)
            fit = fit_bvd(f, z)
            base = save_results(name, f, z, fit)
            print(f"\n{name} ({MEDIUM}):")
            print(f"  R1 = {eng(fit['R1_ohm'], 'ohm')}   L1 = {eng(fit['L1_H'], 'H')}   "
                  f"C1 = {eng(fit['C1_F'], 'F')}   C0 = {eng(fit['C0_F'], 'F')}")
            print(f"  fs = {fit['fs_Hz'] / 1e3:.2f} kHz   fp = {fit['fp_Hz'] / 1e3:.2f} kHz   "
                  f"Qm = {fit['Qm']:.1f}   keff = {fit['keff']:.3f}   "
                  f"tan d = {fit['tan_delta']:.3f}")
            print(f"  measured at 0.3-0.5 fs: C = {eng(fit['C_lowf_F'], 'F')}, "
                  f"tan d = {fit['tand_lowf']:.3f}")
            print(f"  fit {fit['fit_lo_Hz'] / 1e3:.1f}-{fit['fit_hi_Hz'] / 1e3:.1f} kHz, "
                  f"median error {fit['fit_err_pct']:.2f}%")
            print(f"  Saved {base}.csv and {base}.png")
            if SHOW_PLOT:
                os.startfile(os.path.abspath(base + ".png"))
    finally:
        g.write("OUTP OFF")


if __name__ == "__main__":
    main()
