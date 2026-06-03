from pathlib import Path
import csv
import numpy as np
from scipy.optimize import least_squares

# Batch fit EC-Lab ASCII exports (*.txt or *.mpt).
# Extracts FIRST and LAST EIS spectrum from each file by frequency reset.
# Fits: R1 + (R2 || C2) + (R3 || C3)

INPUT_FOLDER = Path(".")
OUTPUT_FOLDER = Path("python_EIS_fit_results")
CURVE_FOLDER = OUTPUT_FOLDER / "fitted_curves"
OUTPUT_FOLDER.mkdir(exist_ok=True)
CURVE_FOLDER.mkdir(exist_ok=True)

def parse_eclab_ascii(path):
    lines = path.read_text(errors="replace").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("time/s") and "freq/Hz" in line and "Re(Z)" in line:
            header_idx = i
            break
    if header_idx is None:
        print(f"SKIP: {path.name} - no EIS table header found")
        return []
    cols = [c for c in lines[header_idx].split("\t") if c.strip()]
    data = []
    for line in lines[header_idx+1:]:
        if not line.strip():
            continue
        parts = [p for p in line.split("\t") if p.strip()]
        if len(parts) < len(cols):
            continue
        try:
            r = {c: float(parts[j]) for j, c in enumerate(cols)}
        except ValueError:
            continue
        freq = r.get("freq/Hz", 0)
        rez = r.get("Re(Z)/Ohm", 0)
        minus_imz = r.get("-Im(Z)/Ohm", 0)
        if freq > 0 and (rez != 0 or minus_imz != 0):
            data.append({
                "time/s": r.get("time/s", float("nan")),
                "cycle number": r.get("cycle number", float("nan")),
                "freq/Hz": freq,
                "Re(Z)/Ohm": rez,
                "-Im(Z)/Ohm": minus_imz,
            })
    return data

def split_by_freq_reset(data):
    spectra, cur, prev_f = [], [], None
    for r in data:
        f = r["freq/Hz"]
        if prev_f is not None and f > prev_f * 10 and cur:
            spectra.append(cur)
            cur = []
        cur.append(r)
        prev_f = f
    if cur:
        spectra.append(cur)
    return spectra

def z_model(freq, R1, R2, C2, R3, C3):
    w = 2 * np.pi * freq
    z2 = 1 / (1/R2 + 1j*w*C2)
    z3 = 1 / (1/R3 + 1j*w*C3)
    return R1 + z2 + z3

def initial_guess(z):
    re = np.real(z)
    rmin = max(float(np.nanmin(re)), 1e-6)
    rmax = max(float(np.nanmax(re)), rmin + 1e-3)
    span = max(rmax - rmin, 1.0)
    return np.array([rmin, span * 0.4, 5e-8, span * 0.6, 1e-6], dtype=float)

def fit_spectrum(spec):
    freq = np.array([r["freq/Hz"] for r in spec], dtype=float)
    z = np.array([r["Re(Z)/Ohm"] - 1j*r["-Im(Z)/Ohm"] for r in spec], dtype=complex)
    order = np.argsort(freq)[::-1]
    freq = freq[order]
    z = z[order]

    def pack(p):
        return np.log(np.maximum(p, 1e-30))

    def unpack(x):
        return np.exp(x)

    x0 = pack(initial_guess(z))
    scale = np.maximum(np.abs(z), 1.0)

    def residual(x):
        p = unpack(x)
        zm = z_model(freq, *p)
        return np.r_[(np.real(zm) - np.real(z)) / scale,
                     (np.imag(zm) - np.imag(z)) / scale]

    lower = pack(np.array([1e-6, 1e-6, 1e-12, 1e-6, 1e-12]))
    upper = pack(np.array([1e6, 1e7, 1e-1, 1e7, 1e-1]))

    res = least_squares(
        residual, x0, bounds=(lower, upper),
        max_nfev=20000, xtol=1e-12, ftol=1e-12, gtol=1e-12
    )
    p = unpack(res.x)
    zfit = z_model(freq, *p)
    chi2_rel = float(np.mean(np.abs(zfit - z)**2 / np.maximum(np.abs(z)**2, 1.0)))
    return freq, z, zfit, p, chi2_rel, res.cost, res.success

def safe_name(s):
    return s.replace("°", "deg").replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")

summary = []
input_files = sorted(list(INPUT_FOLDER.glob("*.txt")) + list(INPUT_FOLDER.glob("*.mpt")))

for path in input_files:
    data = parse_eclab_ascii(path)
    spectra = split_by_freq_reset(data)
    if len(spectra) == 0:
        continue

    for tag, idx in [("FIRST", 0), ("LAST", len(spectra)-1)]:
        spec = spectra[idx]
        try:
            freq, z, zfit, p, chi2_rel, cost, success = fit_spectrum(spec)
            R1, R2, C2, R3, C3 = p
            curve_name = f"{safe_name(path.stem)}_{tag}_spectrum{idx:02d}_fit_curve.csv"
            curve_path = CURVE_FOLDER / curve_name

            with curve_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["freq/Hz", "ReZ_data/Ohm", "-ImZ_data/Ohm", "ReZ_fit/Ohm", "-ImZ_fit/Ohm", "spectrum_index"])
                for ff, zz, zf in zip(freq, z, zfit):
                    w.writerow([ff, np.real(zz), -np.imag(zz), np.real(zf), -np.imag(zf), idx])

            summary.append([path.name, tag, idx, len(spec), len(spectra),
                            R1, R2, C2, R3, C3, chi2_rel, cost, success, curve_name])
            print(f"{path.name} {tag} spectrum {idx}: OK, chi2_rel={chi2_rel:.4g}")

        except Exception as e:
            summary.append([path.name, tag, idx, len(spec), len(spectra),
                            "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", False, str(e)])
            print(f"{path.name} {tag} spectrum {idx}: ERROR {e}")

with (OUTPUT_FOLDER / "fit_summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "which", "spectrum_index", "n_points", "n_spectra_found",
                "R1/Ohm", "R2/Ohm", "C2/F", "R3/Ohm", "C3/F",
                "chi2_rel", "least_squares_cost", "success", "curve_file"])
    w.writerows(summary)

print(f"\nDone. Results folder: {OUTPUT_FOLDER.resolve()}")
