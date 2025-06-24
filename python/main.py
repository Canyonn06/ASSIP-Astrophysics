# -*- coding: utf-8 -*-
"""
Author: ASSIP Cohort '25

Github: https://github.com/Canyonn06/ASSIP-Astrophysics 
    * Visit the github page for instructions, etc.
"""

import warnings
from scipy.optimize import curve_fit, OptimizeWarning
import matplotlib.pyplot as plt
import os
import re
import pickle
import numpy as np
import pandas as pd
import h5py
import eispac as eis
import matplotlib
import sys
matplotlib.use("Qt5Agg", force=True)
import contextlib, io
# hola

_orig_read_cube = eis.read_cube

def read_cube_silent(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return _orig_read_cube(*args, **kwargs)

eis.read_cube = read_cube_silent

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from helpers.config import (
        DATA_DIR,
        EXPORT_PATH,
        WAVELENGTHS,
        COORDS_FILE,
        USE_FITFULL,
        STATE_FILE,
        LOG_FILE,
    )
except ImportError: #this pulls the template but it doesn't have data so program will fail (no fix tho)
    from helpers.config_template import (
        DATA_DIR,
        EXPORT_PATH,
        WAVELENGTHS,
        COORDS_FILE,
        USE_FITFULL,
        STATE_FILE,
        LOG_FILE,
    )




# constants for Doppler shift calculations
#TODO: make this dynamic using abswlart2a.py
DWL_GLOBAL = -0.001559  # global delta-wavelength offset
C_KM_S = 300000.0  # speed of light in km/s

if not EXPORT_PATH:
    EXPORT_PATH = os.getcwd()

def load_coord1(path):
    """
    Reads a two-column whitespace file (skipping any '#…' header lines)
    and returns (base_x, base_y) as integer arrays.

    * This matches the coord format written by this script.
    """
    data = np.loadtxt(path, comments="#", usecols=(0, 1), dtype=int)
    return data[:, 0], data[:, 1]  # returns the data as two seperate arrays

def load_coord(path, xfile, yfile):
    # don't put path in file name
    import numpy as np
    # using numpy's built in load txt function
    x = np.loadtxt(path+xfile)
    y = np.loadtxt(path+yfile)
    # need to become an integer type for the velocity calculation
    x = x.astype(int)
    y = y.astype(int)
    return x, y


def yshift(wl, wl0, y0=0):
    '''
    Parameters
    ----------
    wl : float
        The wavelength of the current line where the shift should be applied.
    wl0 : float
        The reference wavelength.
    y0 : float
        Initial detector position.

    Returns
    -------
    y : FLOAT
        the shift due to the spectrometer
    '''
    # slope of the first(1) and second(2) detector
    m1 = 0.08718
    m2 = 0.076586
    # the wavelength that the jump occurs at
    wls = 220.0
    # the size of the jump between spectrometers
    ys = 11.649

    # cases for applying the y-axis shift depending on the wavelengths
    if wl <= wls and wl0 < wls:
        return m1 * (wl - wl0)
    if wl >= wls and wl0 >= wls:
        return m2 * (wl - wl0)
    if wl >= wls and wl0 < wls:
        return m1 * (wls - wl0) + m2 * (wl - wls) + ys
    return m1 * (wl - wls) + m2 * (wls - wl0) - ys


def gauss3(x, a, b, c, d, e, f, g):
    """
    Double Gaussian function for fitting.
    """
    z = (x - b)**2 / (c**2) / 2.0
    z1 = (x - f)**2 / (g**2) / 2.0
    y = a * np.exp(-z) + d + e * np.exp(-z1)
    return y


def gauss2(x, a, b, c, d):
    """
    Single Gaussian function for fitting.
    """
    z = (x - b)**2 / (c**2) / 2.0
    y = a * np.exp(-z) + d
    return y

def fittest(wd, waves, xs, ys, labwl, z1, z2):
    """
    Fits Gaussian profiles to spectral lines and calculates velocities.

    Parameters:
    - wd: EIS data cube.
    - waves: Wavelength array.
    - inten: Intensity array.
    - xs: X-coordinates of points to analyze.
    - ys: Y-coordinates of points to analyze.
    - labwl: Laboratory wavelengths of the spectral lines.
    - z1: Lower bounds of spectral lines.
    - z2: Upper bounds of spectral lines.

    Returns:
    - v1: List of velocities for the first spectral line.
    - v2: List of velocities for the second spectral line (if applicable).
    - params: List of fitted parameters for each point.
    """
    C = 300000.0  # Speed of light in km/s
    if isinstance(labwl, (list, np.ndarray)):
        labwl_in = np.array(labwl, dtype=float)
    else:
        labwl_in = np.array([labwl], dtype=float)
    nlines = len(labwl_in)  # Number of spectral lines (1 or 2)
    if nlines not in (1, 2):
        raise ValueError("labwl must contain 1 or 2 values")

    order = np.argsort(labwl_in)
    labwl = labwl_in[order]

    # Velocities for the first spectral line
    v1_sorted = np.full(xs.shape, np.nan)
    # Velocities for the second spectral line
    v2_sorted = np.full(xs.shape, np.nan)
    fit_params = []  # Fitted parameters for each point

    wavecor = wd.meta["wave_corr"]
    wdet = waves[z1:z2]

    for k, (xp, yp) in enumerate(zip(xs, ys)):
        xi, yi = int(xp), int(yp)
        prof = wd.data[yi, xi, z1:z2]
        if not np.isfinite(prof).any():
            continue

        wl_lab = wdet - wavecor[yi, xi]
        amp = np.nanmax(prof)  # Maximum intensity
        sigma = 0.02  # Guess for line width
        back = amp / 100.0  # Estimate of background level

        if nlines == 1:
            p0 = [amp, labwl[0], sigma, back]
            ans, _ = curve_fit(gauss2, wl_lab, prof, p0=p0, maxfev=4000)
            v1_sorted[k] = (ans[1] - labwl[0]) / labwl[0] * C
            v2_sorted[k] = 0.0

        else:
            p0 = [amp, labwl[0], sigma, back, amp, labwl[1], sigma]
            ans, _ = curve_fit(gauss3, wl_lab, prof, p0=p0, maxfev=4000)
            v1_sorted[k] = (ans[1] - labwl[0]) / labwl[0] * C
            v2_sorted[k] = (ans[5] - labwl[1]) / labwl[1] * C
        fit_params.append(ans)  # Store fitted parameters

    if nlines == 1:
        return v1_sorted, v2_sorted, fit_params
    # reorder velocities to match original labwl input
    vel_out = [None, None]
    vel_out[order[0]] = v1_sorted
    vel_out[order[1]] = v2_sorted
    return vel_out[0], vel_out[1], fit_params


def fitfull(wd, waves, inten, labwl, z1, z2):
    """Compute velocity maps for the entire raster using Gaussian fits."""
    warnings.filterwarnings("error", category=OptimizeWarning)

    labwl = np.atleast_1d(labwl).astype(float)
    nlines = len(labwl)

    wavecor = wd.meta["wave_corr"]
    wavu = waves[z1:z2]
    ny, nx = wavecor.shape

    v1 = np.zeros((ny, nx))
    v2 = np.zeros((ny, nx))
    params = np.zeros((ny, nx, 7 if nlines == 2 else 4))
    nobad = []

    def _progress(current, total, bar_len=30):
        frac = current / float(total)
        filled = int(frac * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {frac*100:6.2f}%", end="", file=sys.stdout)

    for yi in range(ny):
        _progress(yi + 1, ny)
        for xi in range(nx):
            prof = inten[yi, xi, z1:z2]
            wl_lab = wavu - wavecor[yi, xi]
            a = np.max(prof)
            widguess = 0.02
            back = a / 100.0
            try:
                if nlines == 1:
                    p0 = [a, labwl[0], widguess, back]
                    ans, _ = curve_fit(gauss2, wl_lab, prof, p0)
                    v1[yi, xi] = (ans[1] - labwl[0]) / labwl[0] * C_KM_S
                    params[yi, xi, :4] = ans
                else:
                    p0 = [a, labwl[0], widguess, back, a, labwl[1], widguess]
                    ans, _ = curve_fit(gauss3, wl_lab, prof, p0)
                    v1[yi, xi] = (ans[1] - labwl[0]) / labwl[0] * C_KM_S
                    v2[yi, xi] = (ans[5] - labwl[1]) / labwl[1] * C_KM_S
                    params[yi, xi, :7] = ans
            except Exception:
                nobad.append(f"{xi}, {yi}")
                v1[yi, xi] = np.nan
                v2[yi, xi] = np.nan
                params[yi, xi, :] = np.nan

    print(file=sys.stdout)
    return v1, v2, params, nobad

#only do this if WAVELENGTHS exists
if WAVELENGTHS:
    # load wavelengths CSV and clean columns
    # use `_` so it's not shown in the variable explorer (leads to crowding)
    _csv = (
        pd.read_csv(WAVELENGTHS)
        .pipe(lambda df: df.rename(columns=lambda c: c.strip()))
        .rename(columns={"lines": "Ion", "wavelengths": "rest_wl"})
        .assign(rest_wl=lambda df: pd.to_numeric(df["rest_wl"], errors="coerce")).dropna(
            subset=["rest_wl"]
        )
    )


def findWLauto(w_auto, ion_label, max_delta=0.25):
    # helper to normalize ion label inline
    def lblformatter(s):
        s = s.lower().replace('-', ' ').replace('_', ' ')
        toks = s.split()
        if len(toks) < 2:
            return None
        elem = toks[0]
        num = toks[1]
        if num.isdigit():
            charge = int(num)
        else:
            ROMAN = dict(zip(
                ("CM", "M", "CD", "D", "XC", "C",
                 "XL", "L", "IX", "X", "IV", "V", "I"),
                (900, 1000, 400, 500, 90, 100, 40, 50, 9, 10, 4, 5, 1)
            ))
            # fallback Roman parsing
            i = 0
            val = 0
            num = num.upper()
            while i < len(num):
                for sym, v in ROMAN.items():
                    if num.startswith(sym, i):
                        val += v
                        i += len(sym)
                        break
                else:
                    val = 0
                    break
            charge = val
        return (elem, charge) if charge else None

    target = lblformatter(ion_label) if ion_label else None
    if not WAVELENGTHS:
        return [] #return nothing if csv not found
    
    # filter within range
    cand = _csv[
        _csv["rest_wl"].between(w_auto - max_delta, w_auto + max_delta)
    ].copy()

    if cand.empty:
        return []
    # if user gave an ion, prioritize matching ones
    if target:
        def _match(row):
            return lblformatter(row["Ion"]) == target
        same_ion = cand[cand.apply(_match, axis=1)]
        if not same_ion.empty:
            cand = same_ion

    cand = cand.sort_values("rest_wl")
    return list(cand[["rest_wl", "Ion"]].itertuples(index=False, name=None))


def select_paths(state, log):
    os.makedirs(DATA_DIR, exist_ok=True)
    cube_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(
        ".data.h5"))  # grab data files from directory
    log("\nFound " + str(len(cube_files)) +
        " EIS data cubes (" + DATA_DIR + "):")  # print them
    if cube_files:
        for i, f in enumerate(cube_files, 0):
            log(f"{i:2d}: {f}")
    else:
        log("  (none found)")
    sel = input("Enter file path or index: ")
    state["log"].append(sel)  # add the selection to the log file
    # interpret numeric choice or raw path
    if sel.isdigit() and 0 <= int(sel) < len(cube_files):
        dat_path = os.path.join(DATA_DIR, cube_files[int(sel)])
    else:
        dat_path = sel.strip("\"'")
    # validate user selection
    if not (dat_path.endswith(".data.h5") and os.path.exists(dat_path)):
        log("✗ Valid *.data.h5 not found — Aborting")
        return None
    head_path = dat_path.replace(".data.h5", ".head.h5")
    if not os.path.exists(head_path):
        log("✗ Matching *.head.h5 missing — Aborting")
        return None

    state["dat_path"] = dat_path  # update state
    state["head_path"] = head_path
    return state


def select_windows(state, log):
    # read state from select_paths
    wininfo = eis.read_wininfo(state["dat_path"])
    sel_data_file = os.path.basename(state["dat_path"])
    log("\nFound " + str(len(wininfo)) +
        f" windows in {sel_data_file}:")  # print windows
    for row in wininfo:
        log(f"{row[0]:2d}: {row[1]}")
    # ask which base window to use
    base_win_str = input("Base window index: ")
    state["log"].append(base_win_str)
    base_win = int(base_win_str)
    cmp_raw = input("Comparison windows (comma separated): ")
    state["log"].append(cmp_raw)  # append to state
    cmp_wins = [int(s) for s in cmp_raw.replace(" ", "").split(",") if s]

    state.update({"wininfo": wininfo, "base_win": base_win,
                 "cmp_wins": cmp_wins})  # update state
    # store all the variables that starttest.py normally stores
    # this lets you use art's original functions after this program finishes
    # also lets you debug the last study easier
    wininfo = eis.read_wininfo(state['dat_path'])
    state['wininfo'] = wininfo

    state['dwl'] = DWL_GLOBAL
    state['foundparams'] = [390.796, 185.21176, 0.0066, 173.04]

    win_ids = [state['base_win']] + state['cmp_wins']          # base + comps

    state['wd_all']     = {}   # EISCube objects
    state['inten_all']  = {}   # 3‑D intensity arrays
    state['intav_all']  = {}   # 2‑D average intensity maps
    state['waves_all']  = {}   # wavelength arrays

    with h5py.File(state['head_path'], 'r') as f:
        state['x_scale'], = f['pointing/x_scale']              # single value
        state['wave_corr'] = f['wavelength']['wave_corr'][:]   # one global copy

        for w in win_ids:
            wd_w = eis.read_cube(state['dat_path'], window=w)
            state['wd_all'][w]    = wd_w
            state['inten_all'][w] = wd_w.data
            state['intav_all'][w] = np.average(wd_w.data, axis=2)
            state['waves_all'][w] = np.asarray(f[f"wavelength/win{w:02d}"])

    # Legacy single‑window aliases (still point at the base window)
    state['window_index'] = state['base_win']
    state['wd']    = state['wd_all'][state['base_win']]
    state['inten'] = state['inten_all'][state['base_win']]
    state['intav'] = state['intav_all'][state['base_win']]
    state['waves'] = state['waves_all'][state['base_win']]
    state['inten1'] = state['inten'].copy()
    
    return state


def setupEISLines(state, log):
    # choose wavelengths and bounds for each window
    wininfo = state["wininfo"]
    all_wins = [state["base_win"]] + state["cmp_wins"]

    win_ids, labels, labwls, z_bounds = [], [], [], []

    dat = state["dat_path"]
    base_win = state["base_win"]

    # find brightest pixel in base window
    base_wd = eis.read_cube(dat, window=base_win)
    avg_img = np.average(base_wd.data, 2)
    qy, qx = np.unravel_index(np.nanargmax(avg_img), avg_img.shape)
    state["quick_x"], state["quick_y"] = int(qx), int(qy)  # add it to state

    for w in all_wins:
        lbl_raw = wininfo[w][1]
        elem = lbl_raw.rsplit(" ", 1)[0]

        # extract wavelength and look up possible rest lines
        auto_raw = float(re.search(r"(\d+\.\d+)", lbl_raw).group(1))
        auto_pairs = findWLauto(auto_raw, ion_label=elem)
        auto_wls = [p[0] for p in auto_pairs]

        if auto_wls:
            print(f"\nAuto‑detected for {elem} (win {w}): "
                  f"{', '.join(f'{x:.3f} Å' for x in auto_wls)}")
        else:
            print(f"\nNo rest‑line match found for {elem} – "
                  "you’ll need to type one manually.")
            
        #max is 2 labwl
        ans = input(f"{elem} wavelengths (blank=auto): ")
        state["log"].append(ans)  # adds answer to the log
        if ans:
            # adds wls to list if they are manually inputted, otherwise adds auto
            wls = [float(x) for x in ans.split(",")]
        else:
            wls = auto_wls

        win_ids.append(w)
        labwls.append(wls)
        # changes the name for when graphed
        labels.append(f"{elem} {', '.join(f'{x:.3f}Å' for x in wls)}")

        with h5py.File(state["head_path"], "r") as f:
            waves_arr = np.asarray(f[f"wavelength/win{w:02d}"])

        # show intensity trace and let user pick upper/lower bounds
        wde = eis.read_cube(dat, window=w)
        intene = wde.data[qy, qx, :]

        fig = plt.figure(num=f"idx-{elem}-{w}", clear=True)
        ax = fig.add_subplot(111)
        # plot vs. index instead of wavelength
        indices = np.arange(len(intene))
        ax.plot(indices, intene)
        ax.set_title(
            f"{labels[-1]} – index‑space  (click lower & upper bounds)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Intensity")
        ax.grid(True)

        # add secondary x-axis for wavelength
        # we want to save the coordinate by index but it's hard to tell which peak to bound if we don't also show wl
        # if we remove this, it's ok, it just removes the wl x-axis
        def idx_to_wl(idx_val):
            return np.interp(idx_val, indices, waves_arr)

        def wl_to_idx(wl_val):
            return np.interp(wl_val, waves_arr, indices)

        ax2 = ax.secondary_xaxis("top", functions=(idx_to_wl, wl_to_idx))
        ax2.set_xlabel("Wavelength (Å)")

        log("  Awaiting submission of upper and lower bounds...")

        clicks = []

        # custom click handling
        def on_click(event):
            if event.button == 1 and event.inaxes is ax:
                clicks.append(int(event.xdata))
                ax.axvline(event.xdata, color="r", ls="--")
                fig.canvas.draw_idle()
                if len(clicks) == 2:
                    fig.canvas.mpl_disconnect(cid)
                    plt.close(fig)

        cid = fig.canvas.mpl_connect("button_press_event", on_click)
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

        if len(clicks) != 2:
            raise RuntimeError(
                "Bounds selection cancelled — please try again.")

        # make it so that no matter lcick order, lower value is always first
        z1, z2 = sorted(clicks)
        z_bounds.append((z1, z2))

    state.update(
        {"win_ids": win_ids,
         "labels": labels,
         "labwls": labwls,
         "z_bounds": z_bounds}
    )
    return state

def plot_coords(state, log):
    dat = state["dat_path"]
    bw = state["base_win"]
    hd = state["head_path"]
    base_wd = eis.read_cube(dat, window=bw)
    avg_img = np.average(base_wd.data, 2)
    with h5py.File(hd, "r") as f:
        x_scale = f["pointing/x_scale"][()]
    log("\nPlot loop points (L-add, R-undo, Enter-finish)\n  Awaiting submission...\n")
    fig_map = plt.figure()
    ax_map = fig_map.add_subplot(111)
    im = ax_map.imshow(avg_img, origin="lower", aspect=1.0 / x_scale)
    #extent=base_wd.meta['extent_arcsec'] -- Replacement for aspect?? Need advising.
    fig_map.colorbar(im, ax=ax_map, label="Raw Intensity")
    plt.title("Intensity Map")
    ax_map.set_xlabel("Solar-X [arcsec]")
    ax_map.set_ylabel("Solar-Y [arcsec]")
    plt.show(block=False)
    pts = plt.ginput(n=-1, timeout=0)
    if not pts:
        log("✗ No points plotted — Aborting")
        plt.close(fig_map)
        return None
    # show selection briefly to later export if requested
    base_x, base_y = np.round(np.array(pts).T).astype(int)
    ax_map.plot(base_x, base_y, "r+", ms=6, mew=1)
    fig_map.canvas.draw_idle()
    plt.pause(0.1)
    plt.close(fig_map)
    state.update({"base_x": base_x, "base_y": base_y,
                 "intensity_map": fig_map})
    return state


def plot_vel_coords(state, log):
    """Plot the velocity map and let the user pick points."""
    vmap = state["vel_map"]
    x_scale = state["x_scale"]
    log("\nPlot loop points (L-add, R-undo, Enter-finish)\n  Awaiting submission...\n")
    fig_map = plt.figure()
    ax_map = fig_map.add_subplot(111)
    im = ax_map.imshow(vmap, origin="lower", aspect=1.0 / x_scale,
                       cmap="seismic", vmin=-40, vmax=40)
    fig_map.colorbar(im, ax=ax_map, label="Velocity [km s⁻¹]")
    lbl = state.get("labels", ["Velocity Map"])[0]
    plt.title(f"{lbl} Velocity Map")
    ax_map.set_xlabel("Solar-X (Raster pos [arcsec])")
    ax_map.set_ylabel("Solar-Y (Slit pos [arcsec])")
    plt.show(block=False)
    pts = plt.ginput(n=-1, timeout=0)
    if not pts:
        log("✗ No points plotted — Aborting")
        plt.close(fig_map)
        return None
    base_x, base_y = np.round(np.array(pts).T).astype(int)
    ax_map.plot(base_x, base_y, "r+", ms=6, mew=1)
    fig_map.canvas.draw_idle()
    plt.pause(0.1)
    plt.close(fig_map)
    state.update({"base_x": base_x, "base_y": base_y,
                 "velocity_map_fig": fig_map,
                 "vel_map_fig": fig_map})
    return state


def yshiftsall(state, log):
    # adjust y-coords for each wavelength based on yshift()
    dat = state["dat_path"]
    bw = state["base_win"]

    base_wd = eis.read_cube(dat, window=bw)  # grab the base window
    n_rows = base_wd.data.shape[0]

    base_labwl = state["labwls"][0]

    # flatten to single value if list
    if isinstance(base_labwl, (list, tuple)):
        base_labwl = base_labwl[0]

    bx, by = state["base_x"], state["base_y"]
    xs_all, ys_all = [], []

    for lw_entry in state["labwls"]:
        # if multiple lines, no shift needed

        if isinstance(lw_entry, (list, tuple)):
            xs_all.append(bx.copy())
            ys_all.append(by.copy())
            continue

        shift = int(round(yshift(lw_entry, base_labwl)))  # compute shift
        y_shift = by - shift  # run shift

        # check for off-map points
        off = (y_shift < 0) | (y_shift >= n_rows)
        if off.any():
            resp = input(
                f"{lw_entry:.3f} Å: {off.sum()} points off raster. gap them? (y/n): "
            )
            state["log"].append(resp)
            if resp.lower().startswith("y"):
                y_shift = y_shift.astype(float)
                y_shift[off] = np.nan

        # just duplicate the x points cause we dont shift them
        xs_all.append(bx.copy())
        ys_all.append(y_shift)  # add the points to the array

    state.update({"xs_all": xs_all, "ys_all": ys_all})
    return state


def fittestall(state, log, skip_base=False):
    """Fit velocity curves for each window/wavelength."""
    dat = state["dat_path"]
    hd = state["head_path"]

    vel_curves = {}

    for i, (win_id, (z1, z2), xs, ys, lbl, lwl) in enumerate(zip(
            state["win_ids"],
            state["z_bounds"],
            state["xs_all"],
            state["ys_all"],
            state["labels"],
            state["labwls"],
    )):
        if skip_base and i == 0:
            continue
        log(f"Fitting {lbl}…")

        # grab wd for each window (since we have multiple windows unlike Poland's code)
        wd = eis.read_cube(dat, window=win_id)
        # pull wavelength array from header file
        with h5py.File(hd, "r") as f:
            # grab waves for each window (same reason)
            waves = np.asarray(f[f"wavelength/win{win_id:02d}"])

        v1, v2, _ = fittest(wd, waves, xs, ys, lwl, z1, z2)

        if np.ndim(lwl) == 0 or len(np.atleast_1d(lwl)) == 1:  # if 1 wl for this run through

            lam = lwl[0] if isinstance(lwl, (list, tuple)) else lwl
            elem = lbl.split()[0] + " " + lbl.split()[1]
            key = f"{elem} {lam:.3f}Å"
            vel_curves[key] = v1

        else:  # if 2 (we dont declare it cus we force 1/2 wl when we pick wl earlier in the code)
            # mark double-Gauss entries
            elem = lbl.split()[0] + " " + lbl.split()[1]
            for lam, vel_arr in zip(lwl, (v1, v2)):
                key = f"{elem} {lam:.3f}Å [DG]"
                vel_curves[key] = vel_arr

    state["vel_curves"] = vel_curves  # save these for later
    return state


def run_fullfit(state, log, win_idx=0, show=True):
    """Compute full velocity map for a given window index."""
    dat = state["dat_path"]
    hd = state["head_path"]

    win_id = state["win_ids"][win_idx]
    labwl = state["labwls"][win_idx]
    z1, z2 = state["z_bounds"][win_idx]

    wd = eis.read_cube(dat, window=win_id)
    inten = wd.data
    with h5py.File(hd, "r") as f:
        waves = np.asarray(f[f"wavelength/win{win_id:02d}"])
        x_scale = f["pointing/x_scale"][()]

    v1, v2, params, bad = fitfull(wd, waves, inten, labwl, z1, z2)

    v1 = np.where(np.abs(v1) > 40., np.nan, v1)
   # TODO: 708: RuntimeWarning: All-NaN slice encountered
    med = np.nanmedian(v1[10:40, 0:40])
    if np.isfinite(med):
        v1 -= med

    fig = None
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(v1, origin="lower", aspect=1/x_scale,
                       cmap="seismic", vmin=-40, vmax=40)
        fig.colorbar(im, ax=ax, label="Velocity [km s⁻¹]")
        lbl = state.get("labels", [f"Window {win_id}"])[win_idx]
        plt.title(f"{lbl} Velocity Map")
        plt.show(block=True)

    return {
        "v1": v1,
        "v2": v2,
        "params": params,
        "bad": bad,
        "fig": fig,
        "x_scale": x_scale,
        "win_id": win_id,
    }


def run_fullfit_all(state, log):
    """Compute full velocity maps for all selected windows and show them."""
    maps = {}
    v2_all = {}
    params_all = {}
    bad_all = {}

    log("\n")
    # compute all maps first without showing them
    for i, win_id in enumerate(state["win_ids"]):
        log(f"Running fullfit for window {win_id}...")
        res = run_fullfit(state, log, win_idx=i, show=False)
        maps[win_id] = res["v1"]
        v2_all[win_id] = res["v2"]
        params_all[win_id] = res["params"]
        bad_all[win_id] = res["bad"]
        if i == 0:
            state.update({
                "vel_map": res["v1"],
                "x_scale": res["x_scale"],
                "full_params": res["params"],
                "full_v2": res["v2"],
                "bad_points": res["bad"],
            })

    state.update({
        "full_maps": maps,
        "full_v2_maps": v2_all,
        "full_params_maps": params_all,
        "bad_points_maps": bad_all,
    })

    # display all velocity maps together
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for idx, (ax, (win_id, arr)) in enumerate(zip(axes, maps.items())):
        im = ax.imshow(arr, origin="lower", aspect=1/state["x_scale"],
                       cmap="seismic", vmin=-40, vmax=40)
        lbl = state.get("labels", [f"Window {win_id}"])[idx]
        ax.set_title(lbl)
    fig.colorbar(im, ax=axes, label="Velocity [km s⁻¹]")
    plt.show(block=True)

    state.update({
        "velocity_map_fig": fig,
        "vel_map_fig": fig,
    })

    return state


def velocities_from_map(state, log):
    """Extract velocity values at user-selected points from the velocity map."""
    vmap = state["vel_map"]
    bx, by = state["base_x"], state["base_y"]
    vsel = vmap[by, bx]
    curves = state.get("vel_curves",{})
    curves[state["labels"][0]] = vsel
    state["vel_curves"] = curves
    
    return state


def displayVelocityTraces(state, log):
    # show all velocity traces on one plot
    vel_curves = state["vel_curves"]
    cols= "k r g b m c y".split()
    def pick_col(i):
        return cols[i % len(cols)]
    fig = plt.figure( figsize=(10,6))
    plt.grid(True)
    for i, (lbl, vel) in enumerate(vel_curves.items()):
        plt.plot(vel, "o-", ms=8, color=pick_col(i), label=lbl)
    # below are just graph labels (make it fancy lol)
    plt.xlabel("Point index")
    plt.ylabel("Velocity [km s⁻¹]")
    plt.title("Velocities")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show(block=True)
    state["velocity_plot_figure"] = fig
    return state


def makeSafe(name):
    # make a simple filesystem-safe name
    # were using some weird naming schemes so this is necessary
    """Make a filesystem‑safe folder/file name."""
    name = (name.replace(" Å", "").replace("Å", "")
            .replace(" ", "_").replace(",", "_"))
    name = re.sub(r"_+", "_", name).strip("_.")
    return name


def get_key(curves, base):
    """Return the first matching key in vel_curves for a given base name."""
    # exact match
    if base in curves:
        return base
    # numbered suffixes
    for i in range(1, 6):
        k = f"{base} ({i})"
        if k in curves:
            return k
    # combined wavelengths (comma‑separated)
    for k in curves:
        if k.startswith(base):
            return k
    raise KeyError(base)


def next_free(path):
    """Return `path` or `path_01`, `path_02`, … if it already exists."""
    # helps out when the user saves an export and doesnt name it soemthing unique (prevent overwrited)
    if not os.path.exists(path):
        return path
    base = os.path.basename(path)
    parent = os.path.dirname(path)
    existing = [d for d in os.listdir(
        parent) if d.startswith(base + "_")]  # check
    nums = [int(d.rsplit("_", 1)[-1])
            for d in existing if d.rsplit("_", 1)[-1].isdigit()]  # check num
    suffix = max(nums or [0]) + 1  # increment if needed
    # return new path to be used
    return os.path.join(parent, f"{base}_{suffix:02d}")


def debug_menu(state, log):
    """Simple helper to inspect values stored in *state*."""

    while True:
        sel = input(
            "\nDebug Options:\n"
            "1: List state keys\n"
            "2: Show array shapes\n"
            "3: Last 10 log entries\n"
            "Blank to return\n\n"
            "Selection: "
        )
        if not sel:
            return
        if sel == "1":
            log(", ".join(sorted(state.keys())))
        elif sel == "2":
            for k in ["inten", "intav", "vel_map"]:
                if k in state and hasattr(state[k], "shape"):
                    log(f"{k}: {state[k].shape}")
        elif sel == "3":
            for entry in state.get("log", [])[-10:]:
                print(entry)
        else:
            print("Invalid selection")


def load_study_menu(state, log):
    """Interactive menu for modifying a loaded study."""

    studies = [
        d for d in os.listdir(EXPORT_PATH)
        if os.path.isdir(os.path.join(EXPORT_PATH, d))
        and os.path.exists(os.path.join(EXPORT_PATH, d, STATE_FILE))
    ]
    if not studies:
        log("No saved studies were found - Starting a new session.")
        return None

    print("\nSaved studies:")
    for i, d in enumerate(studies, 0):
        print(f"{i}: {d}")

    sel = input("Enter study index to load: ")
    state["log"].append(sel)
    if not (sel.isdigit() and 0 <= int(sel) < len(studies)):
        log("Invalid selection - Starting a new session.")
        return None

    out_dir = os.path.join(EXPORT_PATH, studies[int(sel)])
    state = pickle.load(open(os.path.join(out_dir, STATE_FILE), "rb"))
    state["_loaded_dir"] = out_dir
    log(f"\nSuccesfully loaded the study from {out_dir}.")

    MENU = (
        "\nLoaded Study Menu:\n"
        "1: Select datacube\n"
        "2: Change window selection\n"
        "3: Change Å and Upper/Lower bounds\n"
        "4: Plot Points\n"
        "5: Re-Compute Y-Shifts\n"
        "6: Re-compute Fit & Show Results\n"
        "7: Plot Saved Results\n"
        "8: Export Results\n"
        "9: Debug\n"
        "Blank/Q: Done\n\n"
        "Selection: "
    )

    while True:
        mod = input(MENU).strip()
        state["log"].append(mod)
        if not mod or mod.lower().startswith("q"):
            return state

        step = mod
        if step == "1":
            select_paths(state, log)
        elif step == "2":
            select_windows(state, log)
        elif step == "3":
            setupEISLines(state, log)
        elif step == "4":
            if "vel_map" in state:
                plot_vel_coords(state, log)
            else:
                plot_coords(state, log)
        elif step == "5":
            yshiftsall(state, log)
        elif step == "6":
            if "vel_map" in state or "full_maps" in state:
                run_fullfit_all(state, log)
                if "base_x" in state and "vel_map" in state:
                    velocities_from_map(state, log)
                if state.get("cmp_wins"):
                    fittestall(state, log, skip_base=True)
            else:
                fittestall(state, log)
            displayVelocityTraces(state, log)
        elif step == "7":
            if "vel_curves" in state:
                displayVelocityTraces(state, log)
            else:
                log("No velocity curves to plot.")
        elif step == "8":
            export(state, log)
        elif step == "9":
            debug_menu(state, log)
        else:
            print("Invalid selection")


def export(state, log):
    # dump all results to disk if user agrees
    if not input("Save results? (y/n): ").lower().startswith("y"):
        log("✓ Done - No files saved.")
        return

    os.makedirs(EXPORT_PATH, exist_ok=True)
    # decide on output directory
    # if we loaded the study, made changes, and then saved, we run this
    if state.get("_loaded_dir"):
        choice = input("Overwrite (o), copy (c) or new name (n): ")
        orig = os.path.basename(state["_loaded_dir"])
        if choice.lower().startswith("o"):
            out_dir = state["_loaded_dir"]  # just overwrite
        elif choice.lower().startswith("c"):
            existing = [d for d in os.listdir(EXPORT_PATH)  # get existing path
                        if d.startswith(orig + "_copy_")]
            n = max([int(d.split("_")[-1]) for d in existing] or [0]) + 1
            out_dir = os.path.join(EXPORT_PATH, f"{orig}_copy_{n:02d}")
        else:
            sub = input(
                f'New folder name inside "{EXPORT_PATH}": ') or "untitled"
            # if we use new name, use this one
            out_dir = os.path.join(EXPORT_PATH, makeSafe(sub))
    else:
        sub = input(
            f'Folder name inside "{EXPORT_PATH}" (blank=untitled): ') or "untitled"
        # just ask for new (in this case, the study was created and not loaded)
        out_dir = next_free(os.path.join(EXPORT_PATH, makeSafe(sub)))

    os.makedirs(out_dir, exist_ok=True)
    log("→ Exporting to: " + out_dir)
    # save state and basic point lists
    pickle.dump(state, open(os.path.join(out_dir, STATE_FILE), "wb"))

    if "base_x" in state and "base_y" in state:
        np.savetxt(
            os.path.join(out_dir, "base_pts.txt"),
            np.vstack([state["base_x"], state["base_y"]]).T,
            fmt="%d %d", header="x   y   (base window)"
        )
    # save figures if present
    if fig := state.get("velocity_plot_figure"):
        fig.savefig(os.path.join(out_dir, "vel_plot.png"), dpi=300)
    if imap := state.get("intensity_map"):
        imap.savefig(os.path.join(out_dir, "intensity_map.png"), dpi=300)

    # organize fullfit outputs under a dedicated folder
    maps_dir = os.path.join(out_dir, "velocity_maps")
    if any(k in state for k in ("vel_map", "full_maps")):
        os.makedirs(maps_dir, exist_ok=True)

    if vmap := state.get("velocity_map_fig") or state.get("vel_map_fig"):
        vmap.savefig(os.path.join(maps_dir, "velocity_maps.png"), dpi=300)

    label_map = {}
    if "win_ids" in state and "labels" in state:
        for i, w in enumerate(state["win_ids"]):
            parts = state["labels"][i].split()
            label_map[w] = " ".join(parts[:2])

    def subdir(w):
        lbl = label_map.get(w, f"win{w:02d}")
        d = os.path.join(maps_dir, makeSafe(lbl))
        os.makedirs(d, exist_ok=True)
        return d

    base_win = state.get("base_win")
    if "vel_map" in state:
        d = subdir(base_win)
        np.savetxt(os.path.join(d, "velocity_map.txt"), state["vel_map"], fmt="%.3f")
        if "full_v2" in state and np.any(np.isfinite(state["full_v2"])):
            np.savetxt(os.path.join(d, "velocity_map_v2.txt"), state["full_v2"], fmt="%.3f")
        if "full_params" in state:
            np.save(os.path.join(d, "fit_params.npy"), state["full_params"])
        if "bad_points" in state and state["bad_points"]:
            with open(os.path.join(d, "bad_points.txt"), "w") as fb:
                fb.write("\n".join(state["bad_points"]))

    if "full_maps" in state:
        for win, arr in state["full_maps"].items():
            if win == base_win:
                continue
            d = subdir(win)
            np.savetxt(os.path.join(d, "velocity_map.txt"), arr, fmt="%.3f")
            v2m = state.get("full_v2_maps", {}).get(win)
            if v2m is not None and np.any(np.isfinite(v2m)):
                np.savetxt(os.path.join(d, "velocity_map_v2.txt"), v2m, fmt="%.3f")
            prm = state.get("full_params_maps", {}).get(win)
            if prm is not None:
                np.save(os.path.join(d, "fit_params.npy"), prm)
            bad = state.get("bad_points_maps", {}).get(win)
            if bad:
                with open(os.path.join(d, "bad_points.txt"), "w") as fb:
                    fb.write("\n".join(bad))
    # prepare shifted point lists
    if "xs_all" in state and "ys_all" in state and "vel_curves" in state:
        pts_x_rep, pts_y_rep = [], []
        for j, lw in enumerate(state["labwls"]):
            n = len(lw) if isinstance(lw, (list, tuple)) else 1
            pts_x_rep.extend([state["xs_all"][j]] * n)
            pts_y_rep.extend([state["ys_all"][j]] * n)
        vel_curves = state["vel_curves"]
        idx = 0

        # for each wl, create folder, save vel, and save shifted pts
        for j, lw_entry in enumerate(state["labwls"]):
            elem = state["labels"][j].split()[0] + " " + \
                state["labels"][j].split()[1]
            is_multi = isinstance(lw_entry, (list, tuple)) and len(lw_entry) > 1

            if not is_multi:
                lam = lw_entry[0] if isinstance(
                    lw_entry, (list, tuple)) else lw_entry
                win_dir = os.path.join(out_dir, makeSafe(f"{elem} {lam:.3f}"))
                os.makedirs(win_dir, exist_ok=True)

                base_key = f"{elem} {lam:.3f}Å"
                key = get_key(vel_curves, base_key)
                np.savetxt(os.path.join(win_dir, "vel.txt"),
                           vel_curves[key], fmt="%.3f",
                           header="Velocity [km s⁻¹]")
                np.savetxt(os.path.join(win_dir, "pts.txt"),
                           np.vstack([pts_x_rep[idx], pts_y_rep[idx]]).T,
                           fmt="%d %.1f", header="x   y (shift‑corrected)")
                idx += 1

            else:
                win_dir = os.path.join(out_dir, makeSafe(
                    f"{elem} (Double Gaussian)"))
                os.makedirs(win_dir, exist_ok=True)

                for lam in lw_entry:
                    base_key = f"{elem} {lam:.3f}Å"
                    key = get_key(vel_curves, base_key)
                    sub_dir = os.path.join(win_dir, makeSafe(
                        f"{lam:.3f}"))
                    os.makedirs(sub_dir, exist_ok=True)

                    np.savetxt(os.path.join(sub_dir, "vel.txt"),
                               vel_curves[key], fmt="%.3f",
                               header="Velocity [km s⁻¹]")
                    np.savetxt(os.path.join(sub_dir, "pts.txt"),
                               np.vstack([pts_x_rep[idx], pts_y_rep[idx]]).T,
                               fmt="%d %.1f", header="x   y (shift‑corrected)")
                    idx += 1

    log("✓ Done - Files Saved.")  # last call

    # write out log
    with open(os.path.join(out_dir, LOG_FILE), "w") as f:  # if we save data
        # check each entry save dto the log array in state dict
        for entry in state.get("log", []):
            # write a log (we do this last because I want any logs from export to be incl.)
            f.write(entry + "\n")


def main():
    os.makedirs(EXPORT_PATH, exist_ok=True)
    state = {"log": []}  # define state and log

    def printAndLog(msg):  # add any printed messages to log by using this
        print(msg)
        state["log"].append(msg)

    # printed normally here because i Don't want this saved to the log (it doesnt matter)
    print('\n* Note: Always verify automatically generated information to ensure accuracy.\n')
    # option to load a previous run
    load_ans = input("Load previous study? (y/n): ")
    state["log"].append(load_ans)
    if load_ans.lower().startswith("y"):
        loaded = load_study_menu(state, printAndLog)
        if loaded is not None:
            return loaded

    # full new session workflow
    # if it's not loaded, we start here with loading the EIS file
    if select_paths(state, printAndLog) is None:
        return

    select_windows(state, printAndLog)
    setupEISLines(state, printAndLog)

    if USE_FITFULL:
        run_fullfit_all(state, printAndLog)
        resp = input("Plot points on velocity map? (y/n): ").lower()
        if resp.startswith("y"):
            if plot_vel_coords(state, printAndLog) is None:
                return
            yshiftsall(state, printAndLog)
            fittestall(state, printAndLog, skip_base=True)
            velocities_from_map(state, printAndLog)
            displayVelocityTraces(state, printAndLog)
            export(state, printAndLog)
            return state
        else:
            export(state, printAndLog)
            return state
    else:
        if COORDS_FILE:
            resp = input(
                f"Load coords from {COORDS_FILE}? (y/n): ").lower()
            if resp.startswith("y"):
                bx, by = load_coord1(COORDS_FILE)
                state.update({"base_x": bx, "base_y": by})
            else:
                if plot_coords(state, printAndLog) is None:
                    return
        else:
            if plot_coords(state, printAndLog) is None:
                return

        yshiftsall(state, printAndLog)
        fittestall(state, printAndLog)

    displayVelocityTraces(state, printAndLog)
    export(state, printAndLog)
    return state


if __name__ == "__main__":
    state = main()
