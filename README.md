# ASSIP EIS Velocity Analysis

A Python script that wraps and extends Dr. Poland’s EIS-analysis utilities so you can  

* select spectral windows  
* choose points on an intensity map  
* compute single- or double-Gaussian Doppler velocities  
* save / reload a study state  
* export velocity plots, shifted-point lists, and a complete console log  

Built on top of the **[eispac](https://eispac.readthedocs.io)** package.

Produced by Dr. Poland's ASSIP Cohort '25 using Dr. Art Poland's provided code.

---

## 🔗 Helpful links

1. https://solarb.mssl.ucl.ac.uk/SolarB/SearchArchive.jsp (Search for a dataset)
2. https://eis.nrl.navy.mil/level1/hdf5/. (Download a dataset)
3. https://iopscience.iop.org/article/10.1086/529378/pdf (To find best wavelength)

---

## 📋 Requirements

| Core |
|------|
| Python ≥ 3.8 |
| eispac |
| numpy ≥ 1.18 |
| scipy ≥ 1.4 |
| matplotlib ≥ 3.1 |
| h5py ≥ 2.9 |
| astropy ≥ 4.2.1 |
| sunpy ≥ 4.0 |
| ndcube ≥ 2.0 |
| pyqt ≥ 5.9 |
| parfive ≥ 1.5 |
| python-dateutil ≥ 2.8 |
| tomli ≥ 1.1.0 (Py < 3.11) |

| Extra |
|------|
| pandas |
| Spyder IDE (recommended) |

---

## ⚙️ Installation

New to coding? Follow these one‑time steps.

1. **Install Python** – grab the latest version from
   [python.org](https://www.python.org/downloads/). On Windows make sure the
   installer option “Add Python to PATH” is checked.

2. **Install Git or GitHub Desktop** – the easiest way is the graphical
   [GitHub&nbsp;Desktop](https://desktop.github.com/) app. Command‑line users can
   instead install Git from [git-scm.com](https://git-scm.com/downloads).

3. **Get this project**
   * Using GitHub Desktop: choose **File → Clone repository** and enter
     `https://github.com/Canyonn06/ASSIP-Astrophysics.git`.
   * Using the command line:

     ```bash
     git clone https://github.com/Canyonn06/ASSIP-Astrophysics.git
     cd ASSIP-Astrophysics
     ```
   To pull later updates just run `git pull` inside the folder.

4. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

5. **Install the required packages**

   ```bash
   pip install -r requirements.txt
   ```

   EISPAC itself is published on PyPI. If needed, you can also install or
   upgrade it directly:

   ```bash
   python -m pip install eispac              # fresh install
   python -m pip install --upgrade eispac    # upgrade to the latest version
   ```

6. **Pull latest changes**
   * Using Github desktop or the command line interface to pull changes everytime you start a new session.
   * Using the command line:
      ```bash
      git pull
      ```

---

## 🔧 Configuration

1. Copy `helpers/config_template.py` → `helpers/config.py`.
2. Edit the paths inside `config.py` to match your machine.
   (The file is ignored by git so your local settings stay private.)

Example options:

```python
DATA_DIR    = "/abs/path/to/eis_data/"   # *.data.h5 & *.head.h5 live here
EXPORT_PATH = "/abs/path/to/exports/"
WAVELENGTHS = "/abs/path/to/wavelengths.csv"   # optional rest-λ table
COORDS_FILE = ""      # optional base_pts.txt produced by this script
USE_FITFULL = False   # False: intensity→vel→plot ; True: fullfit→vel-map

STATE_FILE = "state.pkl"   # reloadable study snapshot
LOG_FILE   = "console.log"
```

When `USE_FITFULL` is enabled the program now calculates velocity maps for **all
selected windows first** and then asks about plotting points.

---

## 🚀 Workflow

1. **Launch**

   ```bash
   python main.py
   ```

2. **Pick a dataset** – enter the list index or paste a full `.data.h5` path.  
3. **Select windows** – choose a base window and any comparison window(s).  
4. **Set wavelengths** – accept auto-detected rest lines or type your own (comma separated).  
   *Click two points* on the index-space plot to mark the fit bounds.  
5. **Plot points** on the intensity map  
   * **L-click** add • **R-click** undo • **Enter** finish  
6. Script computes y-shifts, fits Gaussians and pops up a velocity-vs-point plot.  
7. **Export?** Type `y` to save:

   ```
   state.pkl            base_pts.txt
   velocity_plot.png    intensity_map.png
   vel.txt / pts.txt    (per window / λ)
   console.log
   ```
   Full velocity maps from `USE_FITFULL` are stored inside a new
   `velocity_maps/` subdirectory of your export folder.

8. **Reloading** – run the script and answer **y** when asked to load a study.
   After choosing one you will get a small menu. Pick a step to rerun, export
   or open the debug section. After each step you are returned to the menu so
   you can tweak just the parts you need.

---

## 📂 Dr. Poland’s Original Files

Keep his modules (e.g. `start.py`, `fitful.py`) in the same folder or on `PYTHONPATH`.  
You can import and mix-and-match:

```python
from eispac import read_wininfo
from fitful import yshift, gauss2, gauss3
import pickle

state = pickle.load(open("state.pkl", "rb"))
print(read_wininfo(state["dat_path"]))
```

---

## 🔍 What’s in `state`

Use the values stored in `state` to utilize Art Poland's files (generally for debugging purposes).

| key | type / contents |
|-----|-----------------|
| `log` | `list[str]` – every prompt & console message |
| `dat_path` | `str` – absolute path to the *.data.h5 cube |
| `head_path` | `str` – absolute path to the *.head.h5 cube |
| `_loaded_dir` | `str | None` – export folder when re-loading a study |
| `wininfo` | `list[tuple[int, str]]` – rows from `eis.read_wininfo` |
| `base_win` | `int` – selected base window index |
| `cmp_wins` | `list[int]` – comparison window indices |
| `win_ids` | `list[int]` – `[base_win] + cmp_wins` |
| `dwl` | `float` – global Δλ applied (default −0.001559 Å) |
| `foundparams` | `list[float]` – initial guess for Gaussian fits |
| `wd_all` | `dict[int, eis.EISCube]` – EISCube objects per window |
| `inten_all` | `dict[int, np.ndarray]` – 3-D intensity arrays |
| `intav_all` | `dict[int, np.ndarray]` – 2-D avg-intensity maps |
| `waves_all` | `dict[int, np.ndarray]` – wavelength arrays per window |
| `x_scale` | `float` – solar-x scaling from header |
| `wave_corr` | `np.ndarray` – wavelength correction matrix |
| `window_index` | `int` – alias pointing to `base_win` |
| `wd` | `eis.EISCube` – alias for base window cube |
| `inten` | `np.ndarray` – alias for base intensity cube |
| `intav` | `np.ndarray` – alias for base avg-intensity map |
| `waves` | `np.ndarray` – alias for base wavelength array |
| `inten1` | `np.ndarray` – copy of `inten` (legacy) |
| `labels` | `list[str]` – pretty names for each window / λ |
| `labwls` | `list[list[float]]` – chosen rest wavelengths per window |
| `z_bounds` | `list[tuple[int, int]]` – lower/upper fit indices per window |
| `quick_x`, `quick_y` | `int` – brightest pixel coords (base window) |
| `base_x`, `base_y` | `np.ndarray` – user-picked points (base window) |
| `intensity_map` | `matplotlib.figure.Figure` – figure w/ plotted points |
| `xs_all`, `ys_all` | `list[np.ndarray]` – shift-corrected coords per λ |
| `vel_curves` | `dict[str, np.ndarray]` – velocity arrays keyed by name |
| `velocity_plot_figure` | `matplotlib.figure.Figure` – combined velocity plot |

---

## 📝 Tips & Gotchas

* Backend forced to **Qt5Agg** → keep a Qt-capable backend available.  
* Zooming can drop a rogue point; **R-click** to delete it.  
* In IPython run `%matplotlib` before spawning extra figures.  
* Spyder’s variable explorer is perfect for live-inspecting `state`.

---
