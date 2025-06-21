# Configuration template for main.py
# Copy this file to config.py and fill in your local paths.

# *.data.h5 & *.head.h5 live here
DATA_DIR = "/abs/path/to/eis_data/"

# where exported plots and logs will be saved
EXPORT_PATH = "/abs/path/to/exports/"

# optional rest-λ table
WAVELENGTHS = "/abs/path/to/wavelengths.csv"

# optional base_pts.txt produced by this script
COORDS_FILE = ""

# False: intensity→vel→plot ; True: fullfit→vel-map
USE_FITFULL = False

# reloadable study snapshot
STATE_FILE = "state.pkl"

# console log filename
LOG_FILE = "console.log"
