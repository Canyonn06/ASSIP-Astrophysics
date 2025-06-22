import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fittest1(wd, waves, inten, xpts, ypts, labwl):
    """
    Fits Gaussian profiles to spectral lines and calculates velocities.

    Parameters:
    - wd: EIS data cube.
    - waves: Wavelength array.
    - inten: Intensity array.
    - xpts: X-coordinates of points to analyze.
    - ypts: Y-coordinates of points to analyze.
    - labwl: Laboratory wavelengths of the spectral lines.

    Returns:
    - v1: List of velocities for the first spectral line.
    - v2: List of velocities for the second spectral line (if applicable).
    - params: List of fitted parameters for each point.
    """
    print("Input lower wavelength index:")
    z1 = int(input())
    print("Input upper wavelength index:")
    z2 = int(input())

    c = 300000.0  # Speed of light in km/s
    wavecor = wd.meta["wave_corr"]
    wavu = waves[z1:z2]
    nlines = len(labwl)  # Number of spectral lines (1 or 2)
    npts = len(xpts)

    v1 = []  # Velocities for the first spectral line
    v2 = []  # Velocities for the second spectral line
    params = []  # Fitted parameters for each point

    for m in range(npts):
        xp = xpts[m]
        yp = ypts[m]
        intu = inten[yp, xp, z1:z2]
        wu = wavu - wavecor[yp, xp]

        a = np.max(intu)  # Maximum intensity
        widguess = 0.02  # Guess for line width
        backrnd = a / 100.0  # Estimate of background level

        if nlines == 1:
            p0 = [a, labwl[0], widguess, backrnd]
            # p0 = [a, centre_wl, 0.05, backrnd]   # width guess a bit wider (0.05 Å)
            ans, pcov = curve_fit(gauss2, wu, intu, p0)
            vel1 = (ans[1] - labwl[0]) / labwl[0] * c
            vel2 = 0.0
            params.append(ans)  # Store fitted parameters
        elif nlines == 2:
            p0 = [a, labwl[0], widguess, backrnd, a, labwl[1], widguess]
            ans, pcov = curve_fit(gauss3, wu, intu, p0)
            vel1 = (ans[1] - labwl[0]) / labwl[0] * c
            vel2 = (ans[5] - labwl[1]) / labwl[1] * c
            params.append(ans)  # Store fitted parameters

        v1.append(vel1)
        v2.append(vel2)

    return v1, v2, params


def gauss3(x, a, b, c, d, e, f, g):
    """
    Double Gaussian function for fitting.
    """
    z = (x - b) ** 2 / (c**2) / 2.0
    z1 = (x - f) ** 2 / (g**2) / 2.0
    y = a * np.exp(-z) + d + e * np.exp(-z1)
    return y


def gauss2(x, a, b, c, d):
    """
    Single Gaussian function for fitting.
    """
    z = (x - b) ** 2 / (c**2) / 2.0
    y = a * np.exp(-z) + d
    return y


def waveplot(inten, x, y, fign):
    """
    Plots intensity profiles at specific (x, y) points.
    """
    intu = inten[y, x, :]
    plt.figure(fign)
    plt.plot(intu)
    plt.show(block=False)
    plt.pause(0.001)
    return
