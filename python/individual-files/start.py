import eispac as eis  # for eis codes
import numpy as np  # for mathematics and arrays
import matplotlib.pyplot as plt  # for plotting

# import keyboard                  #to see when keys are pressed
import sys  # for going through all of the data
import h5py  # for h5 files
import astropy.units as u  # astropy functions
from time import sleep
from scipy.optimize import curve_fit

c = 3.0e5  # speed of light, will need later
# for this run for yours you will need to replace d:/EISdata
# fildat='eis_20211204_115903.data.h5'  #good data
# file_name='d:/EISdata/eis_20221214_121103.data.h5'
# file_header='d:/EISdata/eis_20221214_121103.head.h5'
# filhead='eis_20211204_115903.head.h5'
fold = "/Users/canyon/Documents/ASSIP/data/"
# fildat='eis_20240701_224742.data.h5'
# fold='d:/EISdata/'
# fildat='eis_20070601_180758.data.h5'
# fildat='eis_20221214_121103.data.h5'
# filhead='eis_20221214_121103.head.h5'
# filhead='eis_20070601_180758.head.h5'
# filhead='eis_20240701_224742.head.h5'
# 2/22/25
# 5/12/25  fold='C:/Users/artpo/OneDrive/Desktop/EISPython/newdata/trythis/'
fildat = "eis_20100217_042526.data.h5"
filhead = "eis_20100217_042526.head.h5"
# fold='C:/Users/artpo/OneDrive/Desktop/artnewdata/' #new 5/12/25
# new fildat='eis_20240915_184525.data.h5'
# new filhead='eis_20240915_184525.head.h5'
fil = fold + fildat
filehead = fold + filhead
wininfo = eis.read_wininfo(fil)
# dwl and parameters found for this data set
dwl = -0.001559
foundparams = [3.90796e2, 185.21176, 0.0066, 173.04]
print(wininfo)
sleep(1)
print("enter window number")
z = input()
i = int(z)
sleep(1)
print("window number=", i)
wd = eis.read_cube(fil, window=i)
inten = wd.data
intav = np.average(inten, 2)
f_head = h5py.File(filehead, "r")  # open the header file to read
(x_scale,) = f_head["pointing/x_scale"]  # get the scaling for  x axis
if i < 10:
    window = "win0" + str(i)
if i > 9:
    window = "win" + str(i)
waves = np.array(f_head["wavelength/" + window])  # get wavelengths
# plt.plot(wave,inten[200,40,:])
# %matplotlib Qt5Agg    #for gui plot,need to run d uring execution  ????
plt.figure(1)
# plt.imshow(intav,origin='lower',aspect=1/x_scale)
# plt.show()
# pts=plt.ginput(n=-1,timeout=120)
inten1 = np.array(inten)  # in case inten gets changed
print("inten", inten1[20, 20, 3])
z = f_head["wavelength"]  # gets 1 level down
z1 = z["wave_corr"]  # to get to wave_cor
# offset=z1[:,:]   # to make an accessible array
# tempath='C:/Users/artpo/anaconda3/Lib/site-packages/eispac/data/templates/'
# example template=tempath+'fe_11_180_401.1c.template.h5'
# tmpl='fe_12_195_119.1c.template.h5'
# fit_res=eis.fit_spectra(wd,template,ncpu='max')
# to get the params out params=fit_res.fit['params']
# useful statement z=fit_res.fit.keys()  or whatever you want
# print('offset')
print("got to end")
# show1(intav,x_scale)


# def show1(intav,x_scale):
def show1(intav, x_scale):
    import matplotlib.pyplot as plt

    plt.imshow(intav, origin="lower", aspect=1.0 / x_scale)
    return


# need a function for applying the yshift on to the second spectrometer
def yshift(wl, wl0, y0=0):
    # the result needs to be subtracted from previous measured y
    '''
    # Parameters
    # ----------
    wl: FLOAT
    #   the wavelegnth of the current line where the shift should be applied
    # x,y=    wl0 : FLOAT
    #        the base wavelength
    #   y0 : FLOAT
    #       the initial position

    #  Returns
    # -------
    #  y : FLOAT
    #    the shift due to the spectrometer
    '''
    # slope of the first(1) and second(2) detector
    m1 = 0.08718
    m2 = 0.076586
    # the wavelength that the jump occurs at
    wls = 220.0
    # the size of the jump between spectrometers
    ys = 11.649
    # the different cases that can exist for the yshift and the shirt to apply depending on the case
    if wl <= wls and wl0 < wls:
        y = m1 * (wl - wl0)
    elif wl >= wls and wl0 >= wls:
        y = m2 * (wl - wl0)
    elif wl >= wls and wl0 < wls:
        y = (m1 * (wls - wl0)) + (m2 * (wl - wls)) + ys
    elif wl <= wls and wl0 >= wls:
        y = (m1 * (wl - wls)) + (m2 * (wls - wl0)) - ys
    return y
