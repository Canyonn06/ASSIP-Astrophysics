import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning

warnings.filterwarnings('error', category=OptimizeWarning) #converts OptimizeWarning to a fatal error

def fitful(wd, waves, inten, labwl):
    """ #This program gets the velocity of every point in the image.
# It returns all the v1 v2 and params
    #Fits Gaussian profiles to spectral lines and calculates velocities.
 #before you run it, plot a spectrum as a function of pixel so you know
#what the lower and upper indexes are for the profile(s)

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
    nobadpts=[]
    print('Input lower wavelength index:')
    z1 = int(input())
    print('Input upper wavelength index:')
    z2 = int(input())
    print('will print every 50th y so you know its working')
    c = 300000.0  # Speed of light in km/s
    wavecor = wd.meta['wave_corr']  #this is the wavelength correction array
    wavu = waves[z1:z2]
    nlines = len(labwl)  # Number of spectral lines (1 or 2)
    z=wavecor.shape  #get the x and y dimensions of the image
    #these next several lines of code set up arrays to take the velocities as numbers
    m1=z[1]  #number of x points
    n1=z[0]  #number of y points
    print('z=',z)
    v1=np.zeros((n1,m1))  # Velocities for the first spectral line
    q=v1.shape
    print('shapev1',q)
    v2=np.zeros((n1,m1))
    params=np.zeros((n1,m1,4))
    if nlines ==2:
        params=np.zeros((n1,m1,7))
    k=z2-z1  #number of wavelength points in array
    for m in range(n1-1):  #loop over y points
        mm=np.round(m/50)*50
        if mm==m:
           print('y=',m)
        for n in range(m1-1):  #loop over x points
            k1=0   #This next loop gets rid of bad data points - may not really help
            # while k1<k:   #loop over wavelengths to fix bad points
            #     if inten[m,n,k1+z1]<0.:  #if there is a really bad point that will kill the
            #                            #curve_fit program, set it to 1. This seems to work
            #         inten[m,n,k1+z1]=1.
            #     k1=k1+1
            xp = n
            yp = m
            #print(xp,yp)
            intu = inten[yp, xp, z1:z2]
            wu = wavu - wavecor[yp, xp] #Applies the wavelength correction
            a = np.max(intu)  # Maximum intensity
            widguess = 0.02  # Guess for line width
            backrnd = a / 100.0  # Estimate
            try: #attempts to run the code below (lines 73-90)
                if nlines == 1:
                    wl0=labwl[0]
                    #if n == 32 and m==310:
                     #   print(wu,intu)
                    p0 = [a, wl0, widguess, backrnd]  #sets up initial guess
                    ans, pcov = curve_fit(gauss2, wu, intu, p0) 
                    v1[m,n]= (ans[1] - wl0) /wl0* c
                    #if n == 32 and m==310:
                     #   print('ans=',ans,ans[1],labwl[0],c,v1[m,n])
                    v2[m,n]=0.
                    #vel1.append(v1)
                    #vel2.append(v2)
                    params[m,n,:]=ans
                    #if n==32 and m==310:
                     #   print('got to params')
                    #params.append(ans)  # Store fitted parameters
                elif nlines == 2:
                    p0 = [a, labwl[0], widguess, backrnd, a, labwl[1], widguess]
                    ans, pcov = curve_fit(gauss3, wu, intu, p0)
                    v1[m,n] = (ans[1] - labwl[0]) / labwl[0] * c
                    v2[m,n] = (ans[5] - labwl[1]) / labwl[1]* c
                    params[m,n,:]=ans
                    #vel1.append(v1)
                    #vel2.append(v2)
                    #params.append(ans)  # Store fitted parameters
            except: #if the code in the try block fails, run this (lines 92-99) and continue
                #print('error',m,n)
                nobadpts.append(f"{n}, {m}")
                # v1[m,n]=0.
                # v2[m,n]=0.
                # params[m,n,:]=0.
                v1[m,n]=np.nan
                v2[m,n]=np.nan
                params[m,n,:]=np.nan
                #may want to clean up v result with 
                #newv=np.where(abs)v1 < 40.,v1,0.)

    return v1, v2, params,nobadpts
# use newv=np.where(abs(v1) < 40., v1,0.)  to set abs val > 40 to 0.
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

#save results to file in working folder
def saveresult(fold,v1,v2,params,labwl):
    title1='v'+str(labwl[0])
    if np.size(labwl) ==2:
         title2='v'+str(labwl[1])
         np.save(fold+title2,v2)
    title3='params'+str(labwl[0])
    np.save(fold+title1, v1)
    np.save(fold+title3,params)
    return

def restoreresult(fold,file):
    #file has to have the name of the file in the folder
    result=np.load(fold+file)
    return result

#cleanup
def cleanup(v,limit):
    import numpy as np
#return array 2x(number of bad pts) [0,:] bad y indexes [1,:] bad x indexes
    size=v.shape
    ny=size[0]
    nx=size[1]
    z1=0
    k=0
    k1=0
    for k1 in range(ny-1):
      for k in range(nx-1):
        if abs(v[k1,k]) > 40.:
            v[k1,k] = 0.
            z1=z1+1
      k=k+1
    k1=k1+1
    z=np.where(v ==0.)
    z2=np.array(z)
# z1[0,:] and z1[1,:] have indexes of bad v values
    return z2,z1
