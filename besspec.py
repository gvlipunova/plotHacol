import numpy.random
from numpy.random import rand
from numpy import *
from scipy.optimize import fsolve
from scipy.special import expn, jv, yn
from scipy.integrate import simpson
from scipy.interpolate import interp1d

import bassun as bs

import configparser as cp
conffile = 'globals.conf'
config = cp.ConfigParser(inline_comment_prefixes="#")
config.read(conffile)

ifplot =  config['DEFAULT'].getboolean('ifplot')

if ifplot:
    import plots
    from matplotlib.pyplot import ioff
    ioff()

mass1 = 1.5 # !!!temporary: NS mass in Msun
rstar  = 4.86 # !!! temporary: NS radius in GM/c^2
norm = 3.28396e-06 * mass1 * rstar*1.5 # multiplier, if omega is in seconds


def BesspecDS(x, omega):
    '''
    omega in seconds
    '''
    
    return x**3 * (yn(2, omega*norm) * jv(2, omega*norm * x**1.5) - yn(2, omega*norm * x**1.5) * jv(2, omega*norm))
    

def BesspecOmega(xs, nroots, outputn = -1):
    '''
    finds the solution(s) of the equation J2(2/3 omega/omegaK(rstar))Y2(2/3 omega/omegaK(rshock)) - Y2(2/3 omega/omegaK(rstar))J2(2/3 omega/omegaK(rshock))
    that gives the spectrum of vertical p-modes
    '''
    fpulse = lambda x: jv(2, norm*x) * yn(2,norm*x*xs**1.5) - jv(2, norm*x*xs**1.5) * yn(2,norm*x)

    nome = 1000
    omega = arange(nome)/double(nome) / norm * 10.

    wchange = where(fpulse(omega[1:])*fpulse(omega[:-1])<0.)
    
    quintanaroots = fsolve(fpulse, omega[wchange])
    print(quintanaroots[0])
    
    plots.someplots([omega, quintanaroots], [fpulse(omega), quintanaroots*0.], name='fpulse', xtitle=r'$\omega$, s', formatsequence = ['-k', '*r'], xlog=True, multix = True)

    firstroots = quintanaroots[0:nroots]

    nx = 100
    x = arange(nx)/double(nx) * (xs-1.) + 1.

    fs = []

    for k in arange(nroots):
        fs.append(BesspecDS(x, quintanaroots[k]))

    plots.someplots(x, fs, name='fpulse_sol', xtitle=r'$R/R_*$', xlog=True)

    if outputn >= 0:
        fsoln = interp1d(x, fs[outputn], 'linear', fill_value="extrapolate")
        return fsoln
    
