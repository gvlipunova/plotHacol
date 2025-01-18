from numpy import *
import numpy.ma as ma

from matplotlib import gridspec

import os
import sys
import glob

import h5py

# from scipy.optimize import root_scalar
# from scipy.integrate import simpson
# from scipy.integrate import cumulative_trapezoid as cumtrapz

from os.path import exists

import matplotlib
from matplotlib.pyplot import *

import subsonic as sub

cmap = 'viridis'

mass1 = 1.5 # NS mass in Solar masses
xim = 0.5 # magnetosphere radius in RA
rstar = 6.77159 / mass1 * 10./10. # NS radius in GM/c^2 for physical R = 10km

afac = 0.25
drrat = 0.25

# mdot is mass accretion rate in Eddington units LEdd/c^2, assuming kappa = 0.35cm^2/g
# mu30 is magnetic moment in 10^{30}G cm^3 units

def RAlf(mdot, mu30):
    # Alfven radius in GM/c^2 units
    return 1577.74 * mu30**(4./7.)/mdot**(2./7.)/mass1**(10./7.)

def theta0(mdot, mu30):
    return arcsin(sqrt(rstar / RAlf(mdot, mu30)/xim))

def lcor(mdot, mu30):
    # luminosity of the subsonic flow in Eddington units
    cth = cos(theta0(mdot, mu30))
    return afac / drrat * (2. * log((1.+cth)/(1.-cth))-3.*cth)


def shooting_beta(th0, k, verbose = True):
    '''
    restores inner f by choosing proper fout
    '''
    
    theta_out = pi/2.
    
    lf1 = -2.0 ; lf2 = 2.0 ; tol = 1e-10 # brackets and tolerance for lg(fout)
    
    theta, fint1, u1 = sub.uint(th0, 10.**lf1, k, theta_out = theta_out, firstpoint = True)
    theta, fint2, u2 = sub.uint(th0, 10.**lf2, k, theta_out = theta_out, firstpoint = True)

    if verbose:
        print("fout = ", 10.**lf1, ": f(0) = ", fint1, "; u[-1] = ", u1)
        print("fout = ", 10.**lf2, ": f(0) = ", fint2, "; u[-1] = ", u2)

    # magnetic energy density ratio logarithm
    umagrat0 = -12. * log(sin(th0)) + log(1.+3.*cos(th0)**2)
    umagrat_out = -12. * log(sin(theta_out)) + log(1.+3.*cos(theta_out)**2)
    umagrat = umagrat0 - umagrat_out
    
    ucrit1 = u1- umagrat -log(3.)
    ucrit2 = u2 - umagrat - log(3.)

    while (abs(lf2-lf1) >  tol ):
        lf = (lf1+lf2)/2.
        theta, fint, u = sub.uint(th0, 10.**lf, k, theta_out = theta_out, firstpoint = True)
        ucrit = u - umagrat - log(3.)
        if verbose:
            print("fout = ", 10.**lf, ": f(0) = ", fint)
        if ((ucrit*ucrit1) >= 0.):
            lf1 = lf
        else:
            lf2 = lf

    fout = 10.**((lf1+lf2)/2.)
            
    theta, fint, u = sub.uint(th0, 10.**lf, k, theta_out = theta_out, firstpoint = True)
            
    return fint / 0.75 * sin(th0)**2, fout
        
mdot1 = 1.0 ; mdot2 = 1000.0 ; nmdot = 200
mdot = (mdot2/mdot1)**(arange(nmdot)/double(nmdot-1))*mdot1
muar1 = 0.01 ; muar2 = 10. ; nmu = 201
mu = (muar2/muar1)**(arange(nmu)/double(nmu-1))*muar1

mdotar, muar = meshgrid(mdot, mu)

karr = afac / drrat * RAlf(mdotar, muar) * xim / rstar / mdotar

beta_fint = copy(mdotar) * 0.

fint = copy(mdotar)

print(shape(beta_fint))

for kmdot in arange(nmdot):
    for kmu in arange(nmu):
        beta_tmp, fint_tmp = shooting_beta(theta0(mdot[kmdot], mu[kmu]), karr[kmu, kmdot], verbose  = False)
        beta_fint[kmu, kmdot] = beta_tmp
        fint[kmu, kmdot] = fint_tmp
        # ii = input('beta')

betarad_approx = 1.-lcor(mdotar, muar)/mdotar
betarad = beta_fint

excluded = betarad < 1.-rstar/RAlf(mdotar, muar)/xim

betarad_masked = ma.masked_array(betarad, mask = (betarad > 1.) | (betarad < 0.))

clf()
fig, ax = subplots()
pc = ax.pcolormesh(muar, mdotar, log10(fint), vmin = 0., vmax = 1.0)
cb = colorbar(pc)
contour(muar, mdotar, fint, [1.0], colors = 'k', linewidths = 4.)
ax.set_xlabel(r'$\mu$, $10^{30}$G cm$^3$') ; ax.set_ylabel(r'$\dot{M}c^2/L_{\rm Edd}$')
ax.set_xscale('log') ; ax.set_yscale('log')
fig.set_size_inches(10.,8.)
savefig('blimits_fint.png')


clf()
fig, ax = subplots()
pc = ax.pcolormesh(muar, mdotar, betarad, vmin = 0., vmax = 1.0)
cb = colorbar(pc, ax = ax)
cb.set_label(r'$\beta$')
ax.contour(muar, mdotar, betarad, [2./3.], colors = 'k', linewidths = 4.)
c = ax.contour(muar, mdotar, betarad, [0.5, 0.75, 0.9, 0.99], colors = 'k', linewidths = 1.)
c0 = ax.contour(muar, mdotar, betarad_approx, [0.5, 0.75, 0.9, 0.99], colors = 'k', linewidths = 1., linestyles=':')

formatstring = ['0.5', '0.75', '0.9', '0.99']
fmt = {}
for l, s in zip(c.levels, formatstring):
    fmt[l] = s

ax.clabel(c, c.levels, inline_spacing = -10, fmt = fmt)
ax.clabel(c0, c0.levels, inline_spacing = -10, fmt = fmt)
cs1 = ax.contourf(muar, mdotar, excluded, [-1, 0., 1., 2.], linestyles = 'solid', hatches=['', '//'], alpha = 1.0)
cs1.set_facecolor('none')
cs1.set_edgecolor('w')
# bar.set_edgecolor('k')
# cs2 = ax.contourf(muar, mdotar, betarad, [-10., 0., 1.], hatches=['\\', '', '||'], alpha = 1.0)
cs2 = ax.contourf(muar, mdotar, fint, [-10., 0., 1., 1000.], hatches=['\\', '', '\\', '||'], alpha = 1.0)
cs2.set_facecolor('none')
cs2.set_edgecolor('w')
ax.set_xlabel(r'$\mu$, $10^{30}$G cm$^3$') ; ax.set_ylabel(r'$\dot{M}c^2/L_{\rm Edd}$')
ax.set_xscale('log') ; ax.set_yscale('log')
fig.set_size_inches(10.,8.)
savefig('blimits.png')
