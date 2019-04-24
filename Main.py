import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as opt


# TMM function will be optimized (error minimization)
# n1, n3: real refractive indices of 1st (air) and 3rd layer (KBr)
# x = [n2, k2]: complex refractive index of 2nd layer (PdSe2 nanoflake), optimization parameter
# wi: wavelength
# Ri, Ti: reflectance and transmittance at wavelength wi
# d2: thickness of 2nd layer (PdSe2 nanoflake)
def TMM(x):
    n2 = x[0] + x[1]*1j
    r1 = (n1 - n2) / (n1 + n2)
    t1 = 2 * n1 / (n1 + n2)
    r2 = (n2 - n3) / (n2 + n3)
    t2 = 2 * n2 / (n2 + n3)
    k2 = 2 * np.pi * n2 / wi

    rfilm = (r1 + r2 * np.exp(-2 * 1j * k2 * d2)) / (1 + r1 * r2 * np.exp(-2 * 1j * k2 * d2))
    tfilm = (t1 * t2 * np.exp(-1j * k2 * d2)) / (1 + r1 * r2 * np.exp(-2 * 1j * k2 * d2))
    rfilm = rfilm * np.conj(rfilm)
    tfilm = abs(n3 / n1 * tfilm * np.conj(tfilm))

    error = (Ri - rfilm) ** 2 + (Ti - tfilm) ** 2
    return error

os.chdir('data')
# our measured reflectance and transmittance data for PdSe2 nanoflakes
thickness = 135  # nm
flake_names = ['wavenumber', 'wavelength', 'sample', 'reference', 'null']
R_flake = pd.read_csv('R_PdSe2_135nm.csv', names=flake_names)
T_flake = pd.read_csv('T_PdSe2_135nm.csv', names=flake_names)

# our measured transmittance data for reference substrates: KBr and quartz
substrate_names = ['wavenumber', 'wavelength', 'sample', 'reference']
T_KBr = pd.read_csv('T_KBr.csv', names=substrate_names)
T_quartz = pd.read_csv('T_quartz.csv', names=substrate_names)

# Thorlabs measured reflectance data for reference mirror: Au
mirror_names = ['wavelength', 'reflectance']
R_Au = pd.read_csv('R_Au.csv', names=mirror_names)

# Transmittance of reference KBr and quartz substrates and reflectance of reference Au mirror
T_KBr = np.array(T_KBr['sample'])/np.array(T_KBr['reference'])
T_quartz = np.array(T_quartz['sample'])/np.array(T_quartz['reference'])
R_Au = np.flip(np.array(R_Au['reflectance']), axis=0)

b = 3395
# PdSe2 flake reflectance
w = np.array(R_flake['wavelength'])[:b]
R_f = 0.935
R_s = np.array(R_flake['sample'])
R_r = np.array(R_flake['reference'])
R_n = np.array(R_flake['null'])
R = R_f * R_Au * (R_s - R_n)/(R_r - R_n)
R = R[:b]
R_Au = R_Au[:b]

# PdSe2 flake transmittance
T_f = 0.883
T_s = np.array(T_flake['sample'])
T_r = np.array(T_flake['reference'])
T_n = np.array(T_flake['null'])
T = T_f * T_KBr * (T_s - T_n)/(T_r - T_n)
T = T[:b]
T_KBr = T_KBr[:b]

# Single-pass model
f = T/(1-R)
nm2cm = 1e-7
d = thickness * nm2cm
a_sp = 1/d * np.log(1/f)  # single-pass absorption coefficient
hv = 1.24/w  # photon energy
cody_sp = np.sqrt(a_sp/hv)  # single-pass Cody

# Multiple-reflection model (thin-film interference)
nm2um = 1e-3
d2 = thickness * nm2um
# indices of refraction
n1 = 1.0003  # air
n3 = 1.54  # KBr
x0 = [5, 1]  # [n2, k2], PdSe2 flake
n, k = [], []

# Nelder-Mead optimization (error minimization)
for i in range(w.size):
    wi, Ri, Ti = w[i], R[i], T[i]
    xopt = opt.fmin(func=TMM, x0=x0, disp=False)
    n.append(xopt[0])
    k.append(xopt[1])
    x0 = xopt

n, k = np.array(n), np.array(k)
um2cm = 1e-4
a_mr = 4 * np.pi * k / (w * um2cm)
cody_mr = np.real(np.sqrt(a_mr/hv))

ax = plt.subplot(221)
plt.scatter(w, R, s=1)
plt.scatter(w, T, s=1)
ax.set_xlabel('Wavelength'), ax.set_ylabel('R, T')
ax.set_ylim(0, 1)
ax.legend(['R_flake', 'T_flake'], frameon=False)

ax = plt.subplot(222)
plt.scatter(w, R_Au, s=1)
plt.scatter(w, T_KBr, s=1)
ax.set_xlabel('Wavelength'), ax.set_ylabel('R, T')
ax.set_ylim(0, 1)
ax.legend(['R_Au', 'T_KBr'], frameon=False)

ax = plt.subplot(223)
plt.scatter(w, f, s=1)
ax.set_xlabel('Wavelength'), ax.set_ylabel('T/(1-R)')
ax.set_ylim(0.5, 1)
ax.legend(['Flake'], frameon=False)

ax = plt.subplot(224)
plt.scatter(hv, cody_sp, s=1)
plt.scatter(hv, cody_mr, s=1)
ax.set_xlabel('hv'), ax.set_ylabel('Cody')
ax.set_xlim(0.4, 1.42), ax.set_ylim(0, 400)
ax.legend(['Single-pass', 'Multiple-reflection'], frameon=False)

plt.show()