import pandas as pd
import numpy as np
import os

os.chdir('data')
# our measured reflectance and transmittance data for PdSe2 nanoflakes
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

