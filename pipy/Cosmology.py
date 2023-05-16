#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Yuchen Liu
Affiliation: Cavendish Astrophysics, University of Cambridge
Email: yl871@cam.ac.uk

Created in April 2023

Description:

"""

import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM

freq_21cm = 1420.4 # MHz
H0 = 100 # km/s/Mpc
Omega_m0 = 0.30964
cosmo = FlatLambdaCDM(H0=H0,Om0=Omega_m0)

def z_obs(nu):
    """
    nu - MHz
    """
    return freq_21cm / nu - 1
