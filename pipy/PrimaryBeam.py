#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Yuchen Liu
Affiliation: Cavendish Astrophysics, University of Cambridge
Email: yl871@cam.ac.uk

Created in April 2023

Description:

"""

from ImageCube import *

class PrimaryBeam():
    def __init__(self,wavelength,pixel_size,Nx_ref,Ny_ref): # kind='gaussian'
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.Nx_ref = Nx_ref
        self.Ny_ref = Ny_ref
        
    @property
    def FWHM(self):
        return 1.02*self.wavelength/38

    @property
    def sigma(self):
        return self.FWHM / (2*np.sqrt(2*np.log(2)))
        
    @property
    def theta(self):
        pixel_size_rad = np.radians(self.pixel_size)
        theta_x = pixel_size_rad * np.arange(-(self.Nx_ref-1),self.Nx-(self.Nx_ref-1))
        theta_y = pixel_size_rad * np.arange(-(self.Ny_ref-1),self.Ny-(self.Ny_ref-1))
        theta_grid = np.sqrt(theta_x[None,:]**2 + theta_y[:,None]**2)
        return theta_grid
    
    @property
    def beam(self):
        gaussian_beam = np.exp(-0.5*(self.theta/self.sigma)**2)
        return gaussian_beam
        
    

