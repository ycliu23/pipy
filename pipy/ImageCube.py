#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Yuchen Liu
Affiliation: Cavendish Astrophysics, University of Cambridge
Email: yl871@cam.ac.uk

Created in April 2023

Description:

"""

import numpy as np
from scipy.signal import windows
from astropy.io import fits
from Cosmology import *

class ImageCube:
    def __init__(self,image_cube):
        # load data
        ## in fits file: x - axis 1, y - axis 2, freq - axis 3
        ## in array: (freq,y,x)
        self.data, self.header = fits.getdata(image_cube,header=1)

        # pixel number
        self.Nx = self.header['NAXIS1']
        self.Ny = self.header['NAXIS2']
        
        # reference pixel
        self.Nx_ref = self.header['CRPIX1']
        self.Ny_ref = self.header['CRPIX2']
        
        # frequency axis
        self.n_channel = self.header['NAXIS3']
        self.nu_start = self.header['CRVAL3'] / 1e6  # starting frequency, MHz
        self.nu_interval = self.header['CDELT3'] / 1e6 # interval size, MHz
        self.freq_array = self.nu_start + self.nu_interval*np.arange(self.n_channel) # MHz

        self.nu_min = self.freq_array.min() # MHz
        self.nu_max = self.freq_array.max() # MHz
        
        print('>>> Data cube contains %i channels and each channel is in shape of (%i, %i)'%(self.n_channel,self.Ny,self.Nx))
        
        # central frequency, wavelength and redshift
        self.freq_central = np.mean(self.freq_array) # MHz
        self.lambda_central = (const.c / (self.freq_central*u.MHz)).to(u.m).value # metre
        self.z_central = z_obs(self.freq_central)

        # cosmology
        self.D_C = cosmo.comoving_distance(self.z_central).value # LoS comoving distance at central z, Mpc
        self.D_M = cosmo.comoving_transverse_distance(self.z_central).value # transverse comoving distance at central z, Mpc

        # pixel size
        if np.abs(self.header['CDELT1']) == np.abs(self.header['CDELT2']):
            self.pixel_size = np.abs(self.header['CDELT1']) # degree
        else:
            print('Pixel sizes are differnt along x and y directions')
            
        # weighting scheme
        self.weight_type = self.header['WSCWEIGH']
            
        # synthesis beam size
        self.bmaj = self.header['BMAJ'] # degree
        self.bmin = self.header['BMIN'] # degree
        
        # normalization
        self.wsclean_norm_factor = self.header['WSCNORMF']
        self.wsclean_neff = self.header['WSCNVIS']

        # data unit
        self.unit = self.header['BUNIT']
        
    def convert_to_Tb(self,synthesis_beam=None):
        """
        Convert Jy/PSF to mK
        
        I - Jy/beam
        nu - MHz
        B_maj & B_min - arcsec
        """
        nu = self.freq_array
        
        if synthesis_beam == None:
            if self.weight_type == 'natural':
                synthesis_beam = False
            elif self.weight_type == 'uniform':
                synthesis_beam = True
            else:
                raise ValueError('Unrecognized weighting scheme')
        elif synthesis_beam == 'psf':
            pass
        elif synthesis_beam in [True, False]:
            pass
        else:
            raise ValueError("'synthesis_beam' takes input: None, 'psf', True and False")
        
        if synthesis_beam == 'psf':
            beam_area = (self.pixel_size*u.degree)**2
        elif not synthesis_beam:
            beam_area = (self.Nx * self.Ny * (self.pixel_size*u.degree)**2 / self.wsclean_norm_factor)
        elif synthesis_beam:
            B_maj = self.bmaj*u.degree
            B_min = self.bmin*u.degree
            beam_area = np.pi/(4*np.log(2)) * B_maj * B_min
        else:
            raise ValueError('Incorrect type of beam approximation')
            
        equiv = u.brightness_temperature(nu*u.MHz)
        convert_factor = (u.Jy/beam_area).to(u.mK, equivalencies=equiv)
        convert_array = np.broadcast_to(convert_factor,self.data.T.shape).T

        self.data *= convert_array
        self.header['BUNIT'] = 'mK'
        self.unit = self.header['BUNIT']
        
    def save_Tb(self,outfile,overwrite=False):
        Tb_fits = fits.PrimaryHDU(data=self.data,header=self.header)
        Tb_fits.writeto(outfile+'.fits',overwrite=overwrite)
        
    def spatial_filter(self,name):
        window = windows.get_window(name,self.Nx)
        x = np.arange(-(self.Nx_ref-1),self.Nx-(self.Nx_ref-1))
        y = np.arange(-(self.Ny_ref-1),self.Ny-(self.Ny_ref-1))
        r = np.sqrt(x[None,:]**2 + y[:,None]**2)
        return np.interp(r,x,window)
        
    def freq_filter(self,name):
        return windows.get_window(name,self.n_channel)

        
