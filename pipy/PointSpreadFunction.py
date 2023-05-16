#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Yuchen Liu
Affiliation: Cavendish Astrophysics, University of Cambridge
Email: yl871@cam.ac.uk

Created in April 2023

Description:

"""

from scipy import fft
from ImageCube import *

class PointSpreadFunction(ImageCube):
    def __init__(self,psf_cube,spatial_tapering=None):
        super().__init__(psf_cube)
        self.spatial_tapering = spatial_tapering
        
    def ft_psf(self):
        if self.spatial_tapering != None:
            self.data *= self.spatial_filter(self.spatial_tapering)[None,:,:]
        ft2d = fft.fftshift(fft.fft2(fft.ifftshift(self.data,axes=(1,2)),axes=(1,2)),axes=(1,2))
        ft2d[ft2d==0] = 1e-10
        self.vis_weight = ft2d
        return ft2d
        
    @property
    def weights(self):
        return 2 * self.wsclean_neff * self.vis_weight
        
    @property
    def power_weights(self):
        weight = np.abs(self.weights).mean(0)**2
        weight /= weight.max()
        return weight


