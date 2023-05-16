#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Yuchen Liu
Affiliation: Cavendish Astrophysics, University of Cambridge
Email: yl871@cam.ac.uk

Created in April 2023

Description:

"""

import numexpr as ne
from scipy import fft
import matplotlib.pyplot as plt
from ImageCube import *
from PointSpreadFunction import *
from PrimaryBeam import *

class PowerSpectrum(ImageCube):
    def __init__(self,image_cube,psf_cube=None,
                 flux_to_Tb=True,synthesis_beam=None,
                 spatial_tapering=None,frequency_tapering=None,
                 Lmin=None,Lmax=None,
                 demean=True,demean_axis=None):
        
        super().__init__(image_cube)
        print('>>> Image cube initialized')
        
        if psf_cube != None:
            self.psf = True
            psf_cube = PointSpreadFunction(psf_cube,spatial_tapering)
            psf_cube.ft_psf()
            self.vis_weight = psf_cube.vis_weight
            self.power_weight = psf_cube.power_weights
            print('>>> PSF cube initialized')
        elif psf_cube == None:
            self.psf = False
        else:
            raise ValueError('Incorrect type of PSF cube ')
                         
        if flux_to_Tb:
            print('>>> Converting Jy/beam to mK...')
            synthesis_beam = None
            if self.psf:
                synthesis_beam = 'psf'
            self.convert_to_Tb(synthesis_beam)
            print('>>> Unit conversion completed')
            
        # primary beam
        pb = PrimaryBeam(self.lambda_central,self.pixel_size,self.Nx_ref,self.Ny_ref)
        
        # window function
        if spatial_tapering != None:
            self.spatial_window = self.spatial_filter(spatial_tapering)[None,:,:]
        elif spatial_tapering == None:
            self.spatial_window = np.array([1])
        
        if frequency_tapering != None:
            self.frequency_window = self.freq_filter(frequency_tapering)[:,None,None]
        elif frequency_tapering == None:
            self.frequency_window = np.array([1])
        else:
            print('Incorrect type of window function')

        # baseline limit
        if Lmin == None:
            self.Lmin = 1e-10
        else:
            self.Lmin = Lmin
        if Lmax == None:
            self.Lmax = 1e10
        else:
            self.Lmax = Lmax
            
        # subtract mean before Fourier transform
        self.demean = demean
        self.demean_axis = demean_axis

    @property
    def dr_perp(self):
        """
        Step size of transverse comoving distance, determined by angular pixel size
        """
        pixel_size_rad = np.radians(self.pixel_size) # radian
        return self.D_C * pixel_size_rad # Mpc
    
    @property
    def dr_los(self):
        """
        Step size of LoS comoving distance, determined by frequency interval
        """
        return (const.c * (1+self.z_central)**2 * (self.nu_interval*u.MHz) \
               /(cosmo.H(self.z_central) * (freq_21cm*u.MHz))).to(u.Mpc).value # Mpc

    def nu_s(self,axis):
        """
        Sampling frequency along x,y,z directions [1/Mpc]
        """
        if axis in ['x','y']:
            return 1/self.dr_perp
        elif axis == 'z':
            return 1/self.dr_los
        else:
            print('input out of axes')
    
    def k_axis(self,axis):
        """
        Morales, M. F. and Hewitt, J. (2004), Eqn (2) & (3)
        k_x = 2pi * u / D_M
        k_y = 2pi * v / D_M
        """
        if axis == 'x':
            u = fft.fftshift(fft.fftfreq(self.Nx))
            k_x = 2 * np.pi * u / self.dr_perp
            return k_x
        
        elif axis == 'y':
            v = fft.fftshift(fft.fftfreq(self.Ny))
            k_y = 2 * np.pi * v / self.dr_perp
            return k_y
        
        elif axis == 'z':
            k_z = 2 * np.pi * fft.fftshift(fft.fftfreq(self.n_channel)) / self.dr_los
            return k_z
        
        else:
            print('input out of axes')
            
    @property
    def k_perp_grid(self):
        """
        k_perp grid, in shape of (ky, kx)
        """
        k_x = self.k_axis('x')
        k_y = self.k_axis('y')
        return np.sqrt(k_x[None,:]**2 + k_y[:,None]**2)
    
    @property
    def k_perp_min_baseline(self):
        """
        Lmin - shortest baseline in metre
        """
        return (2 * np.pi * (self.Lmin*u.m) * (freq_21cm*u.MHz) \
                / (const.c * (1+self.z_central) * (self.D_M*u.Mpc))).to(1/u.Mpc).value
    
    @property
    def k_perp_max_baseline(self):
        """
        Lmax - longest baseline in metre
        """
        return (2 * np.pi * (self.Lmax*u.m) * (freq_21cm*u.MHz) \
                / (const.c * (1+self.z_central) * (self.D_M*u.Mpc))).to(1/u.Mpc).value

    @property
    def k_perp(self):
        k_perp = self.k_perp_grid
        n_k_perp = int((self.Nx+self.Ny)/2)
        k_perp_min = np.maximum(k_perp.min(),self.k_perp_min_baseline)
        k_perp_max = np.minimum(k_perp.max(),self.k_perp_max_baseline)
        return np.linspace(k_perp_min,k_perp_max,n_k_perp)

    @property
    def k_los(self):
        k_z = self.k_axis('z')[self.k_axis('z') > 0]
        return k_z
    
    @property
    def dk_perp(self):
        return np.diff(self.k_perp).mean()
    
    @property
    def dk_los(self):
        return np.diff(self.k_los).mean()
        
    @property
    def cosmo_volume(self):
        """
        Equation (19) - (23), Mertens et al. (2020)
        """
        angle_to_comoving_distance = self.D_M # Mpc per radian
        freq_to_comoving_distance = (self.lambda_central*u.m * (1+self.z_central) / cosmo.H(self.z_central)).to(u.Mpc/u.MHz).value # Mpc per MHz
        
        cube_size = self.Nx * self.Ny * self.n_channel
        spatial_resolution = np.radians(self.pixel_size) # rad
        frequency_resolution = self.nu_interval # MHz
        
        A_eff = (pb.beam**2 * self.spatial_window**2).mean()
        B_eff = (self.frequency_window**2).mean()
        
        norm = angle_to_comoving_distance**2*freq_to_comoving_distance * spatial_resolution**2*frequency_resolution / cube_size / A_eff / B_eff
        
        return norm

    def spatial_power_spectrum(self,read_ps3d=False,input_ps3d=None):
        """
        Normalized 3D power spectrum of the brightness temperature cube [mK^2 Mpc^3]
        """
        print('>>> Computing 3D Fourier transform of input data cube...')
        
        if read_ps3d:
            self.ps3d = fits.getdata(input_ps3d)
            self.ps3d = np.array(self.ps3d)
        else:
            # spatial transform
            self.data *= self.spatial_window
            FT_2d = fft.fftshift(fft.fft2(fft.ifftshift(self.data,axes=(1,2)),axes=(1,2)),axes=(1,2))
            if self.psf:
                FT_2d /= self.vis_weight
            if self.demean:
                FT_2d -= FT_2d.mean(self.demean_axis)
            
            # delay transform
            FT_2d *= self.frequency_window
            Tb_FT = fft.fftshift(fft.fft(FT_2d,axis=0),axes=0) # mK
                
            power_21cm_3d = np.abs(Tb_FT)**2 * self.cosmo_volume # mK^2
            if self.n_channel % 2:
                power_21cm_3d = power_21cm_3d[self.n_channel//2+1:] + power_21cm_3d[:self.n_channel//2][::-1]
            else:
                power_21cm_3d = power_21cm_3d[self.n_channel//2+1:] + power_21cm_3d[1:self.n_channel//2][::-1]
                
            print('>>> Spatial 3D power spectrum completed')
                
            self.ps3d = power_21cm_3d
        return self.ps3d
    
    def cylindrical_power_spectrum(self,read_ps2d=False,input_ps2d=None):
        """
        Cylindrically averaged power spectrum - (type, k_los, k_perp)
        ps2d[0]: mean; ps2d[1]: stderr; ps2d[2]: num of effective cells
        """
        if read_ps2d:
            self.ps2d = fits.getdata(input_ps2d)
            
        else:
            try:
                self.ps3d
            except:
                self.spatial_power_spectrum()
            
            print('>>> Cylindrically averaging power spectrum...')
            rho = self.k_perp_grid
            k_z = np.abs(self.k_axis('z'))
            
            k_perp = self.k_perp
            dk_perp = self.dk_perp
            k_los = self.k_los

            k_perp_bins = np.append(k_perp-dk_perp/2,[k_perp[-1]+dk_perp*1e-5])
            k_perp_bins[0] = k_perp[0]-dk_perp*1e-5
            index_k_perp = np.digitize(rho,k_perp_bins) - 1

            ps2d = np.zeros((3,len(k_los),len(k_perp)))
            for i in range(len(k_perp)):
                iy, ix = np.where(i == index_k_perp)
                for j in range(len(k_los)):
                    grid_value = self.ps3d[j,iy,ix]
                    n_grid = len(grid_value)
                    
                    if self.psf:
                        grid_weight = self.power_weight[iy,ix]
                    else:
                        grid_weight = None
                        
                    ps2d_value = np.average(grid_value,weights=grid_weight)
                    ps2d_error = np.sqrt(np.average(np.full(n_grid,ps2d_value)**2,weights=grid_weight)/n_grid)
                    
                    ps2d[0,j,i] = ps2d_value
                    ps2d[1,j,i] = ps2d_error
                    ps2d[2,j,i] = n_grid

            self.ps2d = ps2d
            
        print('>>> 2D power spectrum completed')
        return self.ps2d
        
    @property
    def instrument_limit(self):
        k_perp_grid = self.k_perp_grid
        k_perp_min_baseline = self.k_perp_min_baseline
        k_perp_max_baseline = self.k_perp_max_baseline

        limit = np.ones_like(self.ps3d,dtype=bool)
        iy, ix = np.where((k_perp_grid <= k_perp_min_baseline) | (k_perp_grid >= k_perp_max_baseline))
        limit[:,iy,ix] = False # mask regions where k_perp < L_min and k_perp > L_max
        return limit
    
    def eor_window(self,fov):
        """
        fov - field of view in degree
        """
        
        k_los_wedge = self.k_perp_grid * (np.sin(np.radians(fov)) * cosmo.H(self.z_central) * (self.D_C*u.Mpc) \
                      / (const.c * (1+self.z_central))).to(1).value

        self.k_los_wedge = k_los_wedge
              
        # mask
        eor_filter = np.ones_like(self.ps3d,dtype=bool)
        for i, k in enumerate(k_los_wedge): ##########################
            eor_filter[self.k_los < k_los_wedge,i] = False # mask regions where k_par < k_wedge

        self.eor_filter = eor_filter
        return eor_filter

    @property
    def k_grid(self):
        k_x = self.k_axis('x')[None,None,:]
        k_y = self.k_axis('y')[None,:,None]
        k_z = self.k_los[:,None,None]
        grid = ne.evaluate('sqrt(k_x**2 + k_y**2 + k_z**2)')
        grid[~self.instrument_limit] = np.nan
        return grid
    
    def k(self,n_k):
        """
        k-bins to calculate 1D power spectrum
        """
        k_min = np.sqrt(self.k_perp.min()**2 + self.k_los.min()**2)
        k_max = np.sqrt(self.k_perp.max()**2 + self.k_los.max()**2)
        log_k = np.linspace(np.log10(k_min),np.log10(k_max),n_k)
        k_central_value = 10**log_k
        self.dk = np.diff(log_k).mean() # interval size of k-bins, in log scale
        self.k_bin_edge = 10**np.append(log_k[:-1]-np.diff(log_k)/2,
                                        [log_k[-1]-self.dk/2,log_k[-1]+self.dk/2])
        self.k_bin_edge[0] = k_central_value.min()-1e-8
        self.k_bin_edge[-1] = k_central_value.max()+1e-8
        
        return k_central_value
        
    def spherical_power_spectrum(self,n_k=50,dimensionless=True,
                                 eor_window=False,fov=None):
        """
        Spherically averaged 1D power spectrum - (k,Pk,err)
        """
        try:
            self.ps3d
        except:
            self.spatial_power_spectrum()
        
        print('>>> Spherically averaging power spectrum...')
        self.dimensionless = dimensionless
        self.unit_1d = r'%s$^2$ Mpc$^3$'%self.unit
        
        try:
            self.ps1d
        except:
            k_1d = self.k(n_k)
            k_1d_grid = self.k_grid
            if eor_window:
                try:
                    self.eor_filter
                except:
                    self.eor_window(fov)
                k_1d_grid[~self.eor_filter] = np.inf
                
            k_bin_index = np.digitize(k_1d_grid,self.k_bin_edge) - 1
    
            ps1d_k = []
            ps1d_data = []
            ps1d_err = []
            for i, k in enumerate(k_1d):
                iz, iy, ix = np.where(i==k_bin_index)
                grid_value = self.ps3d[iz,iy,ix]
                n_grid = len(grid_value)
                
                if n_grid > 0:
                    if self.psf:
                        grid_weight = self.power_weight[iy,ix]
                    else:
                        grid_weight = None
                        
                    ps1d_value = np.average(grid_value,weights=grid_weight)
                    ps1d_error = np.sqrt(np.average(np.full(n_grid,ps1d_value)**2,weights=grid_weight)/n_grid)
                    
                    ps1d_k.append(k)
                    ps1d_data.append(ps1d_value)
                    ps1d_err.append(ps1d_error)
    
            ps1d = np.c_[ps1d_k,ps1d_data,ps1d_err]
            self.ps1d = ps1d
            
        print('>>> 1D power spectrum completed')

        ### dimensionless power
        norm_factor = self.ps1d[:,0]**3/(2*np.pi**2) # k^3 / (2pi^2)
        self.ps1d_delta = np.zeros_like(self.ps1d)
        self.ps1d_delta[:,0] = self.ps1d[:,0]
        self.ps1d_delta[:,1] = norm_factor*self.ps1d[:,1]
        self.ps1d_delta[:,2] = norm_factor*self.ps1d[:,2]
        if dimensionless:
            self.unit_1d = r'%s$^2$'%self.unit
            return self.ps1d_delta
        
        return self.ps1d
            
    def plot_2d(self,plot_err=False,ps2d_norm='log',ps2d_err_norm='log',
                eor_window=False,fov=None,
                figsize=(16,5),vmin=None,vmax=None,cmap='jet',title=None):
        if title == None:
            title = r'Cylindrically averaged power spectrum'

        x = self.k_perp
        y = self.k_los
        ps2d_mean = self.ps2d[0]
        
        fig = plt.figure(figsize=figsize)
        
        ax = fig.add_subplot(121)
        mappable = ax.pcolormesh(x,y,ps2d_mean,
                                 norm=ps2d_norm,
                                 vmin=vmin,vmax=vmax,cmap=cmap)
        if eor_window:
            try:
                self.eor_filter
            except:
                self.eor_window(fov)

            ax.plot(x,self.k_los_wedge,c='k',ls='--')
            
        ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),
               xscale='log',yscale='log',
               xlabel='$k_{\perp}$ [Mpc$^{-1}$]',ylabel='$k_{\parallel}$ [Mpc$^{-1}$]',
               title=title)
        cb = ax.figure.colorbar(mappable)
        cb.ax.set_xlabel('$P_k$ [%s$^2$ Mpc$^3$]'%self.unit)
        
        if plot_err:
            ps2d_err = self.ps2d[1]
            
            ax_err = fig.add_subplot(122)
            mappable = ax_err.pcolormesh(x,y,ps2d_err,
                                         norm=ps2d_err_norm,
                                         vmin=vmin,vmax=vmax,cmap=cmap)
            ax_err.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),
                       xscale='log',yscale='log',
                       xlabel='$k_{\perp}$ [Mpc$^{-1}$]',ylabel='$k_{\parallel}$ [Mpc$^{-1}$]',
                       title='2D error power spectrum')
            cb = ax_err.figure.colorbar(mappable)
            cb.ax.set_xlabel('Error [%s$^2$ Mpc$^3$]'%self.unit)

        self.fig_2d = fig
        
    def plot_1d(self,plot_err=True,figsize=(8,5),title=None):
        if title == None:
            title = r'Spherically averaged power spectrum'
        
        if self.dimensionless:
            x = self.ps1d_delta[:,0]
            y = self.ps1d_delta[:,1]
            err = self.ps1d_delta[:,2]
            ylabel = r'$\Delta^2(k)$ [%s$^2$]'%self.unit
        else:
            x = self.ps1d[:,0]
            y = self.ps1d[:,1]
            err = self.ps1d[:,2]
            ylabel = r'$P(k)$ [%s$^2$ Mpc$^3$]'%self.unit
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if plot_err:
            ax.errorbar(x,y,err,marker='o')
        else:
            ax.plot(x,y,marker='o')
            
        ax.set(xscale='log',yscale='log',
               xlabel='$k$ [Mpc$^{-1}$]',ylabel=ylabel,
               title=title)
               
        self.fig_1d = fig
        
    def save_3d(self,outfile,overwrite=False):
        self.header['BUNIT'] = '%s^2 Mpc^3" % self.unit'
        fits_file = fits.PrimaryHDU(data=self.ps3d,header=self.header)
        fits_file.writeto(outfile+'.fits',overwrite=overwrite)
        print('>>> PS3D fits file saved')
            
    def save_2d(self,outfile,save_fits=False,save_eor=False,
                save_plot=False,plot_format='pdf',dpi='figure',
                overwrite=False):
        if save_fits:
            # modify header
            self.header['BUNIT'] = '%s^2 Mpc^3" % self.unit'
            
            self.header.append(('KPERPMIN',self.k_perp.min(),'Minimum k_perp'))
            self.header.append(('KPERPMAX',self.k_perp.max(),'Maximum k_perp'))
            self.header.append(('KPERPDEL',self.dk_perp,'Interval size of k_perp'))
            
            self.header.append(('KPARAMIN',self.k_los.min(),'Minimum k_par'))
            self.header.append(('KPARAMAX',self.k_los.max(),'Maximum k_par'))
            self.header.append(('KPARADEL',self.dk_los,'Interval size of k_par'))
        
            fits_file = fits.PrimaryHDU(data=self.ps2d,header=self.header)
            fits_file.writeto(outfile+'.fits',overwrite=overwrite)
            print('>>> PS2D fits file saved')

        if save_eor:
            eor_window_fits = fits.PrimaryHDU(data=self.eor_filter,header=self.header)
            eor_window_fits(outfile+'_eor_window_filter.fits',overwrite=overwrite)
            print('>>> EoR window filter saved')

        if save_plot:
            self.fig_2d.savefig(outfile+'.'+plot_format,dpi=dpi,bbox_inches='tight')
            print('>>> PS2D plot saved')
            
    def save_1d(self,outfile,save_txt=False,
                save_plot=False,plot_format='pdf',dpi='figure',
                overwrite=False):
        if save_txt:
            if self.dimensionless:
                unit = '%s^2'%self.unit
                data = self.ps1d_delta
            else:
                unit = '%s^2'%self.unit + ' Mpc^3'
                data = self.ps1d
            np.savetxt(outfile+'.txt',data,header='k [1/Mpc], Pk [{unit}], error'.format(unit=unit))

        if save_plot:
            self.fig_1d.savefig(outfile+'.'+plot_format,dpi=dpi,bbox_inches='tight')
            print('>>> PS1D plot saved')
