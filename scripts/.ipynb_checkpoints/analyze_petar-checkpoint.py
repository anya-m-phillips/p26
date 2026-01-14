import petar
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import astropy.units as u
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.table import Table
from scipy.stats import binned_statistic_2d

import galpy
from galpy.potential import MWPotential2014
from galpy.orbit import Orbit

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors



class analyze_petar():
    """
    i am a world astronomy champion. the best astronomer in the world
    """
    
    def __init__(self, path, file_n, add_CM=False, r_criterion=10000, plot_rs=True, read_sb=True):
        """
        path = string path to directory with output files
        file_n = an integer
        
        """
        self.read_sb=read_sb
        
        self.file_n = file_n
        
        file = path
        
        # snapshot file with all particles:
        file = path+"data."+str(file_n)
        particle = petar.Particle(interrupt_mode='bse', external_mode='galpy')
        particle.loadtxt(file, skiprows=1)
        particle.calcR2()
        self.particles = particle
        
        r = np.sqrt(particle.r2)
        if plot_rs==True:
            plt.hist(r[r<r_criterion])
            plt.title("all particles")
            plt.show()
        sel = (r<r_criterion)
        self.sel = sel

        # snapshot for only single stars
        if read_sb==True:
            single_file = path+"data."+str(file_n)+".single"
            single_particle = petar.Particle(interrupt_mode="bse",
                                             external_mode='galpy')
            single_particle.loadtxt(single_file, skiprows=1)
            single_particle.calcR2()
            self.single_particles = single_particle

            single_r = np.sqrt(single_particle.r2)

            # if plot_rs==True:
            #     plt.hist(single_r[single_r<r_criterion])
            #     plt.title("single stars")
            #     plt.show()
            sel_single = (single_r<r_criterion)
            self.sel_single = sel_single


            # shapshot file for binaries
            binary_file = path+"data."+str(file_n)+".binary"
            binary_particle = petar.Binary(member_particle_type=petar.Particle,
                                           G=petar.G_MSUN_PC_MYR,
                                           interrupt_mode="bse",
                                           external_mode="galpy")
            binary_particle.loadtxt(binary_file, skiprows=1)
            binary_particle.calcR2()
            self.binary_particles = binary_particle

            binary_r = np.sqrt(binary_particle.r2)
            # if plot_rs==True:
            #     plt.hist(binary_r[binary_r<r_criterion])
            #     plt.title("binaries")
            #     plt.show()
            sel_binary = (binary_r<r_criterion)
            self.sel_binary = sel_binary

        header = petar.PeTarDataHeader(path+'data.'+str(file_n), external_mode="galpy")
        CM_pos = header.pos_offset
        CM_vel = header.vel_offset

        self.CM_pos = CM_pos
        self.CM_vel = CM_vel
        
        if add_CM == True:
            pos = particle.pos + CM_pos
            self.pos = pos[sel]
            vel = particle.vel + CM_vel
            self.vel = vel[sel]

            if read_sb==True:
                
                core_data = petar.Core(interrupt_mode="bse", external_mode="galpy")
                core_data.loadtxt(path+"data.core")
                
                core_pos = core_data.pos[self.file_n]
                self.core_pos = core_pos
                
                core_vel = core_data.vel[self.file_n]
                self.core_vel = core_vel
                
                single_pos = single_particle.pos + core_pos
                self.single_pos = single_pos[sel_single]
                single_vel = (core_vel + single_particle.vel) #* (u.pc/u.Myr)
                self.single_vel = single_vel[sel_single]

                binary_pos = binary_particle.pos + core_pos
                self.binary_pos = binary_pos[sel_binary]
                binary_vel = (core_vel + binary_particle.vel) #* (u.pc/u.Myr)
                self.binary_vel = binary_vel[sel_binary]
                
                self.sb_pos = np.vstack([self.single_pos, self.binary_pos])
                self.sb_vel = np.vstack([self.single_vel, self.binary_vel])

        if add_CM==False:
            
            print("pls just add CM -- don't have handling for SB files without it")
            
            pos=particle.pos 
            self.pos = pos[sel]
            vel = particle.vel 
            self.vel = vel[sel]

            if read_sb==True:
                single_pos = single_particle.pos
                self.single_pos = single_pos[sel_single]
                single_vel = single_particle.vel 
                self.single_vel = single_vel[sel_single]

                binary_pos = binary_particle.pos
                self.binary_pos = binary_pos[sel_binary]
                binary_vel = binary_particle.vel
                self.binary_vel = binary_vel[sel_binary]
                
                self.sb_pos = np.vstack([self.single_pos, self.binary_pos])
                self.sb_vel = np.vstack([self.single_vel, self.binary_vel])
            
            
            
    def calculate_half_mass_radius(self):
        
        radii = np.sqrt(self.particles.r2)
        masses = self.particles.mass
        
        sorted_indices = np.argsort(radii)
        sorted_radii = np.array(radii)[sorted_indices]
        sorted_masses = np.array(masses)[sorted_indices]

        # Compute cumulative mass
        cumulative_mass = np.cumsum(sorted_masses)

        # Total mass
        total_mass = np.sum(masses)

        # Find the half-mass radius
        half_mass = total_mass / 2
        half_mass_radius_index = np.searchsorted(cumulative_mass, half_mass)
        half_mass_radius = sorted_radii[half_mass_radius_index]

        return half_mass_radius
    
    
    def plot_positions(self, plot_single=False, plot_binary=False,
                       aspect=None, save=False, savepath=None):
        fig, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=[17,4])
        ax1.scatter(self.pos[:,0], self.pos[:,1], s=.01, c='k', label="all particles")
        ax2.scatter(self.pos[:,0], self.pos[:,2], s=.01, c='k', label="all particles")
        ax3.scatter(self.pos[:,1], self.pos[:,2], s=.01, c='k', label="all particles")        
        
        if plot_single==True:
            if self.read_sb==False:
                return "no single/binary star files"
            ax1.scatter(self.single_pos[:,0], self.single_pos[:,1], s=.01, c='cornflowerblue', label="single stars")
            ax2.scatter(self.single_pos[:,0], self.single_pos[:,2], s=.01, c='cornflowerblue', label="single stars")
            ax3.scatter(self.single_pos[:,1], self.single_pos[:,2], s=.01, c='cornflowerblue', label="single stars")
            
        if plot_binary==True:
            if self.read_sb==False:
                return "no single/binary star files"
            ax1.scatter(self.binary_pos[:,0], self.binary_pos[:,1], s=.01, c='lightcoral', label="binary stars")
            ax2.scatter(self.binary_pos[:,0], self.binary_pos[:,2], s=.01, c='lightcoral', label="binary stars")
            ax3.scatter(self.binary_pos[:,1], self.binary_pos[:,2], s=.01, c='lightcoral', label="binary stars")
            
        ax1.set_xlabel("x, pc")
        ax1.set_ylabel("y, pc")
        
        ax2.set_xlabel("x, pc")
        ax2.set_ylabel("z, pc")
        
        ax3.set_xlabel("y, pc")
        ax3.set_ylabel("z, pc")
        
        ax3.legend(markerscale=100)
        
        fig.suptitle(str(self.file_n)+" Myr")

        if aspect=='equal':
            for ax in [ax1,ax2,ax3]:
                ax.set_aspect("equal")
                
        if save==True:
            if savepath==None:
                return "specify a save path"
            else:
                plt.savefig(savepath, bbox_inches="tight")
    
    
    def prog_position(self, init_displacement):
        """
        integrate the progenitor orbit up to this point
        init displacement should be in kpc, m/s
        """
        mp = MWPotential2014
        self.x0, self.y0, self.z0, self.vx0, self.vy0, self.vz0 = init_displacement
        o=Orbit(
            [self.x0*u.kpc, self.vx0*u.km/u.s, self.vy0*u.km/u.s, self.z0*u.kpc, self.vz0*u.km/u.s, 0*u.deg],
            ro=self.x0, vo=self.vy0) 
        if self.y0!=0:
            print("check orbit initialization in analyze_petar.py view_orbit(). y0!=0. ")

        ts = np.arange(0,self.file_n+1,1)*u.Myr
        o.integrate(ts, mp,method='odeint')
        
        
        return o.x((self.file_n)*(u.Myr)), o.y((self.file_n)*(u.Myr)), o.z((self.file_n)*(u.Myr)), o.vx((self.file_n)*(u.Myr)), o.vy((self.file_n)*(u.Myr)), o.vz((self.file_n)*(u.Myr)) # return last potition (& velocity i guess)?
        
    
                
    def view_orbit(self, init_displacement, view_sb=False, view_cyl=False, orb_index=-100, save=False, savepath=None):
        """
        initial displacement should be in the form of 
        [x0, y0, z0, vx0, vy0, vz0]
        where positins are in kpc and velocities are in km/s
        and also note that y0 must be zero in this implementation.
        """
        
        mp = MWPotential2014
        self.x0, self.y0, self.z0, self.vx0, self.vy0, self.vz0 = init_displacement
        o=Orbit(
            [self.x0*u.kpc, self.vx0*u.km/u.s, self.vy0*u.km/u.s, self.z0*u.kpc, self.vz0*u.km/u.s, 0*u.deg],
            ro=self.x0, vo=self.vy0) 
        if self.y0!=0:
            print("check orbit initialization in analyze_petar.py view_orbit(). y0!=0. ")

        ts = np.arange(0,self.file_n,1)*u.Myr
        o.integrate(ts, mp,method='odeint')
        if view_cyl==True:
            o.plot()

        fig, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=[17,5])
        
        
        ax1.plot(o.x(ts)[orb_index:], o.y(ts)[orb_index:], lw=2, c='r', label='integrated progenitor orbit')
        ax2.plot(o.x(ts)[orb_index:], o.z(ts)[orb_index:], lw=2, c='r')
        ax3.plot(o.y(ts)[orb_index:], o.z(ts)[orb_index:], lw=2, c='r')


        ax1.scatter(self.pos[:,0]/1000, self.pos[:,1]/1000, s=.01, c='k', label="from data.# files")
        ax1.set_xlabel("x, kpc")
        ax1.set_ylabel("y, kpc")

        ax2.scatter(self.pos[:,0]/1000, self.pos[:,2]/1000, s=.01, c='k')
        ax2.set_xlabel("x, kpc")
        ax2.set_ylabel("z, kpc")

        ax3.scatter(self.pos[:,1]/1000, self.pos[:,2]/1000, s=.01, c='k')
        ax3.set_xlabel("y, kpc")
        ax3.set_ylabel("z, kpc")
        
        if view_sb==True:
            ax1.scatter(self.sb_pos[:,0]/1000, self.sb_pos[:,1]/1000, s=.01, c='cornflowerblue', label="from single/binary files")
            ax2.scatter(self.sb_pos[:,0]/1000, self.sb_pos[:,2]/1000, s=.01, c='cornflowerblue')
            ax3.scatter(self.sb_pos[:,1]/1000, self.sb_pos[:,2]/1000, s=.01, c='cornflowerblue')
            
        ax1.legend(loc='lower right', markerscale=10)
        
        if save==True:
            plt.savefig(savepath)
        
    def compute_eccentricity(self, integration_time=10000, showplot=False):
        """
        integration time in Myr
        """
        mp = MWPotential2014
        o=Orbit(
            [self.x0*u.kpc, self.vx0*u.km/u.s, self.vy0*u.km/u.s, self.z0*u.kpc, self.vz0*u.km/u.s, 0*u.deg],
            ro=self.x0, vo=self.vy0) 
        
        ts = np.arange(0,integration_time,1)*u.Myr
        o.integrate(ts, mp, method='odeint')

        if showplot==True:
            o.plot()

            fig, [ax2,ax3,ax4] = plt.subplots(1,3,figsize=[17,5])
            ax2.plot(o.x(ts), o.y(ts), lw=.1, c='k')
            ax3.plot(o.x(ts), o.z(ts), lw=.1, c='k')
            ax4.plot(o.y(ts), o.z(ts), lw=.1, c='k')

        r_peri = min(o.r(ts))
        r_apo = max(o.r(ts))

        def compute_e(r_peri, r_apo):
            num = r_apo - r_peri
            denom = r_apo + r_peri
            return num/denom

        e = compute_e(r_peri, r_apo)
        self.ecc = e
        return e
    
    def plot_binary_properties(self, semi_units='au', save=False, savepath=None):
        """
        semi_units can be 'au' or 'pc'
        """
        fig = plt.figure(figsize=(4, 4),dpi=150) 
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3]) 

        axexy = fig.add_subplot(gs[1,0])
        axehx = fig.add_subplot(gs[0,0],sharex=axexy)
        axehy = fig.add_subplot(gs[1,1],sharey=axexy)
        axehx.get_xaxis().set_visible(False)
        axehy.get_yaxis().set_visible(False)
        plt.subplots_adjust(hspace=0.02,wspace=0.02)

        # xlim=[0.1,1e4]
        ylim=[0,0.999]
        # axexy.set_xlim(*xlim)
        # axexy.set_ylim(*ylim)
        axehx.set_xscale('log')
        # xlabel set moved down
        axexy.set_ylabel(r'eccentricity',fontsize=12)

        nbinx=30
        nbiny=30
        # log_xbins=np.logspace(np.log10(xlim[0]),np.log10(xlim[1]), nbinx)
        ybins=np.linspace(ylim[0], ylim[1], nbiny)

        if semi_units=='au':
            xlim=[0.1,1e4]
            log_xbins=np.logspace(np.log10(xlim[0]),np.log10(xlim[1]), nbinx)
            x = self.binary_particles.semi[self.sel_binary]*206265
            axexy.set_xlabel(r'Semi-major axes [AU]',fontsize=12)

        if semi_units=='pc':
            xlim=[1e-8, 1e-1]
            log_xbins=np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), nbinx)
            x = self.binary_particles.semi[self.sel_binary]
            axexy.set_xlabel(r"Semi-major axes [pc]", fontsize=12)


        y = self.binary_particles.ecc[self.sel_binary]
        m1 = self.binary_particles.p1.mass[self.sel_binary]
        m2 = self.binary_particles.p2.mass[self.sel_binary]
        mass_ratio = np.minimum(m1,m2)/np.maximum(m1,m2)

        axehx.hist(x, bins=log_xbins,histtype='step',color='grey',
                   weights=np.ones(x.size)/x.size,label='Initial')
        axehy.hist(y, bins=ybins,histtype='step',orientation="horizontal",color='grey',
                   weights=np.ones(x.size)/x.size)
        pt = axexy.scatter(x ,y, s=self.binary_particles.mass[self.sel_binary], c=mass_ratio)
        cbaxes = fig.add_axes([0.95,0.15,0.02,0.7]) 
        cb = plt.colorbar(pt, cax = cbaxes)  
        cb.set_label('mass ratio')

        fig.suptitle(str(self.file_n)+" Myr")

        
        if save==True:
            if savepath==None:
                return "specify a save path"
            else:
                plt.savefig(savepath, bbox_inches="tight")
        
        
    def plot_binned_binary(self, nbins=101, save=True, savepath=None):
        """
        """

        # compute binned statistics:
        n = nbins
        xmin = min(self.sb_pos[:,0])
        xmax = max(self.sb_pos[:,0])

        ymin = min(self.sb_pos[:,1])
        ymax = max(self.sb_pos[:,1])

        zmin = min(self.sb_pos[:,2])
        zmax = max(self.sb_pos[:,2])

        bins_xy = [
            np.linspace(xmin, xmax, n),
            np.linspace(ymin, ymax, n)
        ]

        bins_xz = [
            np.linspace(xmin, xmax, n),
            np.linspace(zmin, zmax, n)
        ]

        bins_yz = [
            np.linspace(ymin, ymax, n),
            np.linspace(zmin, zmax, n)
        ]


        binary_info = np.hstack([self.binary_pos.copy(), np.ones((len(self.binary_pos), 1))])
        single_info = np.hstack([self.single_pos.copy(), np.zeros((len(self.single_pos),1))])

        all_info = np.vstack([binary_info, single_info])

        x_all = all_info[:,0]
        y_all = all_info[:,1]
        z_all = all_info[:,2]

        values_all = all_info[:,0]
        is_binary = all_info[:,3]

        counts_all_xy, xedges_all_xy, yedges_all_xy, binnumbers_all_xy = binned_statistic_2d(x_all, y_all, bins=bins_xy,
                                                                                             values=values_all,
                                                                                             statistic='count')

        counts_all_xz, xedges_all_xz, yedges_all_xz, binnumbers_all_xz = binned_statistic_2d(x_all, z_all, bins=bins_xz,
                                                                                             values=values_all,
                                                                                             statistic='count')

        counts_all_yz, xedges_all_yz, yedges_all_yz, binnumbers_all_yz = binned_statistic_2d(y_all, z_all, bins=bins_yz,
                                                                                             values=values_all,
                                                                                             statistic='count')
        
        # computing weights:
        weights_xy = np.zeros(len(x_all)) # will go through and replace these with 1/count for all binaries
        weights_xz = np.zeros(len(x_all))
        weights_yz = np.zeros(len(x_all))


        # give each binary a 
        for i in range(len(weights_xy)):
            if is_binary[i]==1:
                binnum_xy = binnumbers_all_xy[i]
                count_xy = len(binnumbers_all_xy[binnumbers_all_xy==binnum_xy])
                if count_xy != 0:
                    weights_xy[i] = 1/count_xy # or maybe 2/count idk 

                binnum_xz = binnumbers_all_xz[i]
                count_xz = len(binnumbers_all_xz[binnumbers_all_xz==binnum_xz])
                if count_xz!=0:
                    weights_xz[i] = 1/count_xz

                binnum_yz = binnumbers_all_yz[i]
                count_yz = len(binnumbers_all_yz[binnumbers_all_yz==binnum_yz])
                if count_yz!=0:
                    weights_yz[i] = 1/count_yz  
                    
                    
        # make plots
        fig, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=[18,5])

        obj_xy = ax1.hist2d(x_all, y_all, bins=bins_xy, weights=weights_xy, cmap='magma', vmin=0, vmax=1)
        # fig.colorbar(obj_xy[3], ax=ax1, label="binary fraction")

        obj_xz = ax2.hist2d(x_all, z_all, bins=bins_xz, weights=weights_xz, cmap='magma', vmin=0, vmax=1)
        # fig.colorbar(obj_xz[3], ax=ax2, label="binary fraction")

        obj_yz = ax3.hist2d(y_all, z_all, bins=bins_yz, weights=weights_yz, cmap='magma', vmin=0, vmax=1)
        # fig.colorbar(obj_yz[3], ax=ax3, label="binary fraction")
        divider=make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        colorbar = fig.colorbar(obj_yz[3], cax=cax, orientation="vertical", label="binary fraction")

        ax1.set_xlabel("x [pc]")
        ax1.set_ylabel("y [pc]")

        ax2.set_xlabel("x [pc]")
        ax2.set_ylabel("z [pc]")

        ax3.set_xlabel("y [pc]")
        ax3.set_ylabel("z [pc]")

        plt.suptitle(str(self.file_n)+" Myr", fontsize=20)

        # for ax in [ax1,ax2,ax3]:
        #     ax.set_aspect("equal")  
        
        if save==True:
            if savepath==None:
                return "enter a save path"
            
            else:
                plt.savefig(savepath)
        
        
    def total_binary_fraction(self):
        return 2*len(self.binary_pos)/len(self.pos)
    
    
    def plot_position_density_hists(self, nbins, save=False, savepath=None):
        # compute binned statistics:
        n = nbins
        xmin = min(self.sb_pos[:,0])
        xmax = max(self.sb_pos[:,0])

        ymin = min(self.sb_pos[:,1])
        ymax = max(self.sb_pos[:,1])

        zmin = min(self.sb_pos[:,2])
        zmax = max(self.sb_pos[:,2])

        bins_xy = [
            np.linspace(xmin, xmax, n),
            np.linspace(ymin, ymax, n)
        ]

        bins_xz = [
            np.linspace(xmin, xmax, n),
            np.linspace(zmin, zmax, n)
        ]

        bins_yz = [
            np.linspace(ymin, ymax, n),
            np.linspace(zmin, zmax, n)
        ]

        binary_info = np.hstack([self.binary_pos.copy(), np.ones((len(self.binary_pos), 1))])
        single_info = np.hstack([self.single_pos.copy(), np.zeros((len(self.single_pos),1))])


        x_binary = binary_info[:,0]
        y_binary = binary_info[:,1]
        z_binary = binary_info[:,2]

        x_single = single_info[:,0]
        y_single = single_info[:,1]
        z_single = single_info[:,2]



        norm = mcolors.LogNorm(vmin=1/5000, vmax=1) # customize vmin to 1/(number of single/binaries you started with)
        fig, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2,3,figsize=[18,10])


        counts_binary_xy, x_edges, y_edges, binnumbers_binary = binned_statistic_2d(
            x_binary, y_binary, x_binary, statistic='count', bins=bins_xy)

        frac_counts_bin = counts_binary_xy / np.sum(counts_binary_xy)

        counts_single_xy, x_edges, y_edges, binnumbers_single = binned_statistic_2d(
            x_single, y_single, x_single, statistic='count', bins=bins_xy)
        frac_counts_sin = counts_single_xy / np.sum(counts_single_xy)

        X_xy, Y_xy = np.meshgrid(x_edges, y_edges)

        ob1 = ax1.pcolormesh(X_xy, Y_xy, frac_counts_sin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        fig.colorbar(ob1, ax=ax1, label='fraction in bin', pad=0)
        ax1.set_title("single stars")

        ob4= ax4.pcolormesh(X_xy, Y_xy, frac_counts_bin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        fig.colorbar(ob4, ax=ax4, label='fraction in bin', pad=0)
        ax4.set_title("binaries")


        # define bins for xz, plot ax2/4
        counts_binary_xz, x_edges, z_edges, binnumbers_binary = binned_statistic_2d(
            x_binary, z_binary, x_binary, statistic='count', bins=bins_xz)
        frac_counts_bin = counts_binary_xz/np.sum(counts_binary_xz)

        counts_single_xz, x_edges, z_edges, binnumbers_single = binned_statistic_2d(
            x_single, z_single, x_single, statistic='count', bins=bins_xz)
        frac_counts_sin = counts_single_xz/np.sum(counts_single_xz)


        X_xz, Z_xz = np.meshgrid(x_edges, z_edges)
        ob2 = ax2.pcolormesh(X_xz, Z_xz, frac_counts_sin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        fig.colorbar(ob2, ax=ax2, label='fraction in bin', pad=0)
        ax2.set_title("single stars")

        ob5= ax5.pcolormesh(X_xz, Z_xz, frac_counts_bin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        fig.colorbar(ob5, ax=ax5, label='fraction in bin', pad=0)
        ax5.set_title("binaries")

        # define bins for yz, plot for ax3/6
        counts_binary_yz, y_edges, z_edges, binnumbers_binary = binned_statistic_2d(
            y_binary, z_binary, y_binary, statistic='count', bins=bins_yz)
        frac_counts_bin = counts_binary_yz/np.sum(counts_binary_yz)

        counts_single_yz, y_edges, z_edges, binnumbers_single = binned_statistic_2d(
            y_single, z_single, y_single, statistic='count', bins=bins_yz)
        frac_counts_sin = counts_single_yz/np.sum(counts_single_yz)


        Y_yz, Z_yz = np.meshgrid(y_edges, z_edges)
        ob3 = ax3.pcolormesh(Y_yz, Z_yz, frac_counts_sin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        fig.colorbar(ob3, ax=ax3, label='fraction in bin', pad=0)
        ax3.set_title("single stars")

        ob6= ax6.pcolormesh(Y_yz, Z_yz, frac_counts_bin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        fig.colorbar(ob6, ax=ax6, label='fraction in bin', pad=0)
        ax6.set_title("binaries")


        fig.suptitle("%i Myr"%self.file_n)
        
        if save==True:
            if savepath==None:
                return "enter a save path"
            else:
                plt.savefig(savepath)
                
                
    def plot_pos_scatter_and_density(self, nbins, save=False, savepath=None):
        # compute binned statistics:
        n = nbins
        xmin = min(self.sb_pos[:,0])
        xmax = max(self.sb_pos[:,0])

        ymin = min(self.sb_pos[:,1])
        ymax = max(self.sb_pos[:,1])

        zmin = min(self.sb_pos[:,2])
        zmax = max(self.sb_pos[:,2])

        bins_xy = [
            np.linspace(xmin, xmax, n),
            np.linspace(ymin, ymax, n)
        ]

        bins_xz = [
            np.linspace(xmin, xmax, n),
            np.linspace(zmin, zmax, n)
        ]

        bins_yz = [
            np.linspace(ymin, ymax, n),
            np.linspace(zmin, zmax, n)
        ]

        binary_info = np.hstack([self.binary_pos.copy(), np.ones((len(self.binary_pos), 1))])
        single_info = np.hstack([self.single_pos.copy(), np.zeros((len(self.single_pos),1))])


        x_binary = binary_info[:,0]
        y_binary = binary_info[:,1]
        z_binary = binary_info[:,2]

        x_single = single_info[:,0]
        y_single = single_info[:,1]
        z_single = single_info[:,2]



        norm = mcolors.LogNorm(vmin=1/5000, vmax=1) # customize vmin to 1/(number of single/binaries you started with)
        fig, [[ax0,ax00,ax000], [ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(3,3,figsize=[18,15])

        for ax in [ax0, ax1, ax4]:
            ax.set_xlabel('x [pc]')
            ax.set_ylabel('y [pc]')
        for ax in [ax00,ax2,ax5]:
            ax.set_xlabel('x [pc]')
            ax.set_ylabel('z [pc]')
        for ax in [ax000, ax3, ax6]:
            ax.set_xlabel('y [pc]')
            ax.set_ylabel('z [pc]')

        ax0.scatter(x_single, y_single, c='k', s=.01, rasterized=True)
        ax0.scatter(x_binary, y_binary, c='k', s=.01, rasterized=True)
        ax00.scatter(x_single, z_single, c='k', s=.01, rasterized=True)
        ax00.scatter(x_binary, z_binary, c='k', s=.01, rasterized=True)
        ax000.scatter(y_single, z_single, c='k', s=.01, rasterized=True)
        ax000.scatter(y_binary, z_binary, c='k', s=.01, rasterized=True)

        counts_binary_xy, x_edges, y_edges, binnumbers_binary = binned_statistic_2d(
            x_binary, y_binary, x_binary, statistic='count', bins=bins_xy)

        frac_counts_bin = counts_binary_xy / np.sum(counts_binary_xy)

        counts_single_xy, x_edges, y_edges, binnumbers_single = binned_statistic_2d(
            x_single, y_single, x_single, statistic='count', bins=bins_xy)
        frac_counts_sin = counts_single_xy / np.sum(counts_single_xy)

        X_xy, Y_xy = np.meshgrid(x_edges, y_edges)

        ob1 = ax1.pcolormesh(X_xy, Y_xy, frac_counts_sin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        # fig.colorbar(ob1, ax=ax1, label='fraction in bin', pad=0)
        ax1.set_title("single stars")

        ob4= ax4.pcolormesh(X_xy, Y_xy, frac_counts_bin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        # fig.colorbar(ob4, ax=ax4, label='fraction in bin', pad=0)
        ax4.set_title("binaries")


        # define bins for xz, plot ax2/4
        counts_binary_xz, x_edges, z_edges, binnumbers_binary = binned_statistic_2d(
            x_binary, z_binary, x_binary, statistic='count', bins=bins_xz)
        frac_counts_bin = counts_binary_xz/np.sum(counts_binary_xz)

        counts_single_xz, x_edges, z_edges, binnumbers_single = binned_statistic_2d(
            x_single, z_single, x_single, statistic='count', bins=bins_xz)
        frac_counts_sin = counts_single_xz/np.sum(counts_single_xz)


        X_xz, Z_xz = np.meshgrid(x_edges, z_edges)
        ob2 = ax2.pcolormesh(X_xz, Z_xz, frac_counts_sin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        # fig.colorbar(ob2, ax=ax2, label='fraction in bin', pad=0)
        ax2.set_title("single stars")

        ob5= ax5.pcolormesh(X_xz, Z_xz, frac_counts_bin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        # fig.colorbar(ob5, ax=ax5, label='fraction in bin', pad=0)
        ax5.set_title("binaries")

        # define bins for yz, plot for ax3/6
        counts_binary_yz, y_edges, z_edges, binnumbers_binary = binned_statistic_2d(
            y_binary, z_binary, y_binary, statistic='count', bins=bins_yz)
        frac_counts_bin = counts_binary_yz/np.sum(counts_binary_yz)

        counts_single_yz, y_edges, z_edges, binnumbers_single = binned_statistic_2d(
            y_single, z_single, y_single, statistic='count', bins=bins_yz)
        frac_counts_sin = counts_single_yz/np.sum(counts_single_yz)


        Y_yz, Z_yz = np.meshgrid(y_edges, z_edges)
        ob3 = ax3.pcolormesh(Y_yz, Z_yz, frac_counts_sin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        fig.colorbar(ob3, ax=ax3, label='fraction in bin', pad=0)
        ax3.set_title("single stars")

        ob6= ax6.pcolormesh(Y_yz, Z_yz, frac_counts_bin.T, cmap='magma', norm=norm)#, vmin=0, vmax=1)
        fig.colorbar(ob6, ax=ax6, label='fraction in bin', pad=0)
        ax6.set_title("binaries")


        fig.suptitle("%i Myr"%self.file_n)
        
        if save==True:
            if savepath==None:
                return "enter a save path"
            else:
                plt.savefig(savepath)