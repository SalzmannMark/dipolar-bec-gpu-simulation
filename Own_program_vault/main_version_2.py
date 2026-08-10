import numpy as np
import scipy.fft as fft
from vtk import vtkStructuredPointsReader

from vtkmodules.util import numpy_support as VN
import math
import os
import glob
import sys
from pathlib import Path

from InputParser import InputParser 
from scipy.ndimage import gaussian_filter1d

from scipy.interpolate import RegularGridInterpolator

import scipy.integrate as integrate
from scipy.ndimage import map_coordinates
from scipy import integrate

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import colors
from scipy.ndimage import gaussian_filter, maximum_filter, label, find_objects






'''
latex parameters
'''


fontsize = 15
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['legend.handlelength'] = 0.5


pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": ["DejaVu Sans"]
}
plt.rcParams.update(pgf_with_rc_fonts)

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'],
                  'size': fontsize})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')
plt.rc('legend', fontsize=fontsize,
       title_fontsize=fontsize)



class RawDataExtract:

    # Define class constants 
    hbar = 0.6347*10**5 #in µm^2 u/s
    a_0 = 5.29*1e-5 #Bohr radius in µm
    #All extractions are converted to SI Units to ease plotting, reverting to dimensionless units can be done by multiplying the corresponding factors of a_0, a_ho, w_ho 

    def __init__(self, main_path_directory, parameter_file ='quench.prm', TW_run = False):
        
        #Note that path_directory should be given without last '/'
        self.dimension = 0
        self.TW_run = TW_run
        self.directory = main_path_directory
        self.parameter_file = parameter_file
        self.ip = InputParser(self.parameter_file, self.directory)

        
        #Run and grid paraeterms in SI units
        self.r_max, self.number_grid_points, self.grid_spacing, self.r = self.grid_extract()
        

        self.omega = self.harmonical_trap_extract()
        self.omega_ho = self.omega[-1]
        self.edd, self.edd_final, self.initial_scattering_length, self.final_scattering_length, self.dipolar_length, self.number_of_particles, self.atomic_mass = self.gas_params_extract()

        self.trap_type, self.trap_radius, self.trap_height, self.trap_steepness, self.trap_offset = self.cylindrical_trap_extract()

        self.use_cutoff = self.ip.retrieve_int("use_cutoff")
        self.harm_z = self.ip.retrieve_int("omegaz")
        self.shape = self.ip.retrieve_int("shape")
        self.num_realtime_steps = self.ip.retrieve_int("number of real time steps")
        self.time_step = self.ip.retrieve_float("time step")
        self.write_output_every = self.ip.retrieve_int("write output every")



        self.a_ho = np.sqrt(self.hbar/(self.atomic_mass*self.omega_ho))



        #self.number_of_real_time_steps, self.time_step, self.timestep_between_saved_runs, self.ramp_duration = self.time_extract()

    def grid_extract(self):
        
        # Extraction is in SI units of \mu m

        # Note that grid runs from -x_max to x_max-dx
        x_max = self.ip.retrieve_float("xmax")
        y_max = self.ip.retrieve_float("ymax")
        z_max = self.ip.retrieve_float("zmax")
        r_max = (x_max, y_max, z_max)

        nx = self.ip.retrieve_int("nx")
        ny = self.ip.retrieve_int("ny")
        nz = self.ip.retrieve_int("nz")
        number_grid_points = (nx, ny ,nz)

        dx = 2 * x_max / nx
        dy = 2 * y_max / ny
        dz = 2 * z_max / nz
        grid_spacing = (dx,dy,dz)


        x = np.linspace(-x_max, x_max, nx, endpoint = False)
        y = np.linspace(-y_max, y_max, ny, endpoint = False)
        z = np.linspace(-z_max, z_max, nz, endpoint = False)
        r = (x,y,z)
       
        return r_max, number_grid_points, grid_spacing, r

    def harmonical_trap_extract(self):
        # In units of Hz, factor of 2pi needed compared to param file, note that in Santos code omega_ho = omega_z
        omegax = self.ip.retrieve_float("omegax")
        omegay = self.ip.retrieve_float("omegay")
        omegaz = self.ip.retrieve_float("omegaz")
        omega = (2*np.pi*omegax, 2*np.pi*omegay, 2*np.pi*omegaz)
        return omega
    
    def cylindrical_trap_extract(self):
        trap_type = self.ip.retrieve_string("type")
        radius = self.ip.retrieve_float("radius")
        height = self.ip.retrieve_float("height")
        offset = self.ip.retrieve_float("offset")

        steepness = 0
        if trap_type == "x2":
            steepness = 1
        elif trap_type == "x4":
            steepness = 2
        elif trap_type == "x6":
            steepness = 3
        elif trap_type == "x8":
            steepness = 4
        elif trap_type == "x10":
            steepness = 5
        else:
            steepness = 0

        return trap_type, radius, height, steepness, offset
    
    def rectangular_trap_extract(self):
        trap_type = self.ip.retrieve_string("type")
        wall_x1 = self.ip.retrieve_float("wall_x1")
        wall_x2 = self.ip.retrieve_float("wall_x2")
        wall_y1 = self.ip.retrieve_float("wall_y1")
        wall_y2 = self.ip.retrieve_float("wall_y2")
        height = self.ip.retrieve_float("height")
        steepness = self.ip.retrieve_float("steepness")  
        return trap_type, [wall_x1, wall_y1, wall_x2, wall_y2], height, steepness


     
        

    def make_k_mesh(self):

        # k is [0, ..., kmax,-kmax, ..., 0-dk]
        kx = 2*np.pi*fft.fftfreq(self.number_grid_points[0], self.grid_spacing[0])
        ky = 2*np.pi*fft.fftfreq(self.number_grid_points[1], self.grid_spacing[1])
        kz = 2*np.pi*fft.fftfreq(self.number_grid_points[2], self.grid_spacing[2])
        k = (kx, ky, kz)
        dkx = kx[1] - kx[0]
        dky = ky[1] - ky[0]
        dkz = kz[1] - kz[0]
        dk = (dkx, dky, dkz)

        Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')

        return Kx, Ky, Kz, k, dk
    
    def gas_params_extract(self):
        # in units of the Bohr length a_0
        edd = self.ip.retrieve_float("edd")
        edd_final = self.ip.retrieve_float("edd_final")
        #self.ip.retrieve_float("final scattering length")
        dipolar_length            = self.ip.retrieve_float("dipolar_length")
        number_of_particles = self.ip.retrieve_int("number of particles")
        atomic_mass         = self.ip.retrieve_float("atomic mass")

        bohr_radius = 5.292E-5
        dipolar_length *= bohr_radius # convert to micrometer
        if edd != 0:
            Initial_scattering_length = dipolar_length/edd
        else:
            print("Warning: edd is zero, setting initial scattering length to 0")
            Initial_scattering_length = 0
        
        if edd_final != 0:
            final_scattering_length = dipolar_length/edd_final
        else:
            print("Warning: edd_final is zero, setting final scattering length to 0")
            final_scattering_length = 0


        return edd, edd_final, Initial_scattering_length, final_scattering_length, dipolar_length, number_of_particles, atomic_mass

    def time_extract(self):
        
        number_of_real_time_steps = self.ip.retrieve_int("number of real time steps")
        time_step                       = self.ip.retrieve_float("time step")
        ramp_duration = self.ip.retrieve_float("ramp duration")
        write_output_every = self.ip.retrieve_int("write output every")
        timestep_between_saved_runs = write_output_every*time_step

        return number_of_real_time_steps, time_step, timestep_between_saved_runs, ramp_duration

    def angles_extract(self):
        # for now 0 can add later functionality
        theta_mu                  = self.ip.retrieve_float("theta")
        phi_mu                    = self.ip.retrieve_float("phi")

        return theta_mu, phi_mu

    def numerical_params_extract(self):

        number_of_gradient_descent_steps = self.ip.retrieve_int("number of gradient descent steps")
        number_of_gradient_descent_steps_initial = self.ip.retrieve_int("number of gradient descent steps initial")
        residual                         = self.ip.retrieve_float("residual")
        alpha                            = self.ip.retrieve_float("alpha")
        beta                             = self.ip.retrieve_float("beta")

        return number_of_gradient_descent_steps, number_of_gradient_descent_steps_initial, residual, alpha, beta
    
    def dipolar_cut_extract(self):
        cut_x = self.ip.retrieve_float("cut_x")
        cut_y = self.ip.retrieve_float("cut_y")
        cut_z = self.ip.retrieve_float("cut_z")
        cut = (cut_x, cut_y, cut_z)

        return cut

    #     number_of_real_time_steps = ip.retrieve_int("number of real time steps")
    #     time_step                       = ip.retrieve_float("time step")
        
    #     write_output_every=ip.retrieve_int("write output every")
        
    #     nb_runs= ip.retrieve_int("nb_runs")

    #     ramp_duration = ip.retrieve_float("ramp duration")

    #     cut_x = ip.retrieve_float("cut_x")
    #     cut_y = ip.retrieve_float("cut_y")
    #     cut_z = ip.retrieve_float("cut_z")


    def data_extract_vtk_wavefunction(self, file, dimension, scalars_name = ['real_psi', 'imag_psi'], run_iteration = None, run_time = None):
        reader = vtkStructuredPointsReader()
        
        if run_time is not None:
            file_name = file.replace('.vtk', '') + str(run_time) + '.vtk'
        else:
            file_name = file

        if self.TW_run:
            sub_dir = self.directory.rpartition('/')[-1]+'_'+str(run_iteration)
            input_file_name = os.path.join(self.directory, sub_dir, file_name)
        else:
            input_file_name = os.path.join(self.directory, file_name)

        if os.path.exists(input_file_name)==True:
            reader.SetFileName(input_file_name)
            reader.ReadAllVectorsOn()
            reader.ReadAllScalarsOn()
            reader.Update()
            data = reader.GetOutput()
            real_psi = VN.vtk_to_numpy(data.GetPointData().GetArray(scalars_name[0]))
            imag_psi = VN.vtk_to_numpy(data.GetPointData().GetArray(scalars_name[1]))
        
        else: print('No such file exists')

        psi_grid = self.number_grid_points[0:dimension]

        #print(psi_grid.shape)
        real_psi = real_psi.reshape(psi_grid, order='F')*self.a_ho**(-3/2)
        imag_psi = imag_psi.reshape(psi_grid, order='F')*self.a_ho**(-3/2)

        return real_psi, imag_psi
    
    def data_extract_vtk_density(self, file, dimension, scalars_name = ['real_psi'], run_iteration = None, run_time = None):
        reader = vtkStructuredPointsReader()

        if run_time is not None:
            file_name = file.replace('.vtk', '') + str(run_time) + '.vtk'

        if self.TW_run:
            sub_dir = self.directory.rpartition('/')[-1]+'_'+str(run_iteration)
            input_file_name = os.path.join(self.directory, sub_dir, file_name)
        else:
            input_file_name = os.path.join(self.directory, file_name)

        if os.path.exists(input_file_name)==True:
            reader.SetFileName(input_file_name)
            reader.ReadAllVectorsOn()
            reader.ReadAllScalarsOn()
            reader.Update()
            data = reader.GetOutput()
            density = VN.vtk_to_numpy(data.GetPointData().GetArray(scalars_name[0]))
        
        else: print('No such file exists')

        psi_grid = self.number_grid_points[0:dimension]

        density = density.reshape(psi_grid, order='F')*self.a_ho**(-2)

        return density
    




class TestProcessing(RawDataExtract):
    def __init__(self, data=None, rawdata: RawDataExtract = None,
                 dimension: int | None = None, **kwargs):
        if rawdata is None:
            super().__init__(**kwargs)
            self.RawDataExtract_instance = self

        else:
            self.RawDataExtract_instance = rawdata

            # do NOT call super().__init__, avoid re-parsing

        if dimension is not None:
            self.dimension = int(dimension)
        elif rawdata is not None:
            self.dimension = int(getattr(rawdata, "dimension", 0))
        else:
            self.dimension = 0

            
        self.data = data
        self.iter = kwargs.get('iteration', 0)
        self.file = kwargs.get('file', 'wavefunction.vtk')



    def __getattr__(self, name):
        return getattr(self.RawDataExtract_instance, name)


    def wavefunction_real_space(self):
        if len(self.data) == 2:
            psi = self.data[0] + 1j*self.data[1]
            return psi 
        else:
            return print("This is a density not a wavefunction")
    
    def density_integrated(self):
        if len(self.data) == 1:
            density = self.data[0]
            return density
        else:
            return print("This is not a wavefunction")


    def density_real_space(self):
        if len(self.data) == 2:
            density = np.abs(self.wavefunction_real_space())**2
        elif len(self.data) == 1:
            density = np.abs(self.density_integrated())

        return density

    def phase(self):
        phase = np.angle(self.wavefunction_real_space())
        return(phase)

    def wavefunction_momentum_space(self):
        psi_k = fft.fftshift(fft.fftn(self.wavefunction_real_space()))*self.RawDataExtract_instance.grid_spacing[0]*self.RawDataExtract_instance.grid_spacing[1]*self.RawDataExtract_instance.grid_spacing[2]/(2*np.pi)**(3/2)
        return psi_k

    def density_momentum_space(self):
        density_k = np.abs(self.wavefunction_momentum_space())**2
        return density_k
    
    def print_out_stuff(self):
        realspace_density = self.density_real_space()
        grid_spacing = self.grid_spacing

        bohr_radius = 5.292E-5
        print("initial_scattering_length = {}".format(self.initial_scattering_length/bohr_radius))
        print("final_scattering_length = {}".format(self.final_scattering_length/bohr_radius))
        print("dipolar_length = {}".format(self.dipolar_length/bohr_radius))
        print("number_of_particles = {}".format(self.number_of_particles))
        print("atomic_mass = {}".format(self.atomic_mass))

        print("Particles after simulation: ",np.sum(realspace_density)*np.prod(grid_spacing))

    @classmethod
    def Two_D(cls, file, dimension=2, iteration=0, scalars=['real_psi','imag_psi']):
        main_directory = str(Path(file).parent)

        rde = RawDataExtract(main_directory, parameter_file="individual_parameters.prm")
        psi_real, psi_imag = rde.data_extract_vtk_wavefunction(file, dimension, scalars)
        dimension = dimension

        return cls([psi_real, psi_imag], rawdata=rde, dimension=dimension, iteration=iteration,
                   file=file)
    

    @classmethod
    def Three_D(cls, file, dimension=3, iteration=0, scalars=['real_psi','imag_psi']):
        main_directory = str(Path(file).parent)
        rde = RawDataExtract(main_directory, parameter_file="individual_parameters.prm")
        psi_real, psi_imag = rde.data_extract_vtk_wavefunction(file, dimension, scalars)
        dimension = dimension
        return cls([psi_real, psi_imag], rawdata=rde, dimension=dimension, iteration=iteration,
                   file=file)

    def count_droplets(self, data,
                    neighborhood_size=5,
                    min_distance=10,
                    min_height=None,
                    relative_height=0.1,
                    smoothing_sigma=1.0):
        """
        Improved droplet/peak finder.

        Args:
        data: 2D numpy array (rows=y, cols=x).
        neighborhood_size: kernel size for local-maximum filtering (odd integer recommended).
        min_distance: minimum separation (in pixels) between kept maxima (non-max suppression).
        min_height: absolute threshold for peaks (overrides relative_height if provided).
        relative_height: fraction of the global max used as threshold when min_height is None.
        smoothing_sigma: Gaussian smoothing sigma applied before peak detection (reduces noise).

        Returns:
        list of (row, col) tuples for detected maxima (row = y, col = x).
        """
        # smooth to reduce noise
        sm = gaussian_filter(data, sigma=smoothing_sigma)

        # choose threshold
        if min_height is None:
            min_height = sm.max() * float(relative_height)

        # local maxima mask
        local_max = (sm == maximum_filter(sm, size=neighborhood_size))
        local_max[sm < min_height] = False

        # label and get candidate coords
        labeled, _ = label(local_max)
        slices = find_objects(labeled)

        coords = []
        intensities = []
        for sl in slices:
            if sl is None:
                continue
            # sl[0] -> rows (y), sl[1] -> cols (x)
            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            r = int((r0 + r1) / 2)
            c = int((c0 + c1) / 2)
            coords.append((r, c))
            intensities.append(sm[r, c])

        if len(coords) == 0:
            print("Number of density maxima: 0")
            return []

        coords = np.array(coords)        # shape (N,2) with (row,col)
        intensities = np.array(intensities)

        # non-maximum suppression by min_distance (keep strongest peaks)
        order = np.argsort(intensities)[::-1]  # descending intensity
        keep_mask = np.zeros(len(coords), dtype=bool)
        kept_indices = []
        for idx in order:
            r, c = coords[idx]
            if not kept_indices:
                kept_indices.append(idx)
                keep_mask[idx] = True
                continue
            kept = coords[kept_indices]
            dists = np.sqrt((kept[:, 0] - r) ** 2 + (kept[:, 1] - c) ** 2)
            if np.all(dists >= min_distance):
                kept_indices.append(idx)
                keep_mask[idx] = True

        kept_coords = coords[keep_mask]
        print(f"Number of density maxima: {len(kept_coords)} (candidates before suppression: {len(coords)})")

        # return list of (row, col) to match your existing code's convention
        return [(int(r), int(c)) for r, c in kept_coords]
    
    def PLOT_Two_D_slice(self, x_dot=0, y_dot=0, fontsize=fontsize, r_input= None, draw_lines = False, vmax=None,  save=False, z_slice = None, radial_mask = None, count_droplets = True):

        radius = 0
        if r_input is not None:
            radius = r_input
        else:
            radius = self.trap_radius

        psi_real = self.data[0]
        psi_imag = self.data[1]
        realspace_density = self.density_real_space()

    
        print("dimension = " , self.dimension)
        if radial_mask is not None:
            if self.dimension == 3:
                realspace_density = realspace_density* radial_mask[:,:,None]
            else:
                realspace_density = realspace_density*radial_mask


        X_axis, Y_axis, Z_axis = self.r
        X_idx = np.where(np.abs(X_axis) < radius * 1.2)[0]
        Y_idx = np.where(np.abs(Y_axis) < radius * 1.2)[0]
        
        X_slice = slice(X_idx.min(), X_idx.max() + 1)
        Y_slice = slice(Y_idx.min(), Y_idx.max() + 1)

        if z_slice is not None:
            Plot_slice = np.s_[X_slice, Y_slice, z_slice]
        else:
            Plot_slice = np.s_[X_slice, Y_slice]


        realspace_density = realspace_density[Plot_slice]
        psi_real = psi_real[Plot_slice]
        psi_imag = psi_imag[Plot_slice]

    
        if self.trap_type == "cylinder_hard_wall":
            traptype_name = "cylindrical box trap"
        elif self.trap_type == "x2":
            traptype_name = "harmonic trap"
        elif self.trap_type == "x4":
            traptype_name = r"softened trap $\propto r^4$"
        elif self.trap_type == "x6":
            traptype_name = r"softened trap $\propto r^6$"
        elif self.trap_type == "x8":
            traptype_name = r"softened trap $\propto r^8$"
        elif self.trap_type == "x10":
            traptype_name = r"softened trap $\propto r^{10}$"
        else:
            traptype_name = self.trap_type

        tn= self.iter*self.write_output_every*self.time_step
        edd = self.edd

        if vmax is None:
            vmax = np.max(realspace_density)
        
        exponent = int(np.floor(np.log10(vmax)))
        N_exponent = int(np.floor(np.log10(abs(self.number_of_particles))))
        N_val = self.number_of_particles/10**N_exponent

        


        x_ticks_values = [-radius, 0, radius]
        y_ticks_values = [-radius, 0, radius]
        x_ticks_indices = [np.abs(X_axis[X_idx] - value).argmin() for value in x_ticks_values]
        y_ticks_indices = [np.abs(Y_axis[Y_idx] - value).argmin() for value in y_ticks_values]


        


        density_cmap = plt.get_cmap('viridis')
        phase_cmap = plt.get_cmap('twilight')       
        width = 10
        fig, axes = plt.subplots(1,2, figsize=(width,width*9/15), constrained_layout=True)
        fig.get_layout_engine().set(h_pad=0.06)

        if count_droplets:
            candidates = self.count_droplets(realspace_density,
                            neighborhood_size=45,   # bigger -> merges nearby peaks
                            min_distance=20,        # pixels; increase to avoid multiple per droplet
                            relative_height=0.08,   # lower -> more peaks; raise to be stricter
                            smoothing_sigma=2.0)    # larger -> more smoothing

            # Convert to x,y for plotting (cols = x axis, rows = y axis)
            ys = [p[0] for p in candidates]
            xs = [p[1] for p in candidates]

            axes[0].scatter(xs, ys, s=10, c='r')
            plt.suptitle(rf" {traptype_name}, \quad $\epsilon_{{dd}}$ = {edd:.3f}, \quad number of droplets = {len(candidates)}", fontsize=fontsize+3,ha='center', va="top")
        else:
            plt.suptitle(rf"{traptype_name}, \quad $\epsilon_{{dd}}$ = {edd:.3f}, \quad t = {tn:.2f} ms", fontsize=fontsize+3,ha='center', va="top")
        

        im_density = axes[0].imshow(realspace_density, cmap=density_cmap, vmin=0, vmax=vmax)
        axes[0].set_title(rf"Density, N =${N_val:.2f} \cdot 10^{{{N_exponent}}}$", fontsize=fontsize)
        axes[0].set_xlabel(r'$x$ [\textmu m]', fontsize=fontsize-1)
        axes[0].set_ylabel(r'$y$ [\textmu m]', fontsize=fontsize-1, labelpad=0)

        axes[0].set_xticks(ticks=x_ticks_indices, labels=x_ticks_values)
        axes[0].set_yticks(ticks=y_ticks_indices, labels=y_ticks_values)

        if draw_lines:
            axes[0].axvline(x=x_dot, color='red', linestyle='--')
            axes[0].axhline(y=y_dot, color='red', linestyle='--')

        cbar = fig.colorbar(im_density, ax=axes[0], orientation='horizontal',location = "bottom", pad=0.02, fraction=0.05, ticks =[0.0, vmax/4, vmax/2, vmax*3/4 , vmax] )
        cbar.set_label(label=rf'Density n(x,y,0)($10^{{{exponent}}}$$\mu m^{{-3}}$  )', fontsize=fontsize-1)
        cbar.set_ticklabels([f'0',f'{vmax/4/10**exponent:.2f}', f'{vmax/2/10**exponent:.2f}', f'{vmax*3/4/10**exponent:.2f}', rf'{vmax/10**exponent:.2f}'])


        im_phase = axes[1].imshow(np.angle(psi_real + 1j * psi_imag), cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('Phase', fontsize=fontsize)
        axes[1].set_xlabel(r'$x$ [\textmu m]', fontsize=fontsize-1)


        axes[1].set_xticks(ticks=x_ticks_indices, labels=x_ticks_values)
        axes[1].set_yticks(y_ticks_indices)
        axes[1].set_yticklabels([])



        cbar = fig.colorbar(im_phase, ax=axes[1], orientation='horizontal',location = "bottom", pad=0.02, fraction = 0.05 , ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_label('Phase [rad]', fontsize=fontsize)  
        cbar.set_ticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])

        save_to_name = self.file.split('/')[-2]
        if save:
            os.makedirs("plots", exist_ok=True)
            plt.savefig(f"plots/plot_{self.iter:05d}_{save_to_name}.png", dpi=300, bbox_inches = 'tight', pad_inches = 0.02)
            plt.close()
        else:
            plt.show()

        
        healing_length = lambda a_s, rho: np.sqrt(1/(8*np.pi*a_s*rho))

        
        print("initial healing length at maximal density: ", healing_length(self.initial_scattering_length, np.max(realspace_density)))
        print("initial healing length at minimal density: ", healing_length(self.initial_scattering_length, np.min(realspace_density[np.nonzero(realspace_density)])   ))
        print("final healing length at maximal density: ", healing_length(self.final_scattering_length, np.max(realspace_density)))
        print("final healing length at minimal density: ", healing_length(self.final_scattering_length, np.min(realspace_density[np.nonzero(realspace_density)])   ))
       
     
    def PLOT_summed_up_density(self, x_dot=0, y_dot=0, fontsize=fontsize, r_input= None, draw_lines = False, vmax=None,  save=False, radial_mask = None):

        radius = 0
        if r_input is not None:
            radius = r_input
        else:
            radius = self.trap_radius

        X_axis, Y_axis, Z_axis = self.r

        X_idx = np.where(np.abs(X_axis) < radius * 1.2)[0]
        Y_idx = np.where(np.abs(Y_axis) < radius * 1.2)[0]
        
        X_slice = slice(X_idx.min(), X_idx.max() + 1)
        Y_slice = slice(Y_idx.min(), Y_idx.max() + 1)

        Plot_slice = np.s_[X_slice, Y_slice]


        

        
        psi_real = np.sum( self.data[0]* self.grid_spacing[2], axis=2)
        psi_imag = np.sum( self.data[1]* self.grid_spacing[2], axis=2)


        realspace_density = np.sum(self.density_real_space() * self.grid_spacing[2], axis=2)

        psi_real = psi_real[Plot_slice]
        psi_imag = psi_imag[Plot_slice]
        realspace_density = realspace_density[Plot_slice]


        if radial_mask is not None:
            realspace_density = realspace_density*radial_mask
            psi_real = psi_real*radial_mask
            psi_imag = psi_imag*radial_mask


        tn= self.iter*self.write_output_every*self.time_step
        edd = self.edd

        if vmax is None:
            vmax = np.max(realspace_density)
        
        exponent = int(np.floor(np.log10(vmax)))

        


        x_ticks_values = [-radius, 0, radius]
        y_ticks_values = [-radius, 0, radius]
        x_ticks_indices = [np.abs(X_axis[X_idx] - value).argmin() for value in x_ticks_values]
        y_ticks_indices = [np.abs(Y_axis[Y_idx] - value).argmin() for value in y_ticks_values]



        density_cmap = plt.get_cmap('viridis')
        phase_cmap = plt.get_cmap('twilight')

        
        width = 10
        fig, axes = plt.subplots(1,2, figsize=(width,width*9/15), constrained_layout=True)
        fig.get_layout_engine().set(h_pad=0.06)

        plt.suptitle(rf"Trap radius = {radius:.3f}, \quad $\epsilon_{{dd}}$ = {edd:.3f}, \quad t = {tn:.2f} ms", fontsize=fontsize+3,ha='center', va="top")

        im_density = axes[0].imshow(realspace_density, cmap=density_cmap, vmin=0, vmax=vmax)
        axes[0].set_title('Density', fontsize=fontsize)
        axes[0].set_xlabel(r'$x$ [\textmu m]', fontsize=fontsize-1)
        axes[0].set_ylabel(r'$y$ [\textmu m]', fontsize=fontsize-1, labelpad=0)

        axes[0].set_xticks(ticks=x_ticks_indices, labels=x_ticks_values)
        axes[0].set_yticks(ticks=y_ticks_indices, labels=y_ticks_values)

        if draw_lines:
            axes[0].axvline(x=x_dot, color='red', linestyle='--')
            axes[0].axhline(y=y_dot, color='red', linestyle='--')

        cbar = fig.colorbar(im_density, ax=axes[0], orientation='horizontal',location = "bottom", pad=0.02, fraction=0.05, ticks =[0.0, vmax/4, vmax/2, vmax*3/4 , vmax] )
        cbar.set_label(label=rf'Density n(x,y,0)($10^{{{exponent}}}$$\mu m^{{-3}}$  )', fontsize=fontsize-1)
        cbar.set_ticklabels([f'0',f'{vmax/4/10**exponent:.2f}', f'{vmax/2/10**exponent:.2f}', f'{vmax*3/4/10**exponent:.2f}', rf'{vmax/10**exponent:.2f}'])


        im_phase = axes[1].imshow(np.angle(psi_real + 1j * psi_imag), cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('Phase', fontsize=fontsize)
        axes[1].set_xlabel(r'$x$ [\textmu m]', fontsize=fontsize-1)


        axes[1].set_xticks(ticks=x_ticks_indices, labels=x_ticks_values)
        axes[1].set_yticks(y_ticks_indices)
        axes[1].set_yticklabels([])



        cbar = fig.colorbar(im_phase, ax=axes[1], orientation='horizontal',location = "bottom", pad=0.02, fraction = 0.05 , ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_label('Phase [rad]', fontsize=fontsize)  
        cbar.set_ticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])

        save_to_name = self.file.split('/')[-2]
        if save:
            os.makedirs("plots", exist_ok=True)
            plt.savefig(f"plots/plot_{self.iter:05d}_{save_to_name}.png", dpi=300, bbox_inches = 'tight', pad_inches = 0.02)
            plt.close()
        else:
            plt.show()

        
        healing_length = lambda a_s, rho: np.sqrt(1/(8*np.pi*a_s*rho))

        
        print("initial healing length at maximal density: ", healing_length(self.initial_scattering_length, np.max(realspace_density)))
        print("initial healing length at minimal density: ", healing_length(self.initial_scattering_length, np.min(realspace_density[np.nonzero(realspace_density)])   ))
        print("final healing length at maximal density: ", healing_length(self.final_scattering_length, np.max(realspace_density)))
        print("final healing length at minimal density: ", healing_length(self.final_scattering_length, np.min(realspace_density[np.nonzero(realspace_density)])   ))
       
   
       

    def PLOT_momentum_slice(self, r_input=None, x_dot=0, y_dot=0, vmax = None, save=False, fontsize=fontsize, z_slice=None, radial_mask=None):

        radius = 0
        if r_input is not None:
            radius = r_input
        else:
            radius = self.trap_radius



        Kx, Ky, Kz, k, k_use = self.make_k_mesh()


        middle = int(len(k[0])/2)

        if z_slice is not None:
            Plot_slice = np.s_[middle-radius: middle+radius, middle-radius:middle+radius, 128]
        else:
            Plot_slice = np.s_[middle-radius: middle+radius, middle-radius:middle+radius]



        momentumspace_density = self.density_momentum_space()[Plot_slice]


        if radial_mask is not None:
            momentumspace_density = momentumspace_density*radial_mask

        if vmax is None:
            vmax = np.max(momentumspace_density)

        
        psi_real = self.data[0][Plot_slice]
        psi_imag = self.data[1][Plot_slice]
  

        tn= self.iter*self.write_output_every*self.time_step
        edd = self.edd

        exponent = int(np.floor(np.log10(vmax)))

    
        density_cmap = plt.get_cmap('viridis')
        phase_cmap = plt.get_cmap('twilight')

        
        width = 10
        fig, axes = plt.subplots(1,2, figsize=(width,width*9/15), constrained_layout=True)
        fig.get_layout_engine().set(h_pad=0.06)

        plt.suptitle(rf"Trap radius = {radius:.3f}, \quad $\epsilon_{{dd}}$ = {edd:.3f}, \quad t = {tn:.2f} ms", fontsize=fontsize+3,ha='center', va="top")

        im_density = axes[0].imshow(momentumspace_density, cmap=density_cmap, norm = mcolors.LogNorm())
        axes[0].set_title('Density', fontsize=fontsize)
        axes[0].set_xlabel(r'$x$ [\textmu m]', fontsize=fontsize-1)
        axes[0].set_ylabel(r'$y$ [\textmu m]', fontsize=fontsize-1, labelpad=0)

  
        if True:
            axes[0].axvline(x=x_dot, color='red', linestyle='--')
            axes[0].axhline(y=y_dot, color='red', linestyle='--')

        cbar = fig.colorbar(im_density, ax=axes[0], orientation='horizontal',location = "bottom", pad=0.02, fraction=0.05, ticks =[0.0, vmax/4, vmax/2, vmax*3/4 , vmax] )
        cbar.set_label(label=rf'Density n(x,y,0)($10^{{{exponent}}}$$\mu m^{{-3}}$  )', fontsize=fontsize-1)
     

        im_phase = axes[1].imshow(np.angle(psi_real + 1j * psi_imag), cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('Phase', fontsize=fontsize)
        axes[1].set_xlabel(r'$x$ [\textmu m]', fontsize=fontsize-1)





        cbar = fig.colorbar(im_phase, ax=axes[1], orientation='horizontal',location = "bottom", pad=0.02, fraction = 0.05 , ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_label('Phase [rad]', fontsize=fontsize)  
        cbar.set_ticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])

        save_to_name = self.file.split('/')[-2]
        if save:
            os.makedirs("plots", exist_ok=True)
            plt.savefig(f"plots/plot_{self.iter:05d}_{save_to_name}.png", dpi=300, bbox_inches = 'tight', pad_inches = 0.02)
            plt.close()
        else:
            plt.show()



    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
        
    def cylindrical_mask(self, center=None, radius_1=None, radius_2 = None, angle_1 = -np.pi, delta_angle = 2*np.pi):

        nx = self.number_grid_points[0]
        ny = self.number_grid_points[1]


        if center is None: # use the middle of the image
            center = (int(nx/2), int(ny/2))
        if radius_2 is None: # use the smallest distance between the center and data walls
            radius_2 = min(center[0], center[1], nx-center[0], ny-center[1])
        if radius_1 is None: # create circular mask
            radius_1 = 0

        Y, X = np.ogrid[:ny, :nx]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        theta = np.arctan2(Y - center[1], X - center[0])
        
        mask = np.logical_and(np.logical_and(np.logical_and(dist_from_center <= radius_2 , dist_from_center >= radius_1), np.angle(np.exp(1j*angle_1)) <= theta), theta <= np.angle(np.exp(1j*(angle_1+delta_angle))))
        #print(np.logical_and(dist_from_center <= radius_2 , dist_from_center >= radius_1))
        return mask
    
    def radial_integral_cartesian(self, fxy, nr=512, ntheta=512, center=None):
        """
        Compute the radial integral of a 2D scalar field in Cartesian coordinates.

        Parameters:
            fxy (ndarray): 2D scalar field to integrate.

        Returns:
            theta (ndarray): Angular grid points.
            result (ndarray): Radial integral values for each angle.
        """
        ny, nx = fxy.shape
        if center is None:
            center = (ny // 2, nx // 2)

        y0, x0 = center

        r_max = np.min([x0, nx - x0, y0, ny - y0])

        if nr is None:
            nr = int(r_max)

        r = np.linspace(0, r_max, nr)
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)

        # Create polar grid (r, theta) and map to Cartesian coordinates
        R, Theta = np.meshgrid(r, theta, indexing='ij')
        X = x0 + R * np.cos(Theta)
        Y = y0 + R * np.sin(Theta)

        # Interpolate f(x, y) at these coordinates
        coords = np.array([Y.ravel(), X.ravel()])
        values = map_coordinates(fxy, coords, order=1, mode='reflect').reshape(nr, ntheta)

        # Integrate along r
        result = integrate.trapezoid(values, x=r, axis=0)  # shape: (n_theta,)
        return theta, result
        
    def Leggetts_integral(self, fxyz, nr=512, ntheta=512, delta_z = 0.01, center=None, with_radial_profile=False):
        """
        Compute Leggett's integral and the full volume integral of the scalar field.

        Returns:
            legget_integral (float): Leggett's integral.
            full_volume_integral (float): Full volume integral.
            radial_integral (ndarray, optional): Radial integral values (if with_radial_profile=True).
        """
        z_integral = integrate.trapezoid(fxyz, dx=delta_z, axis=-1)

        theta, radial_integral = self.radial_integral_cartesian(z_integral, nr=nr, ntheta=ntheta, center=center)

        legget_integral = integrate.trapezoid(1 / radial_integral, x=theta)
        full_volume_integral = integrate.trapezoid(radial_integral, x=theta)

        if with_radial_profile:
            return legget_integral, full_volume_integral, radial_integral
        else:
            return legget_integral, full_volume_integral



    def exp_val_energy(self):
        hbar = 0.6347*10**5 # in (micrometer^2 * kHz * mass of 1 atom)
        m = self.atomic_mass # in atomic


        psi_cmpl = self.data[0] + 1j*self.data[1]
        psi_cmpl_conj = self.data[0] - 1j*self.data[1]




        Kx, Ky, Kz, k, dk = self.make_k_mesh()

        realspace_density = self.density_real_space()
        momentumspace_density = self.density_momentum_space()

        Rx, Ry, Rz = np.meshgrid(self.r[0], self.r[1], self.r[2], indexing='ij')
        cut = self.dipolar_cut_extract()
        R_c = np.sqrt(cut[0]**2 + cut[1]**2 + cut[2]**2)

        print("omega values", self.omega[0], self.omega[1], self.omega[2])


        def LHY_f():
            coeff = 2 * 64 * (hbar**2/m) *np.sqrt(np.pi) * (self.initial_scattering_length**(5/2)) / 3
            integrand = lambda x: (1 + self.edd * (3 * x**2 - 1)+0j)**(5/2)+0j
            integral, _ = integrate.quad(integrand, 0, 1, complex_func=True)


            
            return (coeff * integral).real


        def V_dd_tilde():
            k_mod = np.sqrt(Kx**2 + Ky**2 + Kz**2)
            if R_c == None:
                return np.where(k_mod <= 1e-6, -4*np.pi*(hbar**2/m)*self.dipolar_length, 12*np.pi*(hbar**2/m)*self.dipolar_length*((Kz/k_mod)**2-1/3))
            
            else:
                return np.where(k_mod <= 1e-6, 0, 12*np.pi*(hbar**2/m)*self.dipolar_length*((Kz/k_mod)**2-1/3)*(1+3*np.cos(R_c*k_mod)/(R_c*k_mod)**2- 3*np.sin(R_c*k_mod)/(R_c*k_mod)**3))

    
        fft_deriv_2 = lambda: fft.ifftn(-(Kx**2+Ky**2+Kz**2)*fft.fftn(psi_cmpl))

        fd_deriv_2 = lambda: np.diff(psi_cmpl,2, axis=0, append=0, prepend=0)/(self.grid_spacing[0])**2+ np.diff(psi_cmpl,2, axis=1, append=0, prepend=0)/(self.grid_spacing[1])**2+ np.diff(psi_cmpl,2, axis=2, append=0, prepend=0)/(self.grid_spacing[2])**2

        fft_deriv_separate = lambda: fft.ifftn(1j*np.sqrt(Kx**2+Ky**2+Kz**2)*fft.fftn(psi_cmpl))*fft.ifftn(1j*np.sqrt(Kx**2+Ky**2+Kz**2)*fft.fftn(psi_cmpl_conj))


        V_ext = lambda: (m/2)*( self.omega[2]**2*Rz**2 + (self.omega[0]**2*Rx**2 + self.omega[1]**2*Ry**2)**self.trap_steepness )     

        kinetic_term_separate = lambda: -1/2*(hbar**2/m)*integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(fft_deriv_separate(), dx = self.grid_spacing[0] ,axis=0), dx = self.grid_spacing[1] ,axis=0),dx = self.grid_spacing[2] ,axis=0)

        ext_term = lambda: integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(V_ext()*realspace_density, dx = self.grid_spacing[0] ,axis=0), dx = self.grid_spacing[1] ,axis=0),dx = self.grid_spacing[2] ,axis=0)

        kinetic_term = lambda: -1/2*(hbar**2/m)*integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(psi_cmpl*fft_deriv_2().real, dx = self.grid_spacing[0] ,axis=0), dx = self.grid_spacing[1] ,axis=0),dx = self.grid_spacing[2] ,axis=0)

        interaction_term = lambda: integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(4*np.pi*(hbar**2/m)*self.initial_scattering_length*realspace_density**2/2, dx = self.grid_spacing[0] ,axis=0), dx = self.grid_spacing[1] ,axis=0),dx = self.grid_spacing[2] ,axis=0)
   
        LHY_term = lambda: integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(LHY_f()*realspace_density**(5/2), dx = self.grid_spacing[0] ,axis=0), dx = self.grid_spacing[1] ,axis=0),dx = self.grid_spacing[2] ,axis=0)

        phidd_term = lambda: integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(fft.ifftn(V_dd_tilde()*momentumspace_density).real*realspace_density/2, dx = self.grid_spacing[0] ,axis=0), dx = self.grid_spacing[1] ,axis=0),dx = self.grid_spacing[2] ,axis=0)

        #value = ext_term() + kinetic_term_separate() + interaction_term() + LHY_term() + phidd_term()
   
        return ext_term, kinetic_term, interaction_term, LHY_term, phidd_term, kinetic_term_separate
'''
class SingleDataProcessing:
    def __init__(self, data, RawDataExtract_instance):
        self.data = data
        self.RawDataExtract_instance = RawDataExtract_instance

    def wavefunction_real_space(self):
        if len(self.data) == 2:
            psi = self.data[0] + 1j*self.data[1]
            return psi 
        else:
            return print("This is a density not a wavefunction")
    
    def density_integrated(self):
        if len(self.data) == 1:
            density = self.data[0]
            return density
        else:
            return print("This is not a wavefunction")


    def density_real_space(self):
        if len(self.data) == 2:
            density = np.abs(self.wavefunction_real_space())**2
        elif len(self.data) == 1:
            density = np.abs(self.density_integrated())

        return density

    def phase(self):
        phase = np.angle(self.wavefunction_real_space())
        return(phase)

    def wavefunction_momentum_space(self):
        psi_k = fft.fftshift(fft.fftn(self.wavefunction_real_space()))*self.RawDataExtract_instance.grid_spacing[0]*self.RawDataExtract_instance.grid_spacing[1]*self.RawDataExtract_instance.grid_spacing[2]/(2*np.pi)**(3/2)
        return psi_k

    def density_momentum_space(self):
        density_k = np.abs(self.wavefunction_momentum_space())**2
        return density_k

    def create_angular_mask(self, nx, ny, center=None, radius_1=None, radius_2 = None, angle_1 = -np.pi, delta_angle = 2*np.pi):
        
        if center is None: # use the middle of the image
            center = (int(nx/2), int(ny/2))
        if radius_2 is None: # use the smallest distance between the center and data walls
            radius_2 = min(center[0], center[1], nx-center[0], ny-center[1])
        if radius_1 is None: # create circular mask
            radius_1 = 0

        Y, X = np.ogrid[:ny, :nx]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        theta = np.arctan2(Y - center[1], X - center[0])
        
        mask = np.logical_and(np.logical_and(np.logical_and(dist_from_center <= radius_2 , dist_from_center >= radius_1), np.angle(np.exp(1j*angle_1)) <= theta), theta <= np.angle(np.exp(1j*(angle_1+delta_angle))))
        #print(np.logical_and(dist_from_center <= radius_2 , dist_from_center >= radius_1))
        return mask

    

    def radial_sum(self, data, center=None, bin_size=1.0, max_radius='outer'):
        """
        Perform radial summation over concentric circles around a specified center in the data. Mimics diplib implementation.

        Parameters:
        - data (ndarray): Input 2D or 3D NumPy array representing the data.
        - center (tuple): The center point (x, y[, z]) for radial summation. Defaults to the center of the data.
        - bin_size (float): Size of each radial bin.
        - max_radius (str): 'inner' for minimum distance to boundary or 'outer' for maximum distance.

        Returns:
        - bins (ndarray): 1D array with the sum of pixel values in each radial bin.


        -Note multiplication with the spacing is neccessary in order to get the right order of magnitude

        """
        
        # Validate input data dimensions
        if data.ndim < 2:
            raise ValueError("Radial projection requires at least 2 dimensions")

        # Determine the center of the data if not provided
        if center is None:
            center = tuple((np.array(data.shape) - 1) / 2.0)
        center = np.array(center)
        
        # Check if center is within the bounds of the data
        if np.any(center < 0) or np.any(center >= data.shape):
            raise ValueError("Center point is outside of the data bounds")

        # Compute the maximum distance per dimension
        max_distances = np.maximum(center, np.array(data.shape) - center)
        
        # Determine the maximum radius based on the chosen mode
        if max_radius == 'inner':
            radius = max_distances.min()
        elif max_radius == 'outer':
            radius = np.linalg.norm(max_distances)
        else:
            raise ValueError("Invalid max_radius flag. Use 'inner' or 'outer'")

        # Number of bins should match (shape-center) / bin_size
        num_bins = int(np.ceil(radius / bin_size))
        bins = np.zeros(num_bins)
        
        # Create a meshgrid of coordinates
        coords = np.indices(data.shape).reshape(data.ndim, -1).T
        distances = np.linalg.norm(coords - center, axis=1)
        
        # Bin the distances and sum the corresponding pixel values
        for bin_idx in range(num_bins):
            bin_mask = (distances >= bin_idx * bin_size) & (distances < (bin_idx + 1) * bin_size)
            bins[bin_idx] = data.ravel()[bin_mask].sum()
        
        return bins



    def occupation_number_k_space(self, center=None, bin_size=1.0, max_radius='inner', data = None, spacing = None):
      
        if data is None:
            data = self.density_momentum_space()
            spacing = (self.RawDataExtract_instance.dk[0], self.RawDataExtract_instance.dk[1])
        
        # mask_momentum = self.create_angular_mask(range[0]*2, range[1]*2, center = None, radius_1 = radius_1, radius_2 = radius_2)

        # density_k_cut = np.ma.masked_array((data[range[0]:3*range[0],range[1]:3*range[1]]), ~mask_momentum)

        radial_profile_k = self.radial_sum(data = data, center = center, bin_size = bin_size, max_radius = max_radius)

        return radial_profile_k*spacing[0]
    



    def cylindrical_integrate_trapezoidal(self,data, x, y, z, r_max, Nr, Ntheta, endpoint=False):
        """
        Numerically integrate a scalar field `data(x, y, z)` over a cylindrical volume
        using trapezoidal rule in cylindrical coordinates.

        Parameters:
            data     : 3D numpy array, function values on (x, y, z) grid
            x, y, z  : 1D arrays specifying the Cartesian grid points
            r_max    : maximum radius for integration
            Nr       : number of radial points
            Ntheta   : number of angular points
            endpoint : whether to include r_max and pi in grids

        Returns:
            float: the approximate integral over the cylindrical domain
        """
        r = np.linspace(0, r_max, Nr, endpoint=endpoint)
        theta = np.linspace(0, 2*np.pi, Ntheta, endpoint=endpoint)
        z_new = z  # use same vertical discretization

        dr = r[1]-r[0]
        dz = z[1] -z[0]
        dtheta = theta[1] - theta[0]
        # Set up the interpolator for the original grid
        interpolator = RegularGridInterpolator((x, y, z), data, bounds_error=False, fill_value=0)

        # Preallocate result per theta
        f_theta = np.zeros(Ntheta)

        # Integration loop over theta
        for i, th in enumerate(theta):
            # Create 2D mesh for r-z plane
            R, Z = np.meshgrid(r, z_new, indexing='ij')  # shapes (Nr, Nz)

            # Convert to Cartesian
            X = R * np.cos(th)
            Y = R * np.sin(th)

            # Stack for interpolation
            points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

            # Interpolate and reshape back to (Nr, Nz)
            rho = interpolator(points).reshape(R.shape)

            # Multiply by r to account for cylindrical volume element
            weighted_rho = rho * R

            # Integrate over z (axis=1), result shape: (Nr,)
            int_z = integrate.trapezoid(weighted_rho, z_new, axis=1, dx=dz)

            # Integrate over r (axis=0), result: scalar
            f_theta[i] = integrate.trapezoid(int_z, r, dx = dr)

        # Integrate over theta (axis=0)
        integral = integrate.trapezoid(1/f_theta, theta, dx = dtheta)

        return integral, f_theta


    def superfluid_density(self, data, x=None, y=None, z=None, r_max=None, Nr=100, Ntheta=100, p_number=None, endpoint =False):
        if x == None:
            x = self.RawDataExtract_instance.r[0]
        if y == None:
            y = self.RawDataExtract_instance.r[1]
        if z == None:
            z = self.RawDataExtract_instance.r[2]
        if r_max == None:
          r_max = self.RawDataExtract_instance.r_max[0]
        
        if p_number == None:
            p_number = self.RawDataExtract_instance.number_of_particles
            print(p_number)

        integral_value, f_theta = self.cylindrical_integrate_trapezoidal(data, x, y, z, r_max, Nr, Ntheta, endpoint)

        return np.pi*2/(p_number*integral_value), f_theta
'''
   
''' 
class RadialIntegrals:
    fxyz = None

    def __init__(self, center=None, nr=512, ntheta=360, dz=384, with_radial_profile=False):
        """
        Initialize the RadialIntegrals class.

        Parameters:
            fxyz (ndarray): 3D scalar field to integrate.
            center (tuple): Center of the integration domain (y0, x0).
            nr (int): Number of radial points.
            ntheta (int): Number of angular points.
            dz (float): Spacing along the z-axis.
            with_radial_profile (bool): Whether to return the radial profile.
        """

        self.center = center
        self.nr = nr
        self.ntheta = ntheta
        self.dz = dz
        self.with_radial_profile = with_radial_profile

    
    def create_angular_mask(self, nx, ny, center=None, radius_1=None, radius_2 = None, angle_1 = -np.pi, delta_angle = 2*np.pi):

        if center is None: # use the middle of the image
            center = (int(nx/2), int(ny/2))
        if radius_2 is None: # use the smallest distance between the center and data walls
            radius_2 = min(center[0], center[1], nx-center[0], ny-center[1])
        if radius_1 is None: # create circular mask
            radius_1 = 0

        Y, X = np.ogrid[:ny, :nx]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        theta = np.arctan2(Y - center[1], X - center[0])
        
        mask = np.logical_and(np.logical_and(np.logical_and(dist_from_center <= radius_2 , dist_from_center >= radius_1), np.angle(np.exp(1j*angle_1)) <= theta), theta <= np.angle(np.exp(1j*(angle_1+delta_angle))))
        #print(np.logical_and(dist_from_center <= radius_2 , dist_from_center >= radius_1))
        return mask
    
    def radial_integral_cartesian(self, fxy):
        """
        Compute the radial integral of a 2D scalar field in Cartesian coordinates.

        Parameters:
            fxy (ndarray): 2D scalar field to integrate.

        Returns:
            theta (ndarray): Angular grid points.
            result (ndarray): Radial integral values for each angle.
        """
        ny, nx = fxy.shape
        if self.center is None:
            self.center = (ny // 2, nx // 2)

        y0, x0 = self.center

        r_max = np.min([x0, nx - x0, y0, ny - y0])

        if self.nr is None:
            self.nr = int(r_max)

        r = np.linspace(0, r_max, self.nr)
        theta = np.linspace(0, 2 * np.pi, self.ntheta, endpoint=False)

        # Create polar grid (r, theta) and map to Cartesian coordinates
        R, Theta = np.meshgrid(r, theta, indexing='ij')
        X = x0 + R * np.cos(Theta)
        Y = y0 + R * np.sin(Theta)

        # Interpolate f(x, y) at these coordinates
        coords = np.array([Y.ravel(), X.ravel()])
        values = map_coordinates(fxy, coords, order=1, mode='reflect').reshape(self.nr, self.ntheta)

        # Integrate along r
        result = integrate.trapezoid(values, x=r, axis=0)  # shape: (n_theta,)
        return theta, result

    def Leggetts_integral(self):
        """
        Compute Leggett's integral and the full volume integral of the scalar field.

        Returns:
            legget_integral (float): Leggett's integral.
            full_volume_integral (float): Full volume integral.
            radial_integral (ndarray, optional): Radial integral values (if with_radial_profile=True).
        """
        z_integral = integrate.trapezoid(self.fxyz, dx=self.dz, axis=-1)

        theta, radial_integral = self.radial_integral_cartesian(z_integral)

        legget_integral = integrate.trapezoid(1 / radial_integral, x=theta)
        full_volume_integral = integrate.trapezoid(radial_integral, x=theta)

        if self.with_radial_profile:
            return legget_integral, full_volume_integral, radial_integral
        else:
            return legget_integral, full_volume_integral

    #def angular_correlation(self, radius_1=None, radius_2=None, range =  None, data = None)


    """

    deprecated functions that didnt work properly
    """
    # def radial_profile(self, data, spacing ,center = None, num_bins = int(256), min_pixel_count=0, smooth_sigma=1.0):
    #     if center is None:
    #         center = (data.shape[0]/2, data.shape[1]/2)
    #     y, x = np.indices((data.shape))
    #     if spacing is None:Bachelorarbeit/cylindrical_well/with_cutoff/edd1.500000/Data_analyse.ipynb

    #     nr = np.bincount(r.ravel(), minlength = int(data.shape[0]/2))
    #     radial_profile = tbin /np.maximum(nr,1)

    #     bin_edges = np.arange(len(radial_profile) + 1) * np.mean(np.diff(np.unique(r)))

    #     return radial_profile
    #     # print(r.shape, data.shape)
    #     # # Define radial bins
    #     # r_max = r.max()
    #     # bin_edges = np.linspace(0, r_max, num_bins + 1)
    #     # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
    #     # # Calculate the radial profile using histogram
    #     # radial_sum, _ = np.histogram(r, bins=bin_edges, weights=data)
    #     # pixel_count, _ = np.histogram(r, bins=bin_edges)
        
    #     # # Avoid division by zero for empty bins
    #     # radial_profile = np.zeros_like(bin_centers)
    #     # valid_bins = pixel_count >= min_pixel_count
    #     # radial_profile[valid_bins] = radial_sum[valid_bins] / np.maximum(pixel_count[valid_bins],1)
    
    #     # # Apply Gaussian smoothing to the radial profile
    #     # radial_profile = gaussian_filter1d(radial_profile, sigma=smooth_sigma)
        
    #     # # Calculate the areas of the annular bins
    #     # annular_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)

    #     # return radial_profile*annular_areas

    # def occupation_number_k_space(self, radius_1=None, radius_2=None, range =  None, data = None, spacing = None):
        

    #     if range is None:
    #         range = (
    #             int(self.RawDataExtract_instance.number_grid_points[0] / 4),
    #             int(self.RawDataExtract_instance.number_grid_points[1] / 4),
    #         )
        
    #     if data is None:
    #         data = self.density_momentum_space()
    #         spacing = (self.RawDataExtract_instance.dk[0], self.RawDataExtract_instance.dk[1])
        
    #     mask_momentum = self.create_angular_mask(range[0]*2, range[1]*2, center = None, radius_1 = radius_1, radius_2 = radius_2)

    #     density_k_cut = np.ma.masked_array((data[range[0]:3*range[0],range[1]:3*range[1]]), ~mask_momentum)

    #     radial_profile_k = self.radial_profile(data, spacing,center = None)

    #     return radial_profile_k

    # # 

  '''
    
 