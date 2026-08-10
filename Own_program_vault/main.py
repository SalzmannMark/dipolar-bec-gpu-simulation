import numpy as np
import scipy.fft as fft
from vtk import vtkStructuredPointsReader

from vtkmodules.util import numpy_support as VN
import math
import os
import glob
import sys
from InputParser import InputParser 
from scipy.ndimage import gaussian_filter1d

from scipy.interpolate import RegularGridInterpolator

import scipy.integrate as integrate
from scipy.ndimage import map_coordinates
from scipy import integrate


class RawDataExtract:

    # Define class constants 
    hbar = 0.6347*10**5 #in µm^2 u/s
    a_0 = 5.29*1e-5
    #All extractions are converted to SI Units to ease plotting, reverting to dimensionless units can be done by multiplying the corresponding factors of a_0, a_ho, w_ho 

    def __init__(self, main_path_directory, parameter_file ='quench.prm', TW_run = False):
        
        #Note that path_directory should be given without last '/'

        self.TW_run = TW_run
        self.directory = main_path_directory
        self.parameter_file = parameter_file
        self.ip = InputParser(self.parameter_file, self.directory)   
        
        #Run and grid parameters in SI units
        self.r_max, self.number_grid_points, self.grid_spacing, self.r = self.grid_extract()
        
        self.k, self.dk = self.k_grid()
        self.omega = self.harmonical_trap_extract()
        self.omega_ho = self.omega[-1]
        self.edd, self.initial_scattering_length, self.final_scattering_length, self.dipolar_length, self.number_of_particles, self.atomic_mass = self.gas_params_extract()

        self.trap_type, self.trap_radius, self.trap_height, self.trap_steepness, self.trap_offset = self.cylindrical_trap_extract()

        self.use_cutoff = self.ip.retrieve_int("use_cutoff")
        self.harm_z = self.ip.retrieve_int("omegaz")
        self.shape = self.ip.retrieve_int("shape")
        self.num_realtime_steps = self.ip.retrieve_int("number of real time steps")
        self.time_step = self.ip.retrieve_float("time step")
        self.write_output_every = self.ip.retrieve_int("write output every")
        #self.k_rho = np.sqrt(self.k[0]**2+self.k[1]**2)


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
        tp = self.ip.retrieve_int("type")
        radius = self.ip.retrieve_float("radius")
        height = self.ip.retrieve_float("height")
        steepness = self.ip.retrieve_float("steepness")
        offset = self.ip.retrieve_float("offset")


        if tp == 0:
            potential_type = "none"
        elif tp == 1:
            potential_type = "tanh(x)"
        elif tp == 2:
            potential_type = "x^[2n]"
        elif tp == 3:
            potential_type = "constant discontinuous" 
        elif tp == 4:
            potential_type = "x^{" + str(2*steepness) + "}" 
         
        return potential_type, radius, height, steepness, offset
    
    def rectangular_trap_extract(self):
        tp = self.ip.retrieve_int("type")
        if tp == 0:
            potential_type = "none"
        elif tp == 1:
            potential_type = "tanh(x)"
        elif tp == 2:
            potential_type = "x^[2n]"
        elif tp == 3:
            potential_type = "constant discontinuous" 
        wall_x1 = self.ip.retrieve_float("wall_x1")
        wall_x2 = self.ip.retrieve_float("wall_x2")
        wall_y1 = self.ip.retrieve_float("wall_y1")
        wall_y2 = self.ip.retrieve_float("wall_y2")
        height = self.ip.retrieve_float("height")
        steepness = self.ip.retrieve_float("steepness")  
        return potential_type, [wall_x1, wall_y1, wall_x2, wall_y2], height, steepness    


    def k_grid(self):
        # k is [0, ..., kmax,-kmax, ..., 0-dk]
        kx = 2*np.pi*fft.fftfreq(self.number_grid_points[0], self.grid_spacing[0])
        ky = 2*np.pi*fft.fftfreq(self.number_grid_points[1], self.grid_spacing[1])
        kz = 2*np.pi*fft.fftfreq(self.number_grid_points[2], self.grid_spacing[2])
        k = (kx, ky, kz)
        dkx = kx[1] - kx[0]
        dky = ky[1] - ky[0]
        dkz = kz[1] - kz[0]
        dk = (dkx, dky, dkz)

        return k, dk
    
    def gas_params_extract(self):
        # in units of the Bohr length a_0
        edd = self.ip.retrieve_float("edd")
        final_scattering_length         = 0
        #self.ip.retrieve_float("final scattering length")
        dipolar_length            = self.ip.retrieve_float("dipolar_length")
        number_of_particles = self.ip.retrieve_int("number of particles")
        atomic_mass         = self.ip.retrieve_float("atomic mass")
        initial_scattering_length = dipolar_length/edd

        return edd, initial_scattering_length, final_scattering_length, dipolar_length, number_of_particles, atomic_mass

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

        if run_time is None:
            file_name = file + '.vtk'
        else:
            file_name = file + str(run_time) + '.vtk'

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

        if run_time is None:
            file_name = file + '.vtk'
        else:
            file_name = file + str(run_time) + '.vtk'

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
    


# class SingleDataProcessing(RawDataExtract):
#     def __init__(self, file, dimension, main_path_directory, parameter_file ='quench.prm', TW_run = False, scalars_name = ['real_psi', 'imag_psi'], run_iteration = None, run_time = None):
#         self.main_path_directory = main_path_directory
#         self.parameter_file = parameter_file
#         self.TW_run = TW_run
#         RawDataExtract.__init__(self, self.main_path_directory, self.parameter_file, self.TW_run)
#         self.file = file
#         self.dimension = dimension
#         self.scalars_name = scalars_name
#         self.run_iteration = run_iteration
#         self.run_time = run_time


#     def wavefunction_real_space(self):
#         psi = RawDataExtract.data_extract_vtk(self, self.file, self.dimension, self.scalars_name , self.run_iteration, self.run_time)[0]+1j* RawDataExtract.data_extract_vtk(self, self.file, self.dimension, self.scalars_name, self.run_iteration , self.run_time)[1]
#         return psi
#     def density(self):
#         return np.abs(self.wavefunction_real_space())**2



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


    def fft_deriv_2(self):
        return fft.ifftn(-(self.make_k_grid()[0]**2+self.make_k_grid()[1]**2+self.make_k_grid()[2]**2)*fft.fftn(self.psi()[0]))
   
    def fd_deriv_2(self):
        return np.diff(self.psi()[0],2, axis=0, append=0, prepend=0)/(self.spacing()[0])**2+ np.diff(self.psi()[0],2, axis=1, append=0, prepend=0)/(self.spacing()[1])**2+ np.diff(self.psi()[0],2, axis=2, append=0, prepend=0)/(self.spacing()[2])**2

    def fft_deriv_separate(self):
        return fft.ifftn(1j*np.sqrt(self.make_k_grid()[0]**2+self.make_k_grid()[1]**2+self.make_k_grid()[2]**2)*fft.fftn(self.psi()[0]))*fft.ifftn(1j*np.sqrt(self.make_k_grid()[0]**2+self.make_k_grid()[1]**2+self.make_k_grid()[2]**2)*fft.fftn(self.psi()[1]))

    def V_ext(self):
        return 0.5*(self.wx**2*self.make_grid()[0]**2+self.wy**2*self.make_grid()[1]**2+self.wz**2*self.make_grid()[2]**2)
   
    def V_dd_tilde(self):
        k_mod = np.sqrt(self.make_k_grid()[0]**2+self.make_k_grid()[1]**2+self.make_k_grid()[2]**2)
        if self.R_c == None:
            return np.where(k_mod <= 1e-6, -4*np.pi*self.a_dd, 12*np.pi*self.a_dd*((self.make_k_grid()[2]/k_mod)**2-1/3))
           
        else:
            return np.where(k_mod <= 1e-6, 0, 12*np.pi*self.a_dd*((self.make_k_grid()[2]/k_mod)**2-1/3)*(1+3*np.cos(self.R_c*k_mod)/(self.R_c*k_mod)**2- 3*np.sin(self.R_c*k_mod)/(self.R_c*k_mod)**3))
               
   
    def LHY(self):
        def integrand(u, eps):
           return np.sqrt(1+eps*(3*u**2-1)+0j)**5
        #integrand = lambda x, eps: (1+eps*(3*x**2-1))**(5/2)
        I = integrate.quad(integrand, 0, 1, args=(self.eps_dd))
        return 2*64*np.sqrt(np.pi)*self.a_s**(5/2)/3*I[0].real

    def kinetic_term_separate(self):
        return -1/2*integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(self.fft_deriv_separate(), dx = self.spacing()[0] ,axis=0), dx = self.spacing()[1] ,axis=0),dx = self.spacing()[2] ,axis=0)
   
    def ext_term(self):
        return integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(self.V_ext()*self.density(), dx = self.spacing()[0] ,axis=0), dx = self.spacing()[1] ,axis=0),dx = self.spacing()[2] ,axis=0)
   
    def kinetic_term(self):
        return -1/2*integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(self.psi()[1]*self.fft_deriv_2(), dx = self.spacing()[0] ,axis=0), dx = self.spacing()[1] ,axis=0),dx = self.spacing()[2] ,axis=0)
   
    def interaction_term(self):
        return integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(4*np.pi*self.a_s*self.density()**2/2, dx = self.spacing()[0] ,axis=0), dx = self.spacing()[1] ,axis=0),dx = self.spacing()[2] ,axis=0)
   
    def LHY_term(self):
        return integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(self.LHY()*self.density()**(5/2), dx = self.spacing()[0] ,axis=0), dx = self.spacing()[1] ,axis=0),dx = self.spacing()[2] ,axis=0)
   
    def phidd_term(self):  
        return integrate.trapezoid(integrate.trapezoid(integrate.trapezoid(fft.ifftn(self.V_dd_tilde()*self.density_k()).real*self.density()/2, dx = self.spacing()[0] ,axis=0), dx = self.spacing()[1] ,axis=0),dx = self.spacing()[2] ,axis=0)
   
    def energy(self):
        return self.ext_term() + self.kinetic_term_separate() + self.interaction_term() + self.LHY_term() + self.phidd_term()

class RadialIntegrals:
    def __init__(self, fxyz, center=None, nr=512, ntheta=360, dz=384, with_radial_profile=False):
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
        self.fxyz = fxyz
        self.center = center
        self.nr = nr
        self.ntheta = ntheta
        self.dz = dz
        self.with_radial_profile = with_radial_profile

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

  
    
 