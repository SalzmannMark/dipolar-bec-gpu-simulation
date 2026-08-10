#if defined(__has_include)
#  if __has_include(<cuda.h>) && __has_include(<cuda_runtime.h>)
#    include <cuda.h>
#    include <cuda_runtime.h>
#    define QUENCH_HAS_CUDA 1
#  elif __has_include(<cuda.h>)
#    include <cuda.h>
#    define QUENCH_HAS_CUDA 1
#  elif __has_include(<cuda_runtime_api.h>)
#    include <cuda_runtime_api.h>
#    define QUENCH_HAS_CUDA 1
#  else
#    include <cstring>
#    include <complex>
     using cuDoubleComplex = std::complex<double>;
     using cudaError_t = int;
     enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };
     inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind) { std::memcpy(dst, src, count); return 0; }
#    define QUENCH_HAS_CUDA 0
#  endif
#else
#  include <cuda.h>
#  include <cuda_runtime.h>
#  define QUENCH_HAS_CUDA 1
#endif


#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib> 
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
#include "gradient_descent.hpp" // Ensure this header defines Input_from_text
#include "UltraCold.hpp"

using namespace UltraCold;
namespace quench {

    inline Input_from_text load_params_from_file(const std::string& path) 
    {
        UltraCold::Tools::InputParser ip(path.c_str());
        ip.read_input_file();

        Input_from_text p;

        p.grid.xmax = ip.retrieve_double("xmax");
        p.grid.ymax = ip.retrieve_double("ymax");
        p.grid.zmax = ip.retrieve_double("zmax");
        p.grid.nx   = ip.retrieve_int("nx");
        p.grid.ny   = ip.retrieve_int("ny");
        p.grid.nz   = ip.retrieve_int("nz");

        p.phys.dipolar_length          = ip.retrieve_double("dipolar_length");
        p.phys.number_of_particles     = ip.retrieve_int("number of particles");
        p.phys.atomic_mass             = ip.retrieve_double("atomic mass");
        p.phys.final_scattering_length = ip.retrieve_double("final scattering length");

        p.initial.omegax = ip.retrieve_double("omegax");
        p.initial.omegay = ip.retrieve_double("omegay");
        p.initial.omegaz = ip.retrieve_double("omegaz");
        p.initial.theta  = ip.retrieve_double("theta");
        p.initial.phi    = ip.retrieve_double("phi");

        // ensure your .prm provides these *_final keys
        p.final_.omegax = ip.retrieve_double("omegax_final");
        p.final_.omegay = ip.retrieve_double("omegay_final");
        p.final_.omegaz = ip.retrieve_double("omegaz_final");
        p.final_.theta  = ip.retrieve_double("theta_final");
        p.final_.phi    = ip.retrieve_double("phi_final");

        p.initial.type      = ip.retrieve_string("type");
        p.initial.height    = ip.retrieve_double("height");
        p.initial.steepness = ip.retrieve_double("steepness");
        p.initial.offset    = ip.retrieve_double("offset");

        p.final_.type      = ip.retrieve_string("type_final");

        p.final_.height    = ip.retrieve_double("height_final");
        p.final_.steepness = ip.retrieve_double("steepness_final");
        p.final_.offset    = ip.retrieve_double("offset_final");

        p.cutoff.use_cutoff = ip.retrieve_int("use_cutoff");
        p.cutoff.x = ip.retrieve_double("cut_x");
        p.cutoff.y = ip.retrieve_double("cut_y");
        p.cutoff.z = ip.retrieve_double("cut_z");
    
        p.algo.num_initial_graddesc_steps   = ip.retrieve_int("number of gradient descent steps initial");
        p.algo.residual        = ip.retrieve_double("residual");
        p.algo.alpha           = ip.retrieve_double("alpha");
        p.algo.beta            = ip.retrieve_double("beta");
        p.algo.num_realtime_steps = ip.retrieve_int("number of real time steps");
        p.algo.time_step       = ip.retrieve_double("time step");
        p.algo.write_output_every     = ip.retrieve_int("write output every");

        p.initial.edd = ip.retrieve_double("edd");
        p.final_.edd = ip.retrieve_double("edd_final");
        p.initial.radius = ip.retrieve_double("radius");
        p.final_.radius = ip.retrieve_double("radius_final");



        return p;
    }



    inline Input_from_text load_params_without_runparams(const std::string& path) 
    {
        UltraCold::Tools::InputParser ip(path.c_str());
        ip.read_input_file();

        Input_from_text p;

        p.grid.xmax = ip.retrieve_double("xmax");
        p.grid.ymax = ip.retrieve_double("ymax");
        p.grid.zmax = ip.retrieve_double("zmax");
        p.grid.nx   = ip.retrieve_int("nx");
        p.grid.ny   = ip.retrieve_int("ny");
        p.grid.nz   = ip.retrieve_int("nz");

        p.phys.dipolar_length          = ip.retrieve_double("dipolar_length");
        p.phys.number_of_particles     = ip.retrieve_int("number of particles");
        p.phys.atomic_mass             = ip.retrieve_double("atomic mass");
        p.phys.final_scattering_length = ip.retrieve_double("final scattering length");

        p.initial.omegax = ip.retrieve_double("omegax");
        p.initial.omegay = ip.retrieve_double("omegay");
        p.initial.omegaz = ip.retrieve_double("omegaz");
        p.initial.theta  = ip.retrieve_double("theta");
        p.initial.phi    = ip.retrieve_double("phi");

        // ensure your .prm provides these *_final keys
        p.final_.omegax = ip.retrieve_double("omegax_final");
        p.final_.omegay = ip.retrieve_double("omegay_final");
        p.final_.omegaz = ip.retrieve_double("omegaz_final");
        p.final_.theta  = ip.retrieve_double("theta_final");
        p.final_.phi    = ip.retrieve_double("phi_final");

        p.initial.type      = ip.retrieve_string("type");
        p.initial.height    = ip.retrieve_double("height");

        p.initial.offset    = ip.retrieve_double("offset");

        p.final_.type      = ip.retrieve_string("type_final");
        p.final_.height    = ip.retrieve_double("height_final");

        p.final_.offset    = ip.retrieve_double("offset_final");

        p.cutoff.use_cutoff = ip.retrieve_int("use_cutoff");
        p.cutoff.x = ip.retrieve_double("cut_x");
        p.cutoff.y = ip.retrieve_double("cut_y");
        p.cutoff.z = ip.retrieve_double("cut_z");
    
        p.algo.num_initial_graddesc_steps   = ip.retrieve_int("number of gradient descent steps initial");
        p.algo.residual        = ip.retrieve_double("residual");
        p.algo.alpha           = ip.retrieve_double("alpha");
        p.algo.beta            = ip.retrieve_double("beta");
        p.algo.num_realtime_steps = ip.retrieve_int("number of real time steps");
        p.algo.time_step       = ip.retrieve_double("time step");
        p.algo.write_output_every     = ip.retrieve_int("write output every");

        return p;
    }

    class Dipoles3d : public cudaSolvers::DipolarGPSolver 
    {   
        public:
        using DipolarGPSolver::DipolarGPSolver;

        std::string destination_folder;

        inline Dipoles3d(Vector<double>& x,
                Vector<double>& y,
                Vector<double>& z,
                Vector<std::complex<double>>& psi,
                Vector<double>& Vext,
                double scattering_length,
                double dipolar_length,
                double theta_mu,
                double phi_mu,
                Vector<double> cutoff,
                bool lhy,
                const std::string& destination_folder)
        : cudaSolvers::DipolarGPSolver(x, y, z, psi, Vext, scattering_length, dipolar_length, theta_mu, phi_mu, cutoff, lhy),
        destination_folder(destination_folder) {}

        inline Dipoles3d(Vector<double>& x,
                Vector<double>& y,
                Vector<double>& z,
                Vector<std::complex<double>>& psi,
                Vector<double>& Vext,
                double scattering_length,
                double dipolar_length,
                double theta_mu,
                double phi_mu,
                bool lhy,
                bool cuda_FFT,
                const std::string& destination_folder)
        : cudaSolvers::DipolarGPSolver(x, y, z, psi, Vext, scattering_length, dipolar_length, theta_mu, phi_mu, lhy, cuda_FFT),
        destination_folder(destination_folder) {}

        inline Dipoles3d(Vector<double>& x,
                Vector<double>& y,
                Vector<double>& z,
                Vector<std::complex<double>>& psi,
                Vector<double>& Vext,
                double scattering_length,
                double dipolar_length,
                double theta_mu,
                double phi_mu,
                bool lhy,
                const std::string& destination_folder)
        : cudaSolvers::DipolarGPSolver(x, y, z, psi, Vext, scattering_length, dipolar_length, theta_mu, phi_mu, lhy),
        destination_folder(destination_folder) {}

        inline void reinit_a(Vector<double>& Vext, Vector<std::complex<double>>& psi, double scattering_length) {
            cudaMemcpy(wave_function_d,    psi.data(), npoints*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
            cudaMemcpy(external_potential_d, Vext.data(), npoints*sizeof(double),       cudaMemcpyHostToDevice);
            cudaMemcpy(scattering_length_d, &scattering_length, sizeof(double),         cudaMemcpyHostToDevice);
        }

        inline void write_operator_splitting_output(size_t iteration_number,
                                            std::ostream& output_stream) override {

            const int output_index = 2+ iteration_number/write_output_every;
            if ((iteration_number % (write_output_every) == 0) && output_index < 50) {  // outputs every
                
                copy_out_wave_function();
                GraphicOutput::DataWriter data_out;         
                std::string slice_name = destination_folder + "/slice_" + std::to_string(output_index);
                data_out.set_output_name(slice_name.c_str());
                data_out.write_slice2d_vtk(x_axis ,y_axis, wave_function_output,"xy","psi","ASCII");
            }
            else if ((iteration_number % (write_output_every*10) == 0) && output_index >= 50) {  // Outputs every tenth

                copy_out_wave_function();
                GraphicOutput::DataWriter data_out;         

                std::string slice_name = destination_folder + "/slice_" + std::to_string(output_index);
                data_out.set_output_name(slice_name.c_str());
                data_out.write_slice2d_vtk(x_axis ,y_axis, wave_function_output,"xy","psi","ASCII");

            }

                /*if (iteration_number % (write_output_every*10) == 0) {
                GraphicOutput::DataWriter psi_out;   
                
                std::string output_name = destination_folder + "/psi_" + std::to_string(1+iteration_number/write_output_every);
                psi_out.set_output_name(output_name.c_str());
                psi_out.write_vtk(x_axis, y_axis, z_axis, wave_function_output,"psi","ASCII");
                }*/
            

            

        }
    };

    inline std::string GradientDescentSolver::get_executable_directory() {
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        if (count == -1) throw std::runtime_error("Failed to get executable path");
        std::string full_path(result, static_cast<size_t>(count));
        return std::filesystem::path(full_path).parent_path().string();
    } 

    inline GradientDescentSolver::GradientDescentSolver(const Simulation_parameters& run_parameters, bool a, int n)
            : run_parameters_(run_parameters), do_Imaginary_time_evolution(a), num_Imaginary_time_steps(n)   // copy the whole struct
    {
        
        
        Ip.initial.radius = run_parameters_.radius_initial;
        Ip.final_.radius = run_parameters_.radius_final;
        Ip.initial.edd = run_parameters_.edd_initial;
        Ip.final_.edd = run_parameters_.edd_final;

        import_parameters();    

        calculate_omega();
    }
    inline GradientDescentSolver::GradientDescentSolver(double initial_radius, double final_radius, double initial_edd, double final_edd, bool a, int n)
            : do_Imaginary_time_evolution(a), num_Imaginary_time_steps(n)  // default initialize the struct
    {    
        Ip.initial.radius = initial_radius;
        Ip.final_.radius = final_radius;
        Ip.initial.edd = initial_edd;
        Ip.final_.edd = final_edd;

        import_from_omega_database();

        calculate_omega();
    }

    inline GradientDescentSolver::GradientDescentSolver(const Input_from_text& input, bool a, int n)
            : Ip(input), do_Imaginary_time_evolution(a), num_Imaginary_time_steps(n)  // copy the whole struct
    {

        import_new_from_database();

        calculate_omega();
               
   
    }


    inline GradientDescentSolver::GradientDescentSolver(bool a, int n)
            : do_Imaginary_time_evolution(a), num_Imaginary_time_steps(n)  // default initialize the struct
    {    
        import_parameters();

        calculate_omega();
    }


    inline std::vector<TrapDatabase> GradientDescentSolver::load_trap_parameters(const std::string& filename)
    {
        std::ifstream f(filename);
        std::vector<TrapDatabase> traps;

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);

            TrapDatabase trap;
            ss >> trap.trap_type >> trap.trap_radius >> trap.number_of_particles >> trap.omega_z >> trap.edd >> trap.omega_r;
            traps.push_back(trap);
        }
        return traps;
    }

    inline void GradientDescentSolver::display() const {

        std::cout << "initial trap type = " << Ip.initial.type << std::endl;
        std::cout << "final trap type = " << Ip.final_.type << std::endl;
        std::cout << "edd = " << Ip.initial.edd << std::endl;
        std::cout << "edd_final = " << Ip.final_.edd << std::endl;
        std::cout << "radius = " << Ip.initial.radius << std::endl;
        std::cout << "radius_final = " << Ip.final_.radius << std::endl;
        std::cout << "initial scattering_length = " << Ip.phys.initial_scattering_length/Phys::bohr_radius << std::endl;
        std::cout << "final scattering_length = " << Ip.phys.final_scattering_length/Phys::bohr_radius << std::endl;
        std::cout << "dipolar_length = " << Ip.phys.dipolar_length/Phys::bohr_radius << std::endl;


        std::cout << "Trapping frequencies: " <<  "omega_x = " << Ip.initial.omegax*Ip.initial.omega_ho/TWOPI << " , omega_y = " << Ip.initial.omegay*Ip.initial.omega_ho/TWOPI << " , omega_z = " << Ip.initial.omegaz*Ip.initial.omega_ho/TWOPI << std::endl;
        std::cout << "Final Trapping frequencies: " <<  "omega_x = " << Ip.final_.omegax*Ip.final_.omega_ho/TWOPI << " , omega_y = " << Ip.final_.omegay*Ip.final_.omega_ho/TWOPI << " , omega_z = " << Ip.final_.omegaz*Ip.final_.omega_ho/TWOPI << std::endl;
        std::cout << "Harmonic oscillator frequency = " << Ip.initial.omega_ho/TWOPI << std::endl;
        std::cout << "Final Harmonic oscillator frequency = " << Ip.final_.omega_ho/TWOPI << std::endl;
        
        std::cout << "Main directory: " << main_directory << std::endl;
        std::cout << "Destination: " << destination_folder << std::endl;

    }

    inline void GradientDescentSolver::import_from_omega_database()
    {   using std::filesystem::create_directories;
        using std::filesystem::exists;
        using std::filesystem::copy;
        using std::filesystem::copy_options;
        
        std::string source_file = main_directory + "/parameters.prm";
        //std::cout << "source_file = " << source_file << std::endl;


        std::ostringstream edd_oss;
        edd_oss << std::fixed << std::setprecision(3) << Ip.initial.edd;
        std::string edd_initial_str = edd_oss.str();
        std::ostringstream radius_oss;
        radius_oss << std::fixed << std::setprecision(3) << Ip.initial.radius;
        std::string radius_value_str = radius_oss.str();


        if (do_Imaginary_time_evolution)
        {
            destination_folder = main_directory + "/Itime/r_"+ radius_value_str + "_edd_" + edd_initial_str;
        }
        else
        {
            destination_folder = main_directory + "/Rtime/r_"+ radius_value_str + "_edd_" + edd_initial_str;
        }
        
        std::filesystem::create_directories(destination_folder);

        individual_parameters_file= destination_folder + "/individual_parameters.prm";

        if (!std::filesystem::exists(source_file)) {
            std::cerr << "Source file does not exist: " << source_file << std::endl;
        }
        std::filesystem::copy(source_file, individual_parameters_file, std::filesystem::copy_options::overwrite_existing);

        
        // Overwrite or add edd, radius in the parameter file
        std::ifstream infile(individual_parameters_file);
        std::stringstream buffer;
        bool found_edd = false, found_edd_final = false, found_radius = false, found_radius_final = false;


        if (infile.is_open()) {
            std::string line;
            while (std::getline(infile, line)) {
            if (line.find("edd =") == 0) {
                buffer << "edd = " << Ip.initial.edd << "\n";
                found_edd = true;
            } else if (line.find("edd_final =") == 0) {
                buffer << "edd_final = " << Ip.final_.edd << "\n";
                found_edd_final = true;
            } else if (line.find("radius =") == 0) {
                buffer << "radius = " << Ip.initial.radius << "\n";
                found_radius = true;
            } else if (line.find("radius_final =") == 0) {
                buffer << "radius_final = " << Ip.final_.radius << "\n";
                found_radius_final = true;
            }  else {
                buffer << line << "\n";
            }
            }
            infile.close();
        }

        // Add missing keys
        if (!found_edd)         buffer << "edd = " << Ip.initial.edd << "\n";
        if (!found_edd_final)   buffer << "edd_final = " << Ip.final_.edd << "\n";
        if (!found_radius)      buffer << "radius = " << Ip.initial.radius << "\n";
        if (!found_radius_final)buffer << "radius_final = " << Ip.final_.radius << "\n";


        std::ofstream outfile(individual_parameters_file, std::ios::trunc);
        if (outfile.is_open()) {
            outfile << buffer.str();
            outfile.close();
        } else {
            std::cerr << "Error opening file: " << individual_parameters_file << std::endl;
        }

        Ip = load_params_from_file(individual_parameters_file);      

        auto params_from_trapdatabase = [this](const Trap& trap, const std::vector<TrapDatabase>& loaded_traps, int number_of_particles) -> double {
            bool found = false;
            for (const auto& trap_db : loaded_traps){
                if (trap_db.trap_type == trap.type
                && trap_db.trap_radius == trap.radius
                && trap_db.number_of_particles == number_of_particles
                && trap_db.omega_z == trap.omegaz
                && trap_db.edd == trap.edd)
                {
                    found = true;
                    return trap_db.omega_r; 
                }
            }

            if (!found) {
                std::cerr << "No matching trap found in trap_parameters.txt" << std::endl;
                return 0.0;
            }
            return 0.0;
        };
        std::string trap_file = "/einc/prod/users/salzmann/cluster_home/ultracold-dipolar/MyHeaders/trap_parameters.txt";
        std::vector<TrapDatabase> loaded_traps = load_trap_parameters(trap_file);


        Ip.initial.omegax = Ip.initial.omegay = params_from_trapdatabase(Ip.initial, loaded_traps, Ip.phys.number_of_particles);
        Ip.final_.omegax  = Ip.final_.omegay  = params_from_trapdatabase(Ip.final_, loaded_traps, Ip.phys.number_of_particles);
        {
            // Overwrite or add omega entries in the individual parameters file
            std::ifstream infile(individual_parameters_file);
            std::stringstream buffer;
            bool found_omegax = false, found_omegay = false;
            bool found_omegax_final = false, found_omegay_final = false;


            if (infile.is_open()) {
                std::string line;
                while (std::getline(infile, line)) {
                    if (line.find("omegax =") == 0) {
                        buffer << "omegax = " << std::fixed << std::setprecision(4) << Ip.initial.omegax << "\n";
                        found_omegax = true;
                    } else if (line.find("omegay =") == 0) {
                        buffer << "omegay = " << std::fixed << std::setprecision(4) << Ip.initial.omegay << "\n";
                        found_omegay = true;
                    } else if (line.find("omegax_final =") == 0) {
                        buffer << "omegax_final = " << std::fixed << std::setprecision(4) << Ip.final_.omegax << "\n";
                        found_omegax_final = true;
                    } else if (line.find("omegay_final =") == 0) {
                        buffer << "omegay_final = " << std::fixed << std::setprecision(4) << Ip.final_.omegay << "\n";
                        found_omegay_final = true;
                    } else {
                        buffer << line << "\n";
                    }
                }
                infile.close();
            }

            // Append missing omega keys
            if (!found_omegax)        buffer << "omegax = " << std::fixed << std::setprecision(6) << Ip.initial.omegax << "\n";
            if (!found_omegay)        buffer << "omegay = " << std::fixed << std::setprecision(6) << Ip.initial.omegay << "\n";  
            if (!found_omegax_final)  buffer << "omegax_final = " << std::fixed << std::setprecision(6) << Ip.final_.omegax << "\n";
            if (!found_omegay_final)  buffer << "omegay_final = " << std::fixed << std::setprecision(6) << Ip.final_.omegay << "\n";
        

            std::ofstream outfile(individual_parameters_file, std::ios::trunc);
            if (outfile.is_open()) {
                outfile << buffer.str();
                outfile.close();

            } else {
                std::cerr << "Error opening file to write: " << individual_parameters_file << std::endl;
            }

            // Reload parameters to ensure Ip reflects file contents
            Ip = load_params_from_file(individual_parameters_file);
        }

    }

        inline void GradientDescentSolver::import_new_from_database()
    {   using std::filesystem::create_directories;
        using std::filesystem::exists;
        using std::filesystem::copy;
        using std::filesystem::copy_options;
        
        std::string source_file = main_directory + "/parameters.prm";
        //std::cout << "source_file = " << source_file << std::endl;


        std::ostringstream edd_oss;
        edd_oss << std::fixed << std::setprecision(3) << Ip.initial.edd;
        std::string edd_initial_str = edd_oss.str();

        std::ostringstream radius_oss;
        radius_oss << std::fixed << std::setprecision(3) << Ip.initial.radius;
        std::string radius_value_str = radius_oss.str();
        std::ostringstream number_oss;
        number_oss << std::fixed << std::setprecision(3) << Ip.phys.number_of_particles;
        std::string number_value_str = number_oss.str();



        if (do_Imaginary_time_evolution)
        {
            {
                std::filesystem::path parent = std::filesystem::path(main_directory).parent_path();
                std::filesystem::path dest = parent / Ip.initial.type / ("N_" + number_value_str + "_cutoff_" + std::to_string(Ip.cutoff.use_cutoff)) / "Itime" / ("r_" + radius_value_str + "_edd_" + edd_initial_str);
                destination_folder = dest.string();
            }
        }
        else
        {
                std::filesystem::path parent = std::filesystem::path(main_directory).parent_path();
                std::filesystem::path dest = parent / Ip.initial.type / ("N_" + number_value_str + "_cutoff_" + std::to_string(Ip.cutoff.use_cutoff)) / "Rtime" / ("r_" + radius_value_str + "_edd_" + edd_initial_str);
                destination_folder = dest.string();
        }
        
        std::filesystem::create_directories(destination_folder);

        individual_parameters_file= destination_folder + "/individual_parameters.prm";

        if (!std::filesystem::exists(source_file)) {
            std::cerr << "Source file does not exist: " << source_file << std::endl;
        }
        std::filesystem::copy(source_file, individual_parameters_file, std::filesystem::copy_options::overwrite_existing);


        auto determine_steepness = [this](const Trap& trap) -> int {
            if (trap.type == "cylinder_hard_wall")
            {
                return 0;
            }
            else if (trap.type == "x2")
            {
                return 1;
            }
            if (trap.type == "x4")
            {
                return 2;
            }
            else if (trap.type == "x6")
            {
                return 3;
            }
            else if (trap.type == "x8")
            {
                return 4;
            }
            else if (trap.type == "x10")
            {
                return 5;
            }
            else if (trap.type == "none")
            {
                return 0;
            }
            else
            {
                std::cerr << "Unknown trap type: " << trap.type << std::endl;
                return 0; // or some other error handling
            }
        };
        
        Ip.initial.steepness = determine_steepness(Ip.initial);
        Ip.final_.steepness = determine_steepness(Ip.final_);

        // Overwrite or add edd, radius in the parameter file
        std::ifstream infile(individual_parameters_file);
        std::stringstream buffer;
        bool found_edd = false, found_edd_final = false, found_radius = false, found_radius_final = false;
        bool found_number_of_particles = false;
        bool found_type = false, found_type_final = false;
        bool found_use_cutoff = false;
        bool found_steepness = false, found_steepness_final = false;

        if (infile.is_open()) {
            std::string line;
            while (std::getline(infile, line)) {
            if (line.find("edd =") == 0) {
                buffer << "edd = " << Ip.initial.edd << "\n";
                found_edd = true;
            } else if (line.find("edd_final =") == 0) {
                buffer << "edd_final = " << Ip.final_.edd << "\n";
                found_edd_final = true;
            } else if (line.find("radius =") == 0) {
                buffer << "radius = " << Ip.initial.radius << "\n";
                found_radius = true;
            } else if (line.find("radius_final =") == 0) {
                buffer << "radius_final = " << Ip.final_.radius << "\n";
                found_radius_final = true;
            }
            else if (line.find("number of particles =") == 0) {
                buffer << "number of particles = " << Ip.phys.number_of_particles << "\n";
                found_number_of_particles = true;

            } else if (line.find("type =") == 0) {
                buffer << "type = " << Ip.initial.type << "\n";
                found_type = true;

            } else if(line.find("steepness =") == 0) {
                buffer << "steepness = " << Ip.initial.steepness << "\n";
                found_steepness = true;

            } else if (line.find("steepness_final =") == 0) {
                buffer << "steepness_final = " << Ip.final_.steepness << "\n";
                found_steepness_final = true;

            }
            
            else if (line.find("type_final =") == 0) {
                buffer << "type_final = " << Ip.final_.type << "\n";
                found_type_final = true;

            } else if (line.find("use_cutoff =") == 0) {
                buffer << "use_cutoff = " << Ip.cutoff.use_cutoff << "\n";
                found_use_cutoff = true;
            }
            else 
            {
                buffer << line << "\n";
            }
            }
            infile.close();
        }

        // Add missing keys
        if (!found_edd)         buffer << "edd = " << Ip.initial.edd << "\n";
        if (!found_edd_final)   buffer << "edd_final = " << Ip.final_.edd << "\n";
        if (!found_radius)      buffer << "radius = " << Ip.initial.radius << "\n";
        if (!found_radius_final)buffer << "radius_final = " << Ip.final_.radius << "\n";
        if (!found_number_of_particles) buffer << "number of particles = " << Ip.phys.number_of_particles << "\n";
        if (!found_type)                buffer << "type = " << Ip.initial.type << "\n";
        if (!found_type_final)          buffer << "type_final = " << Ip.final_.type << "\n";
        if (!found_steepness)           buffer << "steepness = " << Ip.initial.steepness << "\n";
        if (!found_steepness_final)     buffer << "steepness_final = " << Ip.final_.steepness << "\n";
        if (!found_use_cutoff)          buffer << "use_cutoff = " << Ip.cutoff.use_cutoff << "\n";

 
        std::ofstream outfile(individual_parameters_file, std::ios::trunc);
        if (outfile.is_open()) {
            outfile << buffer.str();
            outfile.close();
        } else {
            std::cerr << "Error opening file: " << individual_parameters_file << std::endl;
        }

        Ip = load_params_from_file(individual_parameters_file);      

        auto params_from_trapdatabase = [this](const Trap& trap, const std::vector<TrapDatabase>& loaded_traps, int number_of_particles) -> double {
            bool found = false;
            for (const auto& trap_db : loaded_traps){
                if (trap_db.trap_type == trap.type
                && trap_db.trap_radius == trap.radius
                && trap_db.number_of_particles == number_of_particles
                && trap_db.omega_z == trap.omegaz
                && trap_db.edd == trap.edd)
                {
                    found = true;
                    return trap_db.omega_r; 
                }
            }

            if (!found) {
                std::cerr << "No matching trap found in trap_parameters.txt" << std::endl;
                return 0.0;
            }
            return 0.0;
        };
        std::string trap_file = "/einc/prod/users/salzmann/cluster_home/ultracold-dipolar/MyHeaders/trap_parameters.txt";
        std::vector<TrapDatabase> loaded_traps = load_trap_parameters(trap_file);


        Ip.initial.omegax = Ip.initial.omegay = params_from_trapdatabase(Ip.initial, loaded_traps, Ip.phys.number_of_particles);
        Ip.final_.omegax  = Ip.final_.omegay  = params_from_trapdatabase(Ip.final_, loaded_traps, Ip.phys.number_of_particles);
        {
            // Overwrite or add omega entries in the individual parameters file
            std::ifstream infile(individual_parameters_file);
            std::stringstream buffer;
            bool found_omegax = false, found_omegay = false;
            bool found_omegax_final = false, found_omegay_final = false;

            if (infile.is_open()) {
                std::string line;
                while (std::getline(infile, line)) {
                    if (line.find("omegax =") == 0) {
                        buffer << "omegax = " << std::fixed << std::setprecision(4) << Ip.initial.omegax << "\n";
                        found_omegax = true;
                    } else if (line.find("omegay =") == 0) {
                        buffer << "omegay = " << std::fixed << std::setprecision(4) << Ip.initial.omegay << "\n";
                        found_omegay = true;
                    } else if (line.find("omegax_final =") == 0) {
                        buffer << "omegax_final = " << std::fixed << std::setprecision(4) << Ip.final_.omegax << "\n";
                        found_omegax_final = true;
                    } else if (line.find("omegay_final =") == 0) {
                        buffer << "omegay_final = " << std::fixed << std::setprecision(4) << Ip.final_.omegay << "\n";
                        found_omegay_final = true;
                    } else {
                        buffer << line << "\n";
                    }
                }
                infile.close();
            }

            // Append missing omega keys
            if (!found_omegax)        buffer << "omegax = " << std::fixed << std::setprecision(6) << Ip.initial.omegax << "\n";
            if (!found_omegay)        buffer << "omegay = " << std::fixed << std::setprecision(6) << Ip.initial.omegay << "\n";  
            if (!found_omegax_final)  buffer << "omegax_final = " << std::fixed << std::setprecision(6) << Ip.final_.omegax << "\n";
            if (!found_omegay_final)  buffer << "omegay_final = " << std::fixed << std::setprecision(6) << Ip.final_.omegay << "\n";
        

            std::ofstream outfile(individual_parameters_file, std::ios::trunc);
            if (outfile.is_open()) {
                outfile << buffer.str();
                outfile.close();
      
            } else {
                std::cerr << "Error opening file to write: " << individual_parameters_file << std::endl;
            }

            // Reload parameters to ensure Ip reflects file contents
            Ip = load_params_from_file(individual_parameters_file);
        }

    }



    inline void GradientDescentSolver::import_parameters(){
        using std::filesystem::create_directories;
        using std::filesystem::exists;
        using std::filesystem::copy;
        using std::filesystem::copy_options;
        
        std::string source_file = main_directory + "/parameters.prm";
        //std::cout << "source_file = " << source_file << std::endl;

        

        if(run_parameters_.radius_initial < 0.0){
            std::cout << "Reading in parameters from individual_paremeters.prm file" << std::endl;


            Ip = load_params_from_file(source_file);

            std::cout << "Overwriting parameters in individual_parameters.prm file" << std::endl;
            
            // Overwrite or add edd, radius values in the parameter file
            std::ifstream infile(individual_parameters_file);
            std::stringstream buffer;
            bool found_edd = false, found_edd_final = false, found_radius = false, found_radius_final = false;

            if (infile.is_open()) {
                std::string line;
                while (std::getline(infile, line)) {
                if (line.find("edd =") == 0) {
                    buffer << "edd = " << Ip.initial.edd << "\n";
                    found_edd = true;
                } else if (line.find("edd_final =") == 0) {
                    buffer << "edd_final = " << Ip.final_.edd << "\n";
                    found_edd_final = true;
                } else if (line.find("radius =") == 0) {
                    buffer << "radius = " << Ip.initial.radius << "\n";
                    found_radius = true;
                } else if (line.find("radius_final =") == 0) {
                    buffer << "radius_final = " << Ip.final_.radius << "\n";
                    found_radius_final = true;
                } 
                else {
                    buffer << line << "\n";
                }
                }
                infile.close();
            }

            // Add missing keys
            if (!found_edd)         buffer << "edd = " << Ip.initial.edd << "\n";
            if (!found_edd_final)   buffer << "edd_final = " << Ip.final_.edd << "\n";
            if (!found_radius)      buffer << "radius = " << Ip.initial.radius << "\n";
            if (!found_radius_final)buffer << "radius_final = " << Ip.final_.radius << "\n";


            std::ofstream outfile(individual_parameters_file, std::ios::trunc);
            if (outfile.is_open()) {
                outfile << buffer.str();
                outfile.close();
            } else {
                std::cerr << "Error opening file: " << individual_parameters_file << std::endl;
            }

            Ip = load_params_from_file(individual_parameters_file);  



        }

        //std::string individual_parameters_file;
        //std::string destination_folder;

        if (Ip.initial.radius < 0.1)
        {

            std::ostringstream edd_oss;
            edd_oss << std::fixed << std::setprecision(3) << Ip.initial.edd;
            std::string edd_initial_str = edd_oss.str();

            if (do_Imaginary_time_evolution)
            {
                destination_folder = main_directory + "/Itime/edd_" + edd_initial_str;
            }
            else
            {
                destination_folder = main_directory + "/Rtime/edd_" + edd_initial_str;
            }
            
            std::filesystem::create_directories(destination_folder);

            individual_parameters_file= destination_folder + "/individual_parameters.prm";
        }
        else
        {
            std::ostringstream edd_oss;
            edd_oss << std::fixed << std::setprecision(3) << Ip.initial.edd;
            std::string edd_initial_str = edd_oss.str();
            std::ostringstream radius_oss;
            radius_oss << std::fixed << std::setprecision(3) << Ip.initial.radius;
            std::string radius_value_str = radius_oss.str();


            if (do_Imaginary_time_evolution)
            {
                destination_folder = main_directory + "/Itime/r_"+ radius_value_str + "_edd_" + edd_initial_str;
            }
            else
            {
                destination_folder = main_directory + "/Rtime/r_"+ radius_value_str + "_edd_" + edd_initial_str;
            }
            
            std::filesystem::create_directories(destination_folder);

            individual_parameters_file= destination_folder + "/individual_parameters.prm";
        }

        // Create a directory for the current edd value
        

        if (!std::filesystem::exists(source_file)) {
            std::cerr << "Source file does not exist: " << source_file << std::endl;
        }

        // Copy the parameter file into the created folder
        std::filesystem::copy(source_file, individual_parameters_file, std::filesystem::copy_options::overwrite_existing);


        if(run_parameters_.radius_initial >= 0.0){
            std::cout << "Overwriting parameters in individual_parameters.prm file" << std::endl;
            
            // Overwrite or add edd, radius in the parameter file
            std::ifstream infile(individual_parameters_file);
            std::stringstream buffer;
            bool found_edd = false, found_edd_final = false, found_radius = false, found_radius_final = false;

            if (infile.is_open()) {
                std::string line;
                while (std::getline(infile, line)) {
                if (line.find("edd =") == 0) {
                    buffer << "edd = " << Ip.initial.edd << "\n";
                    found_edd = true;
                } else if (line.find("edd_final =") == 0) {
                    buffer << "edd_final = " << Ip.final_.edd << "\n";
                    found_edd_final = true;
                } else if (line.find("radius =") == 0) {
                    buffer << "radius = " << Ip.initial.radius << "\n";
                    found_radius = true;
                } else if (line.find("radius_final =") == 0) {
                    buffer << "radius_final = " << Ip.final_.radius << "\n";
                    found_radius_final = true;
                }            
                else {
                    buffer << line << "\n";
                }
                }
                infile.close();
            }

            // Add missing keys
            if (!found_edd)         buffer << "edd = " << Ip.initial.edd << "\n";
            if (!found_edd_final)   buffer << "edd_final = " << Ip.final_.edd << "\n";
            if (!found_radius)      buffer << "radius = " << Ip.initial.radius << "\n";
            if (!found_radius_final)buffer << "radius_final = " << Ip.final_.radius << "\n";


            std::ofstream outfile(individual_parameters_file, std::ios::trunc);
            if (outfile.is_open()) {
                outfile << buffer.str();
                outfile.close();
            } else {
                std::cerr << "Error opening file: " << individual_parameters_file << std::endl;
            }

            Ip = load_params_from_file(individual_parameters_file);      
        }
        
    }

    inline void GradientDescentSolver::calculate_omega(){
    
        auto omega_check = [&](Trap& trap) {        


            trap.omegax = trap.omegax*TWOPI;
            trap.omegay = trap.omegay*TWOPI;
            trap.omegaz = trap.omegaz*TWOPI;

            trap.omega_ho = trap.omegaz;

        if (trap.omegaz < 0.001){
                    trap.omega_ho = 1.0;
                    std::cout << "trap.omegaz cannot be zero, please check your input parameters" << std::endl;
            }
            
            
        }; //omega_check lambda

        Ip.phys.initial_scattering_length = Ip.phys.dipolar_length * Phys::bohr_radius / Ip.initial.edd;
        Ip.phys.final_scattering_length = Ip.phys.dipolar_length * Phys::bohr_radius / Ip.final_.edd;

        Ip.phys.dipolar_length *= Phys::bohr_radius;


        omega_check(Ip.initial);

        omega_check(Ip.final_); 
        

        Ip.initial.omegax = Ip.initial.omegax/Ip.initial.omega_ho;
        Ip.initial.omegay = Ip.initial.omegay/Ip.initial.omega_ho;
        Ip.initial.omegaz = Ip.initial.omegaz/Ip.initial.omega_ho;

        Ip.final_.omegax = Ip.final_.omegax/Ip.final_.omega_ho;
        Ip.final_.omegay = Ip.final_.omegay/Ip.final_.omega_ho;
        Ip.final_.omegaz = Ip.final_.omegaz/Ip.final_.omega_ho;
        
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        Ip.algo.time_step *= TWOPI*Ip.initial.omega_ho/1000.0;



        const double a_ho = std::sqrt(Phys::hbar/(Ip.phys.atomic_mass*Ip.initial.omega_ho));
        const double a_ho_final = a_ho; // std::sqrt(Phys::hbar/(Ip.phys.atomic_mass*Ip.final_.omega_ho));
        
        Ip.phys.initial_scattering_length *= 1/a_ho;
        Ip.phys.final_scattering_length *= 1/a_ho_final;

        Ip.phys.dipolar_length *= 1/a_ho;
        
        Ip.grid.xmax *= 1/a_ho;
        Ip.grid.ymax *= 1/a_ho;
        Ip.grid.zmax *= 1/a_ho;

        Ip.grid.dx = 2 * Ip.grid.xmax / Ip.grid.nx;
        Ip.grid.dy = 2 * Ip.grid.ymax / Ip.grid.ny;
        Ip.grid.dz = 2 * Ip.grid.zmax / Ip.grid.nz;

        Ip.initial.radius *= 1/a_ho;
        Ip.final_.radius *= 1/a_ho_final;

        Ip.cutoff.x *= 1/a_ho;
        Ip.cutoff.y *= 1/a_ho;
        Ip.cutoff.z *= 1/a_ho;
    }

    inline void GradientDescentSolver::calculate_potential(Vector<double>& Vext, Vector<double>& Vext_final, Vector<std::complex<double>>& psi,
                        const Vector<double>& x, const Vector<double>& y, const Vector<double>& z)
    {
       
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0,1);


        //std::default_random_engine generator_2;
        //std::uniform_real_distribution<double> distribution_2(0,1);

        //std::poisson_distribution<int> distribution_3(5);
                
        // Precompute lambda functions for each potential type
        auto potential_func = [this](const Trap& trap, double xi, double yj, double zk) -> double {

            double Mx = (trap.height-trap.offset)/(Ip.grid.xmax - trap.radius);
            double Bx = trap.height - Mx*Ip.grid.xmax;

            if (trap.type == "none") {
                return 0.0;                
            } 
            else if (trap.type == "x2" || trap.type == "x4" || trap.type == "x6" || trap.type == "x8" || trap.type == "x10") {

                return std::min(trap.height, 0.5*(  (trap.omegaz*trap.omegaz)*(zk*zk) + 
                                        std::pow(std::pow(trap.omegax,2)*xi*xi + std::pow(trap.omegay,2)*yj*yj,trap.steepness))); 
            }
            else if (trap.type == "cylinder_hard_wall") {

                if(std::sqrt(xi*xi + yj*yj) < trap.radius) {
                    return std::min(trap.height, 0.5 * std::pow(trap.omegaz,2)*(zk*zk));
                } 
                else {
                    return std::min(trap.height, 0.5*std::pow(trap.omegaz,2)*(zk*zk) + trap.height);
                }
            }
            else if (trap.type == "trapezoidal_hard_wall") {
                return std::min(trap.height, 0.5 * std::pow(trap.omegaz,2)*(zk*zk) + std::max(Mx*std::sqrt(xi*xi + yj*yj) + Bx,0.0));
            }
            else if(trap.type == "masked"){
                return std::min(trap.height, 0.5 * std::pow(trap.omegaz,2)*(zk*zk));
            }
            
            else {
                throw std::invalid_argument("Unknown trap type");
            }
            
            
        };

        

        if (Ip.initial.type == "masked") // initializing wavefunction inside the masked domain
        {
            std::cout << "Adding potential mask from file" << std::endl;

            std::string potential_mask_file = main_directory + "/mask2d.txt";
            
            Vector<double> mask_data(Ip.grid.nx, Ip.grid.ny, Ip.grid.nz);

            std::ifstream in("mask2d.txt");
            if(!in) {
                throw std::runtime_error("mask2d.txt: open failed");
            }

            std::string line;
            line.reserve(Ip.grid.ny);

            for(int i = 0; i < Ip.grid.nx; ++i) {
                std::getline(in, line);
                if(line.size() < Ip.grid.ny) {
                    throw std::runtime_error("mask2d.txt: line too short");
                }

                for(int j = 0; j < Ip.grid.ny; ++j) {
                    const double v = (line[j] == '1' ? 1.0 : 0.0);

                    // replicate along z axis
                    for(int k = 0; k < Ip.grid.nz; ++k) {
                        mask_data(i, j, k) = v;
                    }
                }
            }

            
            for (int i = 0; i < Ip.grid.nx; ++i)
            for (int j = 0; j < Ip.grid.ny; ++j)
            for (int k = 0; k < Ip.grid.nz; ++k)
            {
                if(mask_data(i,j,k) < 0.8) {  
                    double random_number = distribution(generator);
                    psi(i,j,k) = 1*std::pow(random_number,3);
                } 
                
                Vext(i, j, k) = potential_func(Ip.initial, x[i], y[j], z[k]);
                Vext_final(i, j, k) = potential_func(Ip.final_, x[i], y[j], z[k]);
                Vext(i, j, k) += mask_data(i,j,k)*Ip.initial.height;
                Vext_final(i, j, k) += mask_data(i,j,k)*Ip.final_.height;
            }
        }
        else if (Ip.initial.type == "cylinder_hard_wall")
        {        
            for (int i = 0; i < Ip.grid.nx; ++i)
            for (int j = 0; j < Ip.grid.ny; ++j)
            for (int k = 0; k < Ip.grid.nz; ++k)
            {
                Vext(i, j, k) = potential_func(Ip.initial, x[i], y[j], z[k]);
                Vext_final(i, j, k) = potential_func(Ip.final_, x[i], y[j], z[k]);
                if(std::sqrt(x[i]*x[i] + y[j]*y[j]) < Ip.initial.radius) {
                    double random_number = distribution(generator);
                    psi(i,j,k) = 1*std::pow(random_number,3);
                }
            }
        }
        else //Initializing wavefunction in the whole trap
        { 
            for (int i = 0; i < Ip.grid.nx; ++i)
            for (int j = 0; j < Ip.grid.ny; ++j)
            for (int k = 0; k < Ip.grid.nz; ++k)
            {
                double random_number = distribution(generator);
                psi(i,j,k) = 1*std::pow(random_number,3);
                Vext(i, j, k) = potential_func(Ip.initial, x[i], y[j], z[k]);
                Vext_final(i, j, k) = potential_func(Ip.final_, x[i], y[j], z[k]);
            }
        }
        
    
    }

    inline void GradientDescentSolver::start_quench(){

        std::cout << "Starting quench with gradient descent initial state" << std::endl;
        

        Vector<double> x(Ip.grid.nx), y(Ip.grid.ny), z(Ip.grid.nz), kx(Ip.grid.nx), ky(Ip.grid.ny), kz(Ip.grid.nz);

        for (int i = 0; i < Ip.grid.nx; ++i) x[i] = -Ip.grid.xmax + i * Ip.grid.dx;
        for (int i = 0; i < Ip.grid.ny; ++i) y[i] = -Ip.grid.ymax + i * Ip.grid.dy;
        for (int i = 0; i < Ip.grid.nz; ++i) z[i] = -Ip.grid.zmax + i * Ip.grid.dz;
        create_mesh_in_Fourier_space(x, y, z, kx, ky, kz);
        
        Vector<std::complex<double>> psi(Ip.grid.nx, Ip.grid.ny, Ip.grid.nz);
        Vector<double> Vext(Ip.grid.nx, Ip.grid.ny, Ip.grid.nz);
        Vector<double> Vext_final(Ip.grid.nx, Ip.grid.ny, Ip.grid.nz);

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0,1);
        std::default_random_engine generator_2;
        std::uniform_real_distribution<double> distribution_2(0,1);

        std::poisson_distribution<int> distribution_3(5);


        calculate_potential(Vext, Vext_final, psi, x, y, z);
        
 
        std::cout << "Vext(0,0,0) = " << Vext(0, 0, 0) << std::endl;

        std::cout << "Maximum of the harmonic potential = " << Vext(Ip.grid.nx/2, Ip.grid.ny/2, Ip.grid.nz-1) << std::endl;
        std::cout << "Vext(centre) = " << Vext(Ip.grid.nx/2, Ip.grid.ny/2, Ip.grid.nz/2) << std::endl;

        double norm = 0.0;
        for (size_t i = 0; i < psi.size(); ++i) norm += std::norm(psi[i]);
        norm *= (Ip.grid.dx * Ip.grid.dy * Ip.grid.dz);
        for (size_t i = 0; i < psi.size(); ++i) psi[i] *= std::sqrt(Ip.phys.number_of_particles / norm);

        GraphicOutput::DataWriter psi_out;
        //std::string combined_initial_name = destination_folder+ "/initial_wave_function";
        //psi_out.set_output_name(combined_initial_name.c_str());
        //psi_out.write_vtk(x,y,z,psi,"psi","ASCII");
        
        bool lhy = true;

        if (Ip.cutoff.use_cutoff == 1)
        {
            std::cout << "using cutoff" << std::endl;
            Vector<double> cutoff(3);
            cutoff[0] = Ip.cutoff.x;
            cutoff[1] = Ip.cutoff.y;
            cutoff[2] = Ip.cutoff.z;
            
            bool cuda_cutoff = true;
            Dipoles3d dipolar_gp_solver(x,y,z,psi,Vext,Ip.phys.initial_scattering_length,
                                                        Ip.phys.dipolar_length,
                                                        Ip.initial.theta, Ip.initial.phi, lhy, cuda_cutoff, destination_folder);
        
            //std::fstream gradient_descent_output_stream;
            //gradient_descent_output_stream.open(destination_folder +"/gradient_descent_output.csv",std::ios::out);

            double chemical_potential;
            std::tie(psi, chemical_potential) = dipolar_gp_solver.run_gradient_descent(Ip.algo.num_initial_graddesc_steps,
                                                                                    //residual,
                                                                                    Ip.algo.alpha,
                                                                                    Ip.algo.beta,
                                                                                    std::cout,
                                                                                    10);

            //gradient_descent_output_stream.close();

            
            std::string combined_ground_state_name = destination_folder+ "/ground_state_wave_function";
            psi_out.set_output_name(combined_ground_state_name.c_str());
            psi_out.write_vtk(x,y,z,psi,"psi","ASCII");

            std::string slice_name = destination_folder+ "/slice_0";
            psi_out.set_output_name(slice_name.c_str());
            psi_out.write_slice2d_vtk(x,y,psi,"xy","psi","ASCII");
            
            /*
            for (int i = 0; i < Ip.grid.nx; ++i)
            for (int j = 0; j < Ip.grid.ny; ++j)
            for (int k = 0; k < Ip.grid.nz; ++k)
                    {
                        double noise = distribution(generator);
                        psi(i,j,k) = psi(i,j,k) + noise;
                    } */

            dipolar_gp_solver.reinit(Vext_final, psi, Ip.phys.final_scattering_length);

            if(do_Imaginary_time_evolution)
            {
                std::cout << "Starting additional imaginary time evolution" << std::endl;
                std::cout << "Number of imaginary time steps = " << num_Imaginary_time_steps<< std::endl;  
                std::tie(psi, chemical_potential) = dipolar_gp_solver.run_gradient_descent(num_Imaginary_time_steps,
                                                                                    //residual,
                                                                                    Ip.algo.alpha,
                                                                                    Ip.algo.beta,
                                                                                    std::cout,
                                                                                    10);
                std::string name = destination_folder+ "/slice_1";
                psi_out.set_output_name(name.c_str());
                psi_out.write_slice2d_vtk(x,y,psi,"xy","psi","ASCII");
            }
            
            std::cout << "Starting real time evolution" << std::endl;      
            dipolar_gp_solver.run_operator_splitting(Ip.algo.num_realtime_steps,Ip.algo.time_step,std::cout,Ip.algo.write_output_every); 

        }
        if (Ip.cutoff.use_cutoff == 0)
        {
            std::cout << "without cutoff" << std::endl;

            Dipoles3d dipolar_gp_solver(x,y,z,psi,Vext,Ip.phys.initial_scattering_length,
                                                        Ip.phys.dipolar_length,
                                                        Ip.initial.theta, Ip.initial.phi, lhy, destination_folder);
            
        
            //std::fstream gradient_descent_output_stream;
            //gradient_descent_output_stream.open(destination_folder +"/gradient_descent_output.csv",std::ios::out);

            double chemical_potential;
            std::tie(psi, chemical_potential) = dipolar_gp_solver.run_gradient_descent(Ip.algo.num_initial_graddesc_steps,
                                                                                    //residual,
                                                                                    Ip.algo.alpha,
                                                                                    Ip.algo.beta,
                                                                                    std::cout,
                                                                                    10);

            //gradient_descent_output_stream.close();

            std::string combined_ground_state_name = destination_folder+ "/ground_state_wave_function";
            psi_out.set_output_name(combined_ground_state_name.c_str());
            psi_out.write_vtk(x,y,z,psi,"psi","ASCII");

            std::string slice_name = destination_folder+ "/slice_0";
            psi_out.set_output_name(slice_name.c_str());
            psi_out.write_slice2d_vtk(x,y,psi,"xy","psi","ASCII");
            
            /*
            for (int i = 0; i < Ip.grid.nx; ++i)
            for (int j = 0; j < Ip.grid.ny; ++j)
            for (int k = 0; k < Ip.grid.nz; ++k)
                    {
                        double noise = distribution(generator);
                        psi(i,j,k) = psi(i,j,k) + noise;
                    } */

            dipolar_gp_solver.reinit(Vext_final, psi, Ip.phys.final_scattering_length);

            if(do_Imaginary_time_evolution)
            {
                std::cout << "Starting additional imaginary time evolution" << std::endl;      
                std::cout << "Number of imaginary time steps = " << num_Imaginary_time_steps<< std::endl;  
                std::tie(psi, chemical_potential) = dipolar_gp_solver.run_gradient_descent(num_Imaginary_time_steps,
                                                                                    //residual,
                                                                                    Ip.algo.alpha,
                                                                                    Ip.algo.beta,
                                                                                    std::cout,
                                                                                    10);
                                                                                    
                std::string name = destination_folder+ "/slice_1";
                psi_out.set_output_name(name.c_str());
                psi_out.write_slice2d_vtk(x,y,psi,"xy","psi","ASCII");
            }

            
            std::cout << "Starting real time evolution" << std::endl;      
            dipolar_gp_solver.run_operator_splitting(Ip.algo.num_realtime_steps,Ip.algo.time_step,std::cout,Ip.algo.write_output_every);
        }
        else
        {
            std::cout << "cutoff not defined" << std::endl;
        }
        
    }


    inline void GradientDescentSolver::start_gradient_descent_static(){


        std::cout << "Starting static gradient descent (without quench)" << std::endl;
        
        Vector<double> x(Ip.grid.nx), y(Ip.grid.ny), z(Ip.grid.nz), kx(Ip.grid.nx), ky(Ip.grid.ny), kz(Ip.grid.nz);

        for (int i = 0; i < Ip.grid.nx; ++i) x[i] = -Ip.grid.xmax + i * Ip.grid.dx;
        for (int i = 0; i < Ip.grid.ny; ++i) y[i] = -Ip.grid.ymax + i * Ip.grid.dy;
        for (int i = 0; i < Ip.grid.nz; ++i) z[i] = -Ip.grid.zmax + i * Ip.grid.dz;

        create_mesh_in_Fourier_space(x, y, z, kx, ky, kz);
        
        Vector<std::complex<double>> psi(Ip.grid.nx, Ip.grid.ny, Ip.grid.nz);
        Vector<double> Vext(Ip.grid.nx, Ip.grid.ny, Ip.grid.nz);
        Vector<double> Vext_final(Ip.grid.nx, Ip.grid.ny, Ip.grid.nz);

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0,1);
        std::default_random_engine generator_2;
        std::uniform_real_distribution<double> distribution_2(0,1);

        std::poisson_distribution<int> distribution_3(5);


        calculate_potential(Vext, Vext_final, psi, x, y, z);

        std::cout << "Vext(0,0,0) = " << Vext(0, 0, 0) << std::endl;

        std::cout << "Maximum of the harmonic potential = " << Vext(Ip.grid.nx/2, Ip.grid.ny/2, Ip.grid.nz-1) << std::endl;
        std::cout << "Vext(centre) = " << Vext(Ip.grid.nx/2, Ip.grid.ny/2, Ip.grid.nz/2) << std::endl;

        double norm = 0.0;
        for (size_t i = 0; i < psi.size(); ++i) norm += std::norm(psi[i]);
        norm *= (Ip.grid.dx * Ip.grid.dy * Ip.grid.dz);
        for (size_t i = 0; i < psi.size(); ++i) psi[i] *= std::sqrt(Ip.phys.number_of_particles / norm);

        GraphicOutput::DataWriter psi_out;
        //std::string combined_initial_name = destination_folder+ "/initial_wave_function";
        //psi_out.set_output_name(combined_initial_name.c_str());
        //psi_out.write_vtk(x,y,z,psi,"psi","ASCII");
        
        bool lhy = true;

        if (Ip.cutoff.use_cutoff == 1)
        {
            std::cout << "using cutoff" << std::endl;
            Vector<double> cutoff(3);
            cutoff[0] = Ip.cutoff.x;
            cutoff[1] = Ip.cutoff.y;
            cutoff[2] = Ip.cutoff.z;

            bool cuda_cutoff = true;
            Dipoles3d dipolar_gp_solver(x,y,z,psi,Vext,Ip.phys.initial_scattering_length,
                                                        Ip.phys.dipolar_length,
                                                        Ip.initial.theta, Ip.initial.phi, lhy, cuda_cutoff, destination_folder);
            
        


            double chemical_potential;
            std::tie(psi, chemical_potential) = dipolar_gp_solver.run_gradient_descent(Ip.algo.num_initial_graddesc_steps,
                                                                                    //residual,
                                                                                    Ip.algo.alpha,
                                                                                    Ip.algo.beta,
                                                                                    std::cout,
                                                                                    10);



            std::string combined_ground_state_name = destination_folder+ "/ground_state_wave_function";
            psi_out.set_output_name(combined_ground_state_name.c_str());
            psi_out.write_vtk(x,y,z,psi,"psi","ASCII");

            std::string slice_name = destination_folder+ "/slice_0";
            psi_out.set_output_name(slice_name.c_str());
            psi_out.write_slice2d_vtk(x,y,psi,"xy","psi","ASCII");

        }

        if (Ip.cutoff.use_cutoff == 0)
        {
            std::cout << "without cutoff" << std::endl;

            Dipoles3d dipolar_gp_solver(x,y,z,psi,Vext,Ip.phys.initial_scattering_length,
                                                        Ip.phys.dipolar_length,
                                                        Ip.initial.theta, Ip.initial.phi, lhy, destination_folder);
            
        


            double chemical_potential;
            std::tie(psi, chemical_potential) = dipolar_gp_solver.run_gradient_descent(Ip.algo.num_initial_graddesc_steps,
                                                                                    //residual,
                                                                                    Ip.algo.alpha,
                                                                                    Ip.algo.beta,
                                                                                    std::cout,
                                                                                    10);


            std::string combined_ground_state_name = destination_folder+ "/ground_state_wave_function";
            psi_out.set_output_name(combined_ground_state_name.c_str());
            psi_out.write_vtk(x,y,z,psi,"psi","ASCII");

            std::string slice_name = destination_folder+ "/slice_0";
            psi_out.set_output_name(slice_name.c_str());
            psi_out.write_slice2d_vtk(x,y,psi,"xy","psi","ASCII");


        }
        else
        {
            std::cout << "cutoff not defined" << std::endl;
        }
        
    }

    inline void GradientDescentSolver::start_gradient_descent_static_saveTwoD(){


        std::cout << "Starting static gradient descent (without quench) saving only 2D data" << std::endl;
        
        Vector<double> x(Ip.grid.nx), y(Ip.grid.ny), z(Ip.grid.nz), kx(Ip.grid.nx), ky(Ip.grid.ny), kz(Ip.grid.nz);

        for (int i = 0; i < Ip.grid.nx; ++i) x[i] = -Ip.grid.xmax + i * Ip.grid.dx;
        for (int i = 0; i < Ip.grid.ny; ++i) y[i] = -Ip.grid.ymax + i * Ip.grid.dy;
        for (int i = 0; i < Ip.grid.nz; ++i) z[i] = -Ip.grid.zmax + i * Ip.grid.dz;

        create_mesh_in_Fourier_space(x, y, z, kx, ky, kz);
        
        Vector<std::complex<double>> psi(Ip.grid.nx, Ip.grid.ny, Ip.grid.nz);
        Vector<double> Vext(Ip.grid.nx, Ip.grid.ny, Ip.grid.nz);
        Vector<double> Vext_final(Ip.grid.nx, Ip.grid.ny, Ip.grid.nz);

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0,1);
        std::default_random_engine generator_2;
        std::uniform_real_distribution<double> distribution_2(0,1);

        std::poisson_distribution<int> distribution_3(5);


        calculate_potential(Vext, Vext_final, psi, x, y, z);

        std::cout << "Vext(0,0,0) = " << Vext(0, 0, 0) << std::endl;

        std::cout << "Maximum of the harmonic potential = " << Vext(Ip.grid.nx/2, Ip.grid.ny/2, Ip.grid.nz-1) << std::endl;

        std::cout << "Vext(centre) = " << Vext(Ip.grid.nx/2, Ip.grid.ny/2, Ip.grid.nz/2) << std::endl;


        double norm = 0.0;
        for (size_t i = 0; i < psi.size(); ++i) norm += std::norm(psi[i]);
        norm *= (Ip.grid.dx * Ip.grid.dy * Ip.grid.dz);
        for (size_t i = 0; i < psi.size(); ++i) psi[i] *= std::sqrt(Ip.phys.number_of_particles / norm);

        GraphicOutput::DataWriter psi_out;
        //std::string combined_initial_name = destination_folder+ "/initial_wave_function";
        //psi_out.set_output_name(combined_initial_name.c_str());
        //psi_out.write_vtk(x,y,z,psi,"psi","ASCII");
        
        bool lhy = true;
    

        if (Ip.cutoff.use_cutoff == 1)
        {   
        
            std::cout << "using cutoff" << std::endl;
            Vector<double> cutoff(3);
            cutoff[0] = Ip.cutoff.x;
            cutoff[1] = Ip.cutoff.y;
            cutoff[2] = Ip.cutoff.z;

            bool cuda_cutoff = true;
            Dipoles3d dipolar_gp_solver(x,y,z,psi,Vext,Ip.phys.initial_scattering_length,
                                                        Ip.phys.dipolar_length,
                                                        Ip.initial.theta, Ip.initial.phi, lhy, cuda_cutoff, destination_folder);
            
        
            
            double chemical_potential;
            std::tie(psi, chemical_potential) = dipolar_gp_solver.run_gradient_descent(Ip.algo.num_initial_graddesc_steps,
                                                                                    //residual,
                                                                                    Ip.algo.alpha,
                                                                                    Ip.algo.beta,
                                                                                    std::cout,
                                                                                    10);


            //std::string combined_ground_state_name = destination_folder+ "/ground_state_wave_function";
            //psi_out.set_output_name(combined_ground_state_name.c_str());
            //psi_out.write_vtk(x,y,z,psi,"psi","ASCII");

            std::string slice_name = destination_folder+ "/slice_0";
            psi_out.set_output_name(slice_name.c_str());
            psi_out.write_slice2d_vtk(x,y,psi,"xy","psi","ASCII");

        }

        if (Ip.cutoff.use_cutoff == 0)
        {
            std::cout << "without cutoff" << std::endl;

            Dipoles3d dipolar_gp_solver(x,y,z,psi,Vext,Ip.phys.initial_scattering_length,
                                                        Ip.phys.dipolar_length,
                                                        Ip.initial.theta, Ip.initial.phi, lhy, destination_folder);
            
        


            double chemical_potential;
            std::tie(psi, chemical_potential) = dipolar_gp_solver.run_gradient_descent(Ip.algo.num_initial_graddesc_steps,
                                                                                    //residual,
                                                                                    Ip.algo.alpha,
                                                                                    Ip.algo.beta,
                                                                                    std::cout,
                                                                                    10);


            //std::string combined_ground_state_name = destination_folder+ "/ground_state_wave_function";
            //psi_out.set_output_name(combined_ground_state_name.c_str());
            //psi_out.write_vtk(x,y,z,psi,"psi","ASCII");

            std::string slice_name = destination_folder+ "/slice_0";
            psi_out.set_output_name(slice_name.c_str());
            psi_out.write_slice2d_vtk(x,y,psi,"xy","psi","ASCII");


        }
        else
        {
            std::cout << "cutoff not defined" << std::endl;
        }
        
    }

} // namespace quench