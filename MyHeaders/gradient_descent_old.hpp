#pragma once
#include <complex>
#include <string>
#include <vector> // Include for std::vector




namespace UltraCold { template<class T> class Vector;  }

namespace quench{


struct Grid { double xmax{}, ymax{}, zmax{}, dx{}, dy{}, dz{}; int nx{}, ny{}, nz{}; };
struct Trap { double omegax{}, omegay{}, omegaz{}, omega_ho{}, theta{}, phi{}, height{}, steepness{}, radius{}, offset{}, edd{}; std::string type{};};
struct Cutoff { int use_cutoff{}; double x{}, y{}, z{};};
struct Algo { int num_initial_graddesc_steps{}, num_realtime_steps{}, write_output_every{}; double residual{}, alpha{}, beta{}, time_step{};  int point_x{}, point_y{}, point_z{};  };
struct Phys { double dipolar_length{}, atomic_mass{},  initial_scattering_length{} , final_scattering_length{}; int number_of_particles{}; 
              static constexpr double hbar = 0.6347e5;    // u*μm^2/s
              static constexpr double bohr_radius = 5.292e-5; // μm
};

struct Input_from_text {
    Grid grid;
    Trap initial, final_;
    Cutoff cutoff;
    Algo algo;
    Phys phys;
};

Input_from_text load_params_from_file(const std::string& path);
Input_from_text load_params_without_runparams(const std::string& path);


struct Simulation_parameters
{

    double edd_initial = -1.0;
    double edd_final = -1.0;
    double radius_initial = -1.0;
    double radius_final = -1.0;



    Simulation_parameters(double edd_i, double edd_f, double radius_i, double radius_f)
        :edd_initial(edd_i), edd_final(edd_f), radius_initial(radius_i), radius_final(radius_f)
    { }
    
    Simulation_parameters() = default; // default initialize the struct
};

struct TrapDatabase {
    std::string trap_type; // "hard_wall", "x2", "x4", "x6", "x8", "x10"
    double trap_radius; // in micrometers
    int number_of_particles; // 100000, 150000, etc.
    double omega_z; // in 2pi*Hz
    double edd;
    double omega_r;
};

class GradientDescentSolver {
    Input_from_text Ip{};
    Simulation_parameters run_parameters_;
    TrapDatabase trap_data;
    bool do_Imaginary_time_evolution;
    int num_Imaginary_time_steps;
    
    static inline std::string get_executable_directory(); // defined in .ipp

    // resolved paths
    const std::string main_directory = get_executable_directory();
    std::string destination_folder;
    std::string individual_parameters_file;

    // parsed parameters
    

public:
    explicit GradientDescentSolver(const Simulation_parameters& run_parameters, bool a, int n);
    explicit GradientDescentSolver(bool a = false, int n = 0);
    explicit GradientDescentSolver(double initial_radius, double final_radius, double initial_edd, double final_edd, bool a = false, int n = 0);
    explicit GradientDescentSolver(const Input_from_text& input, bool a = false, int n = 0);

    const Simulation_parameters& parameters_from_code() const noexcept { return run_parameters_; }
    const Input_from_text&       input_parameters_from_text() const noexcept { return Ip; }
    const std::string&           input_text_file() const noexcept { return individual_parameters_file; }

    void display() const;
    void import_parameters();
    void import_from_omega_database();
    void import_new_from_database();

    void calculate_omega();

    void calculate_potential(UltraCold::Vector<double>& Vext, UltraCold::Vector<double>& Vext_final,
                             UltraCold::Vector<std::complex<double>>& psi,
                             const UltraCold::Vector<double>& x, const UltraCold::Vector<double>& y, const UltraCold::Vector<double>& z);

    void start_quench();
    void start_gradient_descent_static();
    void start_gradient_descent_static_saveTwoD();
    std::vector<TrapDatabase> load_trap_parameters(const std::string& filename);
};

} // namespace quench


#ifdef ULTRACOLD_HEADER_ONLY
#  include "gradient_descent.ipp"
#endif




















