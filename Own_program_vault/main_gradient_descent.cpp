#include "gradient_descent.hpp"

#include <iostream>
#include <vector>

#include <fstream>
#include <string>



int main()
{   

    
    {
        std::vector<double> edd_values = {1.500, 1.600};
        std::vector<double> Particle_numbers = {350e3};


        // Trap types: cylinder_hard_wall, x2, x4, x6, x8, x10. For tested values, look at "MyHeaders/trap_parameters.txt"

        std::vector<std::string> trap_types = {"cylinder_hard_wall"};
        for (const auto& trap_type : trap_types){
            for (const auto& N : Particle_numbers){
                for(const auto& edd : edd_values){
                    
                    for (int i = 0; i < 2; ++i) 
                    {

                        quench::Input_from_text input;
                        
                        input.phys.number_of_particles = N;
                        input.initial.radius = 10.185;
                        input.final_.radius = 0.0;
                        
                        input.initial.edd = edd;
                        input.final_.edd = 1.0;
                        input.cutoff.z = 6.0
                        input.initial.type = trap_type;
                        input.final_.type = "none";
                        
                        input.cutoff.use_cutoff = i;
                        
                        quench::GradientDescentSolver solver(input);
                        solver.display();
                        solver.start_gradient_descent_static();
                    }
                    
                }
            }
        }
    }
*/



}
