# GPU-Accelerated Simulation of Dipolar Bose-Einstein Condensates

Numerical simulation of three-dimensional dipolar Bose-Einstein condensates
(BECs) by solving the Gross-Pitaevskii equation (GPE) using C++ and CUDA.

The code was developed as part of my Bachelor thesis at Heidelberg University
and is designed for GPU-accelerated simulations of dipolar quantum gases,
including ground-state calculations and time evolution.

## Bachelor Thesis

The methods, numerical implementation, and physical results presented in this
repository are described in my Bachelor thesis:

[**Bachelor Thesis — PDF**](thesis/Bachelorarbeit_Mark_Salzmann.pdf)

## Overview

The program provides a numerical framework for simulating dipolar Bose-Einstein
condensates in three spatial dimensions. Physical parameters such as particle
number, interaction strengths, trap geometry, and numerical resolution can be
adjusted to investigate different regimes of the system.

The main focus of the project is the numerical investigation of long-range
dipolar interactions and the influence of interaction cutoffs on the simulated
condensate.

## Physical Model

The condensate is described by the Gross-Pitaevskii equation

$ \alpha + \beta = \gamma$

$$
i\hbar\frac{\partial\Psi(\mathbf r,t)}{\partial t}
=
\left[
-\frac{\hbar^2\nabla^2}{2m}
+V(\mathbf r)
+g|\Psi(\mathbf r,t)|^2
+\Phi_{\mathrm{dd}}(\mathbf r,t)
\right]\Psi(\mathbf r,t),
$$

where the non-local dipolar mean-field potential is given by

$$
\Phi_{\mathrm{dd}}(\mathbf r)
=
\int d^3r'\,
V_{\mathrm{dd}}(\mathbf r-\mathbf r')
|\Psi(\mathbf r')|^2.
$$

The simulations consider three-dimensional dipolar condensates and allow
different external trapping potentials and interaction parameters to be
investigated.

## Numerical Methods

- Gross-Pitaevskii equation
- Imaginary-time evolution for ground-state calculations
- Gradient-based energy minimization
- Fourier-space evaluation of the non-local dipolar interaction
- Fast Fourier transforms (FFT)
- Split-step Fourier method for real-time evolution
- Three-dimensional spatial discretization

## Computational Implementation

The simulation framework is implemented using:

- **C++** for the numerical implementation
- **CUDA** for GPU acceleration
- **cuFFT** for Fourier transforms
- **CMake** for building the project
- **Linux/HPC** for large-scale simulations

The code is designed to exploit GPU parallelism for computationally intensive
parts of the simulation, allowing significantly larger spatial grids to be
studied than would be practical with a purely CPU-based implementation.

## External Dependencies

This project uses the
**UltraCold** software framework for parts of the numerical implementation.

UltraCold is an external community project and is not developed as part of
this repository. The code specific to the simulations and analysis presented
here is maintained separately in this repository.

[UltraCold repository](<INSERT-ULTRACOLD-GITHUB-LINK>)

## Results

The simulations were used to investigate the influence of a finite interaction
cutoff on dipolar Bose-Einstein condensates.

Among the investigated effects are:

- Changes in the critical interaction strength for the onset of structured
  density configurations
- Modifications of the condensate density near the boundaries
- Formation and arrangement of density droplets
- Differences between simulations with and without an interaction cutoff
- Effects of the external trapping potential on the resulting ground states

Example simulation results:

<p align="center">
  <img src="images/results/example_result.png" width="700">
</p>

*Example ground-state density obtained from the numerical simulations.*

## Performance

The simulations are GPU-accelerated using CUDA and cuFFT. This allows the
three-dimensional Gross-Pitaevskii equation to be evaluated on spatial grids
with hundreds of points in each spatial direction.

For quantitative performance comparisons, benchmark results should be reported
for a specified hardware configuration, grid size, and simulation procedure.

## Reproducibility

### Requirements

- Linux
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- CMake
- C++ compiler
- UltraCold and its dependencies

### Building

Clone this repository:

```bash
git clone https://github.com/SalzmannMark/dipolar-bec-gpu-simulation.git
cd dipolar-bec-gpu-simulation
