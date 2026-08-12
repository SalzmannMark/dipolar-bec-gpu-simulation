# GPU-Accelerated Simulation of Dipolar Bose-Einstein Condensates

Numerical investigation of phase transitions and finite-size effects in dipolar Bose-Einstein condensates using GPU-accelerated solutions of the three-dimensional Gross-Pitaevskii equation.

The project was developed as part of my Bachelor's thesis in Physics at Heidelberg University. It extends the **UltraCold** C++/CUDA simulation framework with custom solvers, trapping potentials, physical models, and simulation workflows for large-scale dipolar BEC calculations.

## Bachelor Thesis

The methods, numerical analysis, and physical results implemented in this repository are documented in the accompanying Bachelor's thesis:

[**Bachelor Thesis — PDF**](thesis/Bachelorarbeit_Mark_Salzmann.pdf)

## Overview

Dipolar Bose-Einstein condensates are quantum many-body systems in which long-range, anisotropic dipole-dipole interactions compete with short-range contact interactions. By varying the relative interaction strength, the condensate can undergo transitions from a superfluid state to structured droplet and supersolid phases.

This project investigates these phase transitions by numerically solving the three-dimensional Gross-Pitaevskii equation for a dipolar condensate of (^{164}\mathrm{Dy}) atoms. Ground-state configurations are obtained through energy minimization on a three-dimensional spatial grid, with simulations involving approximately (10^5) to (5\times10^5) atoms.

A particular focus of the project is the effect of **periodic boundary conditions on long-range dipolar interactions**. Fourier-space evaluation of the dipolar interaction using Fast Fourier Transforms introduces periodic replicas of the simulated system. Because the dipolar interaction is long-ranged, these replicas can interact with the physical condensate and produce unphysical numerical effects. An interaction cutoff is therefore implemented to suppress these periodic-image interactions.

The simulations compare ground states obtained with and without the cutoff across different trapping potentials and interaction regimes. The results demonstrate that periodic-image interactions can qualitatively modify the computed ground states, including droplet configurations, edge behavior, critical interaction strengths, and rotational symmetry.

## Numerical Methods

The simulations are based on the three-dimensional dipolar Gross-Pitaevskii equation.

### Ground-state calculation

* Energy minimization using gradient descent
* Three-dimensional spatial discretization
* Fourier-space evaluation of the dipolar interaction
* Fast Fourier Transforms for efficient convolution
* Numerical convergence and stability analysis
* Comparison of ground states with and without interaction cutoffs

### Interaction cutoff

The dipolar interaction is long-ranged, while FFT-based calculations impose periodic boundary conditions. The resulting periodic replicas can therefore introduce spurious interactions.

The implemented cutoff addresses this by restricting the dipolar interaction to a finite spatial range while ensuring that the simulation domain is sufficiently large to prevent interactions with periodic replicas.

The effect of this numerical treatment is investigated systematically across different atom numbers, trapping potentials, and interaction strengths.

## Computational Implementation

The project is implemented primarily in **C++ and CUDA** and builds on the [UltraCold](https://github.com/) scientific simulation framework.

The custom simulation layer extends the underlying framework with:

* Gradient-descent ground-state solvers
* Custom trapping potentials
* Dipolar interaction models and cutoff treatment
* Simulation parameter handling
* Automated simulation workflows
* CUDA-accelerated computational routines
* Numerical analysis and convergence checks

The underlying UltraCold codebase contains more than 20,000 lines of scientific C++/CUDA code. The project involved analyzing and extending this existing framework to support the requirements of the research problem.

Large-scale simulations were performed on Heidelberg University's **EINC GPU cluster**, using CMake and Bash for compilation, job execution, and workflow automation.

## Results

The simulations demonstrate that periodic-image interactions can produce significant numerical artifacts in dipolar BEC simulations.

### Shift of the critical interaction strength

For a cylindrical box trap with radius

[
R = 10.185,\mu\mathrm{m}
]

and (N=10^5) atoms, the critical relative dipolar interaction strength was found to be approximately

[
1.41 < \epsilon_{dd} < 1.425
]

without the interaction cutoff, compared with

[
1.404 < \epsilon_{dd} < 1.41
]

when the cutoff was applied.

The cutoff result is closer to the previously reported value of approximately

[
\epsilon_{dd}=1.40.
]

### Periodic-image effects at high atom numbers

Increasing the particle number to

[
N=5\times10^5
]

causes the condensate to extend closer to the boundaries of the simulation domain, making interactions with periodic replicas more pronounced.

Without the cutoff, these interactions can modify the topology of the ground-state density and lead to configurations that are absent when the cutoff is applied.

### Spurious symmetry breaking

For softened cylindrical trapping potentials, simulations without the cutoff can exhibit explicit breaking of the rotational symmetry of the trap.

For example, in the droplet regime, the number and arrangement of droplets can differ between simulations with and without the cutoff. These effects are numerical artifacts rather than consequences of the underlying rotationally symmetric trapping potential.

### Edge effects and droplet formation

The cutoff also affects the behavior of droplets near the boundary of the condensate. In some regimes, the formation of the droplet ring is delayed without the cutoff, and the resulting edge fraction differs from the cutoff calculation.

These effects demonstrate that periodic-image interactions can influence not only the phase transition itself but also the detailed structure of the resulting ground state.

### Harmonic trapping potential

For (N=5\times10^5) atoms in a harmonic trap, increasing (\epsilon_{dd}) produces a regime in which the trap volume becomes populated by droplets.

At sufficiently large interaction strength, the droplet densities become increasingly homogeneous throughout the condensate. The simulations also reveal rotational-symmetry-breaking configurations when periodic-image interactions are not suppressed.

## Computational Scale

The simulations were performed on three-dimensional grids with physical dimensions of approximately

[
44\times44\times22,\mu\mathrm{m}^3
]

and particle numbers ranging from approximately

[
10^5 \quad\text{to}\quad 5\times10^5.
]

GPU acceleration was essential for performing the large parameter studies required to investigate convergence, phase transitions, and cutoff effects.

## Repository Structure

```text
.
├── MyHeaders/
│   ├── gradient_descent.hpp
│   ├── gradient_descent.ipp
│   ├── MyTools.hpp
│   └── ...
├── Own_program_vault/
│   ├── main_gradient_descent.cpp
│   ├── InputParser.py
│   ├── parameters.prm
│   └── analysis notebooks
├── Results/
│   └── selected simulation results
├── thesis/
│   └── Bachelorarbeit_Mark_Salzmann.pdf
├── ultracold-dipolar/
│   └── UltraCold simulation framework
└── README.md
```

The repository contains both the C++/CUDA simulation code and Python/Jupyter tools used for post-processing, data collection, and visualization.

## Dependencies

The main dependencies are:

* C++
* CUDA
* CUDA FFT (`cuFFT`)
* CMake
* Python
* Jupyter / NumPy / SciPy / Matplotlib
* [UltraCold](https://github.com/)

The simulations were developed and tested in a Linux/HPC environment with NVIDIA GPUs.

## Reproducibility

The simulation code depends on the UltraCold framework contained in the `ultracold-dipolar` directory.

A typical workflow consists of:

1. Building the UltraCold library and the custom simulation code with CMake.
2. Configuring the physical and numerical parameters.
3. Running the GPU-accelerated gradient-descent simulation.
4. Processing the resulting density data with the included Python/Jupyter tools.
5. Comparing converged ground states for different interaction strengths, trapping potentials, and cutoff configurations.

Detailed numerical parameters and methodological choices are documented in the accompanying thesis.

Because the simulations were designed primarily for GPU-based HPC execution, reproducing the full parameter studies requires access to a suitable NVIDIA GPU environment.

## Scientific Context

The project investigates numerical effects that arise when long-range interactions are evaluated using Fourier methods in finite simulation domains.

The central observation is that periodic boundary conditions, while computationally convenient, can introduce interactions between the physical condensate and its periodic replicas. For dipolar systems, these interactions can become sufficiently strong to alter observable properties of the numerically obtained ground state.

The results therefore highlight the importance of carefully controlling finite-size and periodic-boundary effects when performing numerical simulations of long-range interacting quantum systems.

## References

**Bachelor thesis**

M. Salzmann, *[Title of Bachelor's Thesis]*, Heidelberg University, 2025.

See [`thesis/Bachelorarbeit_Mark_Salzmann.pdf`](thesis/Bachelorarbeit_Mark_Salzmann.pdf).

**Relevant literature**

S. M. Roccuzzo et al., study of supersolid phases in dipolar Bose-Einstein condensates.

Additional references and the complete bibliography are provided in the accompanying thesis.
