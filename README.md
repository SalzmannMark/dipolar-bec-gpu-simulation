# GPU-Accelerated Simulation of Dipolar Bose–Einstein Condensates

**Bachelor's thesis research project** | C++ · CUDA · FFT-based numerical methods · Python · HPC

Numerical study of finite-domain artifacts in three-dimensional dipolar Bose–Einstein condensates (BECs). This project extends the [UltraCold](https://github.com/smroccuzzo/UltraCold) C++/CUDA framework with a GPU-accelerated workflow for computing ground states of the extended Gross–Pitaevskii equation (eGPE), analysing phase transitions, and suppressing unphysical periodic-image interactions through a dipolar-interaction cutoff.

It was developed for my 2025 B.Sc. Physics thesis at Heidelberg University, supervised at the Kirchhoff Institute for Physics by Prof. Dr. Thomas Gasenzer.

> **Research question:** How strongly do FFT-imposed periodic boundary conditions bias calculated ground states of long-range interacting quantum systems, and how effectively does a real-space cutoff remove that bias?

## Highlights

- Implemented and evaluated a finite-range cutoff for the long-range dipole–dipole interaction in an FFT-based 3D solver.
- Ran GPU-accelerated parameter studies for $^{164}\mathrm{Dy}$ condensates with $10^5$–$5\times10^5$ atoms on Heidelberg University's EINC cluster.
- Computed ground states through constrained energy minimisation, and post-processed density fields with Python and Jupyter notebooks.
- Identified numerical artifacts from periodic replicas, including shifts in the apparent transition point, altered droplet topology, and spurious rotational-symmetry breaking.
- Compared cutoff and unmodified calculations across cylindrical, softened box-like, and harmonic traps.

## Key result

For a cylindrical box trap with $N=10^5$ atoms, the estimated transition interval changed from

| Dipolar-interaction treatment | Estimated critical interval for $\varepsilon_{dd}$ |
| --- | ---: |
| No cutoff | $1.41 < \varepsilon_{dd} < 1.425$ |
| Cutoff applied | $1.404 < \varepsilon_{dd} < 1.41$ |

The cutoff result is consistent with the approximately $\varepsilon_{dd}=1.40$ value reported in the literature. At larger particle numbers, the effect becomes more pronounced because the condensate reaches closer to the numerical boundary.

![Superfluid fraction versus relative dipolar interaction strength.](Results/plots/rocuzzo_superfluid_fraction.png)

*Finite-range cutoff shifts the observed transition interval in the box-trap calculation. The density-modulated phase is signalled by the drop in superfluid fraction.*

| Cutoff applied | No cutoff |
| --- | --- |
| [![Real-space density with cutoff.](Results/simulations/x6_mit_N500k.png)](Results/simulations/x6_mit_N500k.png) | [![Real-space density without cutoff.](Results/simulations/x6_ohne_N500k.png)](Results/simulations/x6_ohne_N500k.png) |

*Real-space densities in a softened $r^6$ trap at $N=5\times10^5$. Open either figure for the full-resolution parameter sweep.*

## Methods

The model is the three-dimensional eGPE for a dipolar $^{164}\mathrm{Dy}$ condensate. It includes kinetic energy, external confinement, contact interactions, dipole–dipole interactions, and the Lee–Huang–Yang correction.

### Physical model

The condensate is represented by a complex order parameter $\psi(\mathbf r,t)$, whose density is $n(\mathbf r,t)=|\psi(\mathbf r,t)|^2$. Its evolution is described by the eGPE

$$
i\hbar \frac{\partial \psi}{\partial t} =
\left[
  -\frac{\hbar^2}{2m}\nabla^2
  + V_\mathrm{ext}
  + g|\psi|^2
  + \Phi_\mathrm{dd}
  + \gamma_\mathrm{QF}|\psi|^3
\right]\psi.
$$

Here, $g=4\pi\hbar^2a_s/m$ is the contact-interaction coupling, $V_\mathrm{ext}$ describes the trap, and $\gamma_\mathrm{QF}|\psi|^3$ is the local Lee–Huang–Yang (LHY) correction. The LHY term captures the leading contribution from quantum fluctuations and provides a stabilising repulsion that is essential for finite-density droplet states.

The non-local dipolar mean field is a convolution,

$$
\Phi_\mathrm{dd}(\mathbf r) =
\int V_\mathrm{dd}(\mathbf r-\mathbf r')\,n(\mathbf r')\,d^3r',
\qquad
V_\mathrm{dd}(\mathbf r) =
\frac{C_\mathrm{dd}}{4\pi}
\frac{1-3\cos^2\theta}{r^3}.
$$

For magnetically polarised $^{164}\mathrm{Dy}$, $C_\mathrm{dd}=\mu_0\mu^2$ and the competition between dipolar and contact interactions is expressed by $\varepsilon_\mathrm{dd}=a_\mathrm{dd}/a_s$. Its anisotropy makes the interaction repulsive side-by-side and attractive head-to-tail; tuning $\varepsilon_\mathrm{dd}$ therefore drives the transition between smooth condensates and density-modulated droplet or supersolid configurations.

### Spectral evaluation and cutoff

The convolution is evaluated efficiently with the convolution theorem:

$$
\Phi_\mathrm{dd} =
\mathcal F^{-1}\!\left[
  \widetilde V_\mathrm{dd}(\mathbf k)\,
  \widetilde n(\mathbf k)
\right].
$$

This reduces a non-local $3$D operation to pointwise multiplication in momentum space and is the reason GPU-accelerated FFTs are central to the workflow. The trade-off is that a discrete FFT evaluates a *periodic* convolution: the condensate is implicitly tiled beyond the simulation cell. Since $V_\mathrm{dd}\propto r^{-3}$, those replicas can still contribute appreciably near the boundary.

The cutoff calculations truncate the dipolar kernel to a finite interaction region contained in the numerical domain before its Fourier-space evaluation. Comparing otherwise matched cutoff and unmodified runs isolates periodic-image effects from physical changes caused by trap geometry or interaction strength.

| Component | Approach |
| --- | --- |
| Ground-state calculation | Constrained gradient descent with wavefunction renormalisation; heavy-ball acceleration available |
| Long-range interaction | Fourier-space convolution using CUDA/cuFFT |
| Boundary-artifact control | Finite real-space cutoff for the dipolar kernel |
| Dynamics | Split-step Fourier / operator-splitting propagation utilities |
| Parameter exploration | Sweeps over interaction strength, atom number, trap geometry, and cutoff configuration |
| Analysis | Python/Jupyter post-processing for radial data, 2D/3D densities, and comparative plots |

The numerical motivation is broadly relevant to computational physics and scientific ML: spectral methods make non-local operators tractable, but their implicit boundary conditions must be validated to prevent physically incorrect results.

## Technical contribution

My project-specific work is concentrated in the simulation and analysis layer:

- `MyHeaders/gradient_descent.hpp` and `MyHeaders/gradient_descent.ipp` — parameterised ground-state solver, trap construction, cutoff handling, and parameter-database integration.
- `MyHeaders/MyTools.hpp` — shared numerical and simulation utilities.
- `Own_program_vault/main_gradient_descent.cpp` — experiment configuration and batch-run entry point.
- `Own_program_vault/*.ipynb` — analysis pipelines for density data and plots.
- `MyHeaders/Omega_database/` — calibrated trap-frequency data used by the simulations.

The GPU numerical infrastructure is provided by UltraCold; this repository contains the research extensions, parameter sets, analysis notebooks, selected outputs, and the full thesis.

## Repository layout

```text
.
├── MyHeaders/                 # C++ headers for solver, traps, utilities, and parameters
│   └── Omega_database/        # Trap-frequency calibration data
├── Own_program_vault/         # C++ run configuration, Python helpers, Jupyter analysis
├── Results/
│   ├── plots/                 # Selected figures
│   └── simulations/           # Representative density outputs
├── thesis/
│   └── Bachelorarbeit_Mark_Salzmann.pdf
├── ultracold-dipolar/         # UltraCold framework snapshot / repository dependency
└── README.md
```

## Getting started

This is an **archived research-code repository**, not a packaged end-user application. The original calculations were run in a Linux HPC environment with NVIDIA GPUs. Reproducing a full study therefore requires an NVIDIA CUDA toolchain and a compatible build of UltraCold.

### Prerequisites

- Linux
- C++ compiler with C++17 support
- NVIDIA GPU, CUDA Toolkit, and cuFFT
- CMake
- Python 3 with NumPy, SciPy, Matplotlib, and Jupyter (for analysis)
- [UltraCold](https://github.com/smroccuzzo/UltraCold) and its documented dependencies

### Reproduction workflow

1. Obtain a CUDA-enabled UltraCold build following the [official UltraCold documentation](https://smroccuzzo.github.io/UltraCold/html/index.html).
2. Add the project headers and compile the simulation entry point against that build. The original run configuration is in `Own_program_vault/main_gradient_descent.cpp`.
3. Select a parameter set from `MyHeaders/` and configure the trap, atom number, $\varepsilon_{dd}$, grid, and cutoff mode.
4. Run matched calculations with and without the cutoff.
5. Analyse output density fields with the notebooks in `Own_program_vault/` and compare the converged states.

The tracked `ultracold-dipolar` entry is a framework dependency from the original research environment. If it is empty after cloning, use the upstream UltraCold project linked above and adapt the local build paths to your installation.

## Results and limitations

The results show that FFT periodicity can change both quantitative and qualitative conclusions in long-range dipolar simulations. In particular, periodic replicas can shift estimated critical interaction strengths and induce density configurations incompatible with the rotational symmetry of the physical trap.

This repository preserves the code and selected data used in the thesis. It has not been refactored into a general-purpose, versioned simulation package; paths and build integration may require adjustment for a new system. The thesis is the authoritative description of numerical settings, convergence checks, and interpretation of results.

## Thesis and citation

Read the complete thesis: [*Effects of long-range interaction cutoffs in trapped dipolar Bose–Einstein condensates* (PDF)](thesis/Bachelorarbeit_Mark_Salzmann.pdf)

If you use or discuss this work, please cite:

```bibtex
@thesis{salzmann2025cutoffs,
  author = {Mark Salzmann},
  title = {Effects of long-range interaction cutoffs in trapped dipolar Bose--Einstein condensates},
  school = {Heidelberg University},
  year = {2025},
  type = {Bachelor's thesis}
}
```

## Acknowledgements

This work builds on [UltraCold](https://github.com/smroccuzzo/UltraCold), developed by S. M. Roccuzzo and collaborators. The research was conducted at Heidelberg University with computing support from the EINC GPU cluster.

## Contact

For questions about the research or repository, please open a GitHub issue or contact [Mark Salzmann](https://github.com/SalzmannMark).
