"""Compare an infinite square well with its analytical solution."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from finite_difference_solver import solve_schrodinger

# Define physical constants and parameters
hbar = 1.0  # Reduced Planck's constant
m = 1.0     # Mass of the particle
L = 20.0    # Length of the spatial domain
N = 200  # Number of spatial points
x = np.linspace(-L/2, L/2, N)  # Spatial grid
dx = x[1] - x[0]  # Spatial step size

V = np.zeros(N)  # Potential energy (infinite square well)

# The solver handles finite differences, zero boundaries, and normalization.
eigenvalues, wavefunctions = solve_schrodinger(x, V, mass=m, hbar=hbar)
psi0 = wavefunctions[:, 0].copy()
E0 = eigenvalues[0]  # Ground state energy
print("Ground state energy:", E0)
# Extract the first five energy levels.
E = eigenvalues[:5]
# Compare with the analytical solution.
# Compare energy levels.
n = np.array([1, 2, 3, 4, 5])  # Quantum numbers for the first five energy levels
E_analytical = n**2 * np.pi**2 * hbar**2 / (2 * m * L**2)
error = np.abs(E - E_analytical) / np.abs(E_analytical)
print(f"{'n':>3} {'Numerical energy':>18} {'Analytical energy':>18} {'Relative error':>18}")
for quantum_number, numerical, analytical, relative_error in zip(n, E, E_analytical, error):
    print(f"{quantum_number:3d} {numerical:18.10f} {analytical:18.10f} {relative_error:18.6e}")
figure_directory = Path(__file__).resolve().parent / 'figure'
figure_directory.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(n, error, 'o-', label='Relative Error')
plt.xlabel('Quantum number n')
plt.ylabel('Relative energy error')
plt.title('Infinite Square Well: Energy Errors')
plt.xticks(n)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(figure_directory / 'infinite_square_well_energy_errors.png', dpi=300, bbox_inches='tight')
# Compare the ground-state wavefunction and probability density.
psi0_analytical = np.sqrt(2 / L) * np.sin(np.pi * (x + L/2) / L)
# Align the arbitrary eigenvector sign using overlap with the analytical state.
overlap = np.sum(psi0 * psi0_analytical) * dx
if overlap < 0:
    psi0 = -psi0

density = np.abs(psi0)**2
density_analytical = np.abs(psi0_analytical)**2
normalization = np.sum(density) * dx
analytical_normalization = np.sum(density_analytical) * dx
density_max_error = np.max(np.abs(density - density_analytical))
print(f"Numerical normalization: {normalization:.12f}")
print(f"Analytical normalization on this grid: {analytical_normalization:.12f}")
print(f"Maximum absolute probability density error: {density_max_error:.6e}")

plt.figure(figsize=(10, 6))
plt.plot(x, psi0, color='red', label='Numerical')
plt.plot(x, psi0_analytical, color='blue', label='Analytical', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('Wavefunction')
plt.title('Infinite Square Well: Ground State Wavefunction')
plt.legend()
plt.tight_layout()
plt.savefig(figure_directory / 'infinite_square_well_ground_state_wavefunction.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(x, density, color='red', label='Numerical')
plt.plot(x, density_analytical, color='blue', label='Analytical', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Infinite Square Well: Ground State Probability Density')
plt.legend()
plt.tight_layout()
plt.savefig(figure_directory / 'infinite_square_well_ground_state_density.png', dpi=300, bbox_inches='tight')
plt.show()
