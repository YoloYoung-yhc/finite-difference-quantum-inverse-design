"""Study the lowest states and energy splitting of a symmetric double well."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from finite_difference_solver import solve_schrodinger

# Define physical constants and parameters
hbar = 1.0  # Reduced Planck's constant
m = 1.0     # Mass of the particle
L = 20.0    # Length of the computational domain
N = 811     # Default grid puts the walls at +/-a and +/-b between grid points
a = 5.0     # Outer half-width of the two wells together
b = 1.0     # Half-width of the central barrier (0 < b < a)
V0 = 50.0   # Potential outside the wells
V1 = 1.0    # Height of the central barrier; well bottoms remain zero
if hbar <= 0 or m <= 0 or V0 <= 0 or V1 < 0 or not 0 < b < a < L / 2 or N < 5:
    raise ValueError("Require hbar, m, V0 > 0, V1 >= 0, 0 < b < a < L/2 and N >= 5.")
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

# Add a finite central barrier to the finite square well.
# Wells occupy (-a, -b) and (b, a); the central barrier has width 2*b.
V = np.where(np.abs(x) <= a, 0.0, V0)
V[np.abs(x) <= b] = V1

# Place zero-valued domain endpoints far from bound-state tails.
# The solver handles discretization, normalization, and zero endpoints.
eigenvalues, wavefunctions = solve_schrodinger(x, V, mass=m, hbar=hbar)
# The exterior potential V0 sets the bound-state threshold.
bound = eigenvalues < V0
print("Numerical bound-state candidates (E < V0):", np.count_nonzero(bound))
if np.count_nonzero(bound) < 2:
    raise ValueError("Fewer than two bound-state candidates; adjust the potential or grid.")
E = eigenvalues[bound][:5]
n = np.arange(len(E))  # n=0 is the ground state
print(f"{'n':>3} {'Numerical energy':>18}")
for quantum_number, numerical in zip(n, E):
    print(f"{quantum_number:3d} {numerical:18.10f}")
E0 = eigenvalues[0]
E1 = eigenvalues[1]
delta_E = E1 - E0
print(f"Energy splitting E1 - E0: {delta_E:.10e}")
print(f"Both lowest states below central barrier: {E1 < V1}")

# Extract the first two normalized states with zero endpoints.
psi0 = wavefunctions[:, 0].copy()
psi1 = wavefunctions[:, 1].copy()
# Choose positive peak amplitude on the right without changing the nodes.
idx = np.flatnonzero(x > 0)
if psi0[idx[np.argmax(np.abs(psi0[idx]))]] < 0:
    psi0 = -psi0
if psi1[idx[np.argmax(np.abs(psi1[idx]))]] < 0:
    psi1 = -psi1

density = np.abs(psi0)**2
density1 = np.abs(psi1)**2
normalization = np.sum(density) * dx
normalization1 = np.sum(density1) * dx
overlap = np.sum(psi0 * psi1) * dx
# For this symmetric potential, the lowest states are even and odd.
error_even = np.sqrt(np.sum(np.abs(psi0 - psi0[::-1])**2) * dx)
error_odd = np.sqrt(np.sum(np.abs(psi1 + psi1[::-1])**2) * dx)
print(f"Ground-state normalization: {normalization:.12f}")
print(f"First-excited-state normalization: {normalization1:.12f}")
print(f"Overlap between the two states: {overlap:.6e}")
print(f"Ground-state even-parity error: {error_even:.6e}")
print(f"First-excited-state odd-parity error: {error_odd:.6e}")
print(f"Ground-state probability in central barrier: {np.sum(density[np.abs(x) <= b])*dx:.6e}")
print(f"First-excited-state probability in central barrier: {np.sum(density1[np.abs(x) <= b])*dx:.6e}")
# Large barriers can yield numerically unresolved, mixed near-degenerate states.
# Check convergence and parity before interpreting a tiny energy splitting.


figure_directory = Path(__file__).resolve().parent / 'figure'
figure_directory.mkdir(parents=True, exist_ok=True)

# Reuse the plotting loop to create four independent figures.
for values, ylabel, title in [
    (psi0, 'Wavefunction', 'Ground State Wavefunction'),
    (psi1, 'Wavefunction', 'First Excited State Wavefunction'),
    (density, 'Probability density', 'Ground State Probability Density'),
    (density1, 'Probability density', 'First Excited State Probability Density'),
]:
    plt.figure(figsize=(10, 6))
    plt.plot(x, values, color='blue', label='Numerical')
    for wall in [-a, -b, b, a]:
        plt.axvline(wall, color='gray', linestyle=':', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel(ylabel)
    plt.title('Double Square Well: ' + title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = 'double_square_well_' + title.lower().replace(' ', '_') + '.png'
    plt.savefig(figure_directory / filename, dpi=300, bbox_inches='tight')
plt.show()
