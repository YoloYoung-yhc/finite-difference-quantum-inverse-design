"""Compare a finite square well with boundary-matching reference solutions."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from finite_difference_solver import solve_schrodinger

# Define physical constants and parameters
hbar = 1.0  # Reduced Planck's constant
m = 1.0     # Mass of the particle
L = 20.0    # Length of the computational domain (not the well width)
N = 803     # With L=20 and a=5, N=4*k+3 places each wall between grid points
a = 5.0     # Half-width of the well
V0 = 50.0   # Potential outside the well; potential inside is zero
if hbar <= 0 or m <= 0 or V0 <= 0 or not 0 < a < L / 2 or N < 5:
    raise ValueError("Require hbar, m, V0 > 0, 0 < a < L/2 and N >= 5.")
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

V = np.where(np.abs(x) <= a, 0.0, V0)
# Place zero-valued domain endpoints far from bound-state tails.
# The solver handles discretization, normalization, and zero endpoints.
eigenvalues, wavefunctions = solve_schrodinger(x, V, mass=m, hbar=hbar)
# The exterior potential V0 sets the bound-state threshold.
bound = eigenvalues < V0
print("Numerical bound-state candidates (E < V0):", np.count_nonzero(bound))
if not np.any(bound):
    raise ValueError("No bound-state candidates resolved; check domain and grid.")

# Copy the normalized state so sign alignment does not mutate the output.
psi0 = wavefunctions[:, 0].copy()
if psi0[np.argmax(np.abs(psi0))] < 0:
    psi0 = -psi0  # Choose a consistent overall sign
E0 = eigenvalues[0]
print("Ground state energy:", E0)

# Independent reference from continuity of the wavefunction and derivative:
# Even states: k*tan(k*a)=kappa; odd states: -k*cot(k*a)=kappa.
# With z=k*a and z0=a*sqrt(2*m*V0)/hbar, combine both conditions as
# z + arcsin(z/z0) = n*pi/2, where n=1,2,... indexes the states.
# The left side is monotonic without poles, so bisection needs no SciPy.
z0 = a * np.sqrt(2 * m * V0) / hbar
E_reference_all = []
n_reference = 1
while n_reference * np.pi / 2 < z0 + np.pi / 2:
    target = n_reference * np.pi / 2
    lower, upper = 0.0, z0
    for _ in range(80):
        z = (lower + upper) / 2
        if z + np.arcsin(z / z0) < target:
            lower = z
        else:
            upper = z
    z = (lower + upper) / 2
    E_reference_all.append(hbar**2 * z**2 / (2 * m * a**2))
    n_reference += 1
E_reference_all = np.array(E_reference_all)
print("Reference bound-state count on the infinite domain:", len(E_reference_all))
# Near-threshold candidates need separate grid and domain checks.
count = min(5, np.count_nonzero(bound), len(E_reference_all))
n = np.arange(1, count + 1)
E = eigenvalues[bound][:count]
E_reference = E_reference_all[:count]
error = np.abs(E - E_reference) / np.abs(E_reference)
print(f"{'n':>3} {'Numerical energy':>18} {'Reference energy':>18} {'Relative error':>18}")
for quantum_number, numerical, reference, relative_error in zip(n, E, E_reference, error):
    print(f"{quantum_number:3d} {numerical:18.10f} {reference:18.10f} {relative_error:18.6e}")

# The ground state is a cosine inside, joined to decaying exterior tails.
# Normalize over the full real axis; do not truncate the reference at endpoints.
k = np.sqrt(2 * m * E_reference[0]) / hbar
kappa = np.sqrt(2 * m * (V0 - E_reference[0])) / hbar
A = 1 / np.sqrt(a + np.sin(2 * k * a) / (2 * k) + np.cos(k * a)**2 / kappa)
psi0_reference = A * np.where(
    np.abs(x) <= a,
    np.cos(k * x),
    np.cos(k * a) * np.exp(-kappa * np.maximum(np.abs(x) - a, 0)),
)
density = np.abs(psi0)**2
density_reference = np.abs(psi0_reference)**2
normalization = np.sum(density) * dx
probability_outside = np.sum(density[np.abs(x) > a]) * dx
probability_outside_reference = A**2 * np.cos(k * a)**2 / kappa
density_max_error = np.max(np.abs(density - density_reference))
print(f"Grid spacing: {dx:.6f}; reference ground-state decay length: {1/kappa:.6f}")
print(f"Numerical normalization: {normalization:.12f}")
print(f"Reference normalization on this grid: {np.sum(density_reference)*dx:.12f}")
print(f"Probability outside well: {probability_outside:.6e}")
print(f"Reference probability outside well (infinite domain): {probability_outside_reference:.6e}")
print(f"Maximum absolute probability density error: {density_max_error:.6e}")

figure_directory = Path(__file__).resolve().parent / 'figure'
figure_directory.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(n, error, 'o-', label='Relative Error')
plt.xlabel('State number n (ground state = 1)')
plt.ylabel('Relative energy error')
plt.title('Finite Square Well: Bound-State Energy Errors')
plt.xticks(n)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(figure_directory / 'finite_square_well_energy_errors.png', dpi=300, bbox_inches='tight')

# Use a separate figure for each quantity.

plt.figure(figsize=(10, 6))
plt.plot(x, psi0, color='red', label='Numerical')
plt.plot(x, psi0_reference, color='blue', linestyle='--', label='Reference')
plt.xlabel('x')
plt.ylabel('Wavefunction')
plt.title('Finite Square Well: Ground State Wavefunction')
plt.axvline(-a, color='gray', linestyle=':', alpha=0.7)
plt.axvline(a, color='gray', linestyle=':', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(figure_directory / 'finite_square_well_ground_state_wavefunction.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(x, density, color='red', label='Numerical')
plt.plot(x, density_reference, color='blue', linestyle='--', label='Reference')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('Finite Square Well: Ground State Probability Density')
plt.axvline(-a, color='gray', linestyle=':', alpha=0.7)
plt.axvline(a, color='gray', linestyle=':', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(figure_directory / 'finite_square_well_ground_state_density.png', dpi=300, bbox_inches='tight')
plt.show()
