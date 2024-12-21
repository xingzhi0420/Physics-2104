import numpy as np
import matplotlib.pyplot as plt


k = 8.99e9
e = 1.6e-19
hbar = 1.054e-34
L = 1  # Angular momentum
Z = 6  # Nuclear charge
m = 9.109e-31

# Quantum numbers and electron count
n = 2  # Principal quantum number
N = 4  # Number of electrons for symmetry

theta = np.linspace(0, 2 * np.pi, 500)  # Angular range

# Function to compute effective coefficient
def compute_alpha(N, Z):
    return -Z + (N**2 - 1) / 12

# Function to compute R
def R_realistic_updated(theta, n, L, k, e, alpha, hbar, m):
    denominator = (
        -(mke2_alpha := (m * k * e**2 * alpha) / (L**2 * hbar**2))
        - mke2_alpha * np.sqrt(1 - (L**2) / (n**2)) * np.cos(theta)
    )
    return 1 / denominator

# Compute alpha
alpha = compute_alpha(N, Z)

# Plot all N electrons
plt.figure(figsize=(8, 8))
for j in range(N):
    theta_shifted = theta + 2 * np.pi * j / N
    R_values = R_realistic_updated(theta_shifted, n, L, k, e, alpha, hbar, m)
    x = R_values * np.cos(theta)
    y = R_values * np.sin(theta)
    plt.plot(x, y, label=f"Electron {j + 1}")


plt.scatter(0, 0, color='red', label='Nucleus')


plt.gca().set_aspect('equal', adjustable='box')
plt.title(f"All {N} Electrons (Z={Z}, n={n}, L={L}, α={alpha:.3f})")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.grid()
plt.show()
