import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Given Z, N
Z = 4
N = 4

alpha = -Z + (N**2 - 1)/12

# Constants
k = 1.0
e_charge = 1.0
m = 1.0

# Angular momentum L
L = 2.0

# Initial conditions for Lagrangian approach
R0 = 1.0
R_dot0 = 0.1
theta0 = 0.0

def lagrangian_ode(t, y):
    R, R_dot, theta = y
    theta_dot = L/(m*R**2)

    R_ddot = R * theta_dot**2 + (k * e_charge**2 * alpha) / (m * R**2)
    return [R_dot, R_ddot, theta_dot]

t_span = [0, 20]
t_eval = np.linspace(0, 20, 1000)
y_init = [R0, R_dot0, theta0]

# Solve Lagrangian-based ODE for the reference electron (j=0)
sol = solve_ivp(lagrangian_ode, t_span, y_init, t_eval=t_eval, method='RK45')
R_lagr = sol.y[0]
theta_lagr = sol.y[2]

plt.figure(figsize=(8,8))
for j in range(N):

    theta_j = theta_lagr + 2*np.pi*j/N
    x_j = R_lagr * np.cos(theta_j)
    y_j = R_lagr * np.sin(theta_j)
    plt.plot(x_j, y_j, label=f"Electron {j}")

plt.scatter(0, 0, color="red", s=100, label="Nucleus")
plt.title(f"Lagrangian - All {N} Electrons (Z={Z}, α={alpha:.3f})")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.legend()
plt.grid()
plt.show()
