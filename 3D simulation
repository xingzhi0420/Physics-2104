import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# Constants
k = 8.9875e9
Z = 4
q_e = 1.6e-19
m_e = 9.11e-31
N = 4          # Number of electrons
R_init = 1e-9  # initial radial distance (m)

# Time settings
t_span = (0, 1e-14)          # simulation time
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Regular tetrahedron coordinates (unscaled)
v = np.array([
    [1,  1,  1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1,-1,  1]
], dtype=float)

# Normalize to unit vectors
v = v / np.sqrt(3.0)

# Scale by R_init
positions = v * R_init

# Initial velocities
angular_velocity = 1e15
velocities = np.cross(positions, np.array([0, 0, angular_velocity]))

def compute_forces(positions):
    forces = np.zeros_like(positions)
    for i in range(N):
        r_i = np.linalg.norm(positions[i])
        forces[i] -= (k * Z * q_e**2 / r_i**3) * positions[i]
        for j in range(N):
            if i != j:
                r_ij = positions[i] - positions[j]
                dist_ij = np.linalg.norm(r_ij)
                forces[i] += (k * q_e**2 / dist_ij**3) * r_ij
    return forces

def equations_of_motion(t, y):
    pos = y[:3*N].reshape((N, 3))
    vel = y[3*N:].reshape((N, 3))
    forces = compute_forces(pos)
    acc = forces / m_e
    dydt = np.concatenate([vel.flatten(), acc.flatten()])
    return dydt

# Initial conditions
y0 = np.concatenate([positions.flatten(), velocities.flatten()])

# Solve ODEs
solution = solve_ivp(
    equations_of_motion,
    t_span,
    y0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-9,
    atol=1e-12
)


positions_sol = solution.y[:3*N].reshape((N, 3, -1))

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for i in range(N):
    ax.plot(
        positions_sol[i, 0, :],
        positions_sol[i, 1, :],
        positions_sol[i, 2, :],
        label=f'Electron {i+1}'
    )

# Nucleus at origin
ax.scatter(0, 0, 0, color='red', s=100, label='Nucleus')

ax.set_title('3D Trajectories of Electrons with Electron-Electron Interactions')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.legend()
plt.show()
