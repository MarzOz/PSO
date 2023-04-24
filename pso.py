'''
    Particle Swarm Optimization(PSO) Implementation
    @author: Lawrence Chang Chien
    Last Modified: 2023/04/13
'''
# using PSO to solve the problem of the function
# f(x,y)=x^2-10cos(2*pi*x)+y^2-10cos(2*pi*y)+20, x,y in [-5,5]
# and visualize the process of the algorithm

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser()
parser.add_argument('--n_particles', type=int, default=40, help='number of particles')
parser.add_argument('--n_dimensions', type=int, default=2, help='number of dimensions')
parser.add_argument('--n_iter', type=int, default=50, help='number of iterations')
parser.add_argument('--up_bound', type=float, default=5, help='upper bound of the search space')
parser.add_argument('--low_bound', type=float,default=-5, help='lower bound of the search space')
parser.add_argument('--w', type=float, default=0.8, help='inertia weight')
parser.add_argument('--c1', type=float, default=0.2, help='cognitive weight')
parser.add_argument('--c2', type=float, default=0.2, help='social weight')
parser.add_argument('--v_max', type=float, default=2, help='max velocity')

args = parser.parse_args()

# define the objective function
def f(x, y):
    return x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y) + 20

# compute and plot the function in 3D within [-5,5]x[-5,5]
x = np.arange(args.low_bound , args.up_bound, 0.1)
y = np.arange(args.low_bound , args.up_bound, 0.1)
x, y = np.meshgrid(x, y)
z = f(x, y)

# Find the global minimum
x_min = x.flatten()[z.argmin()]
y_min = y.flatten()[z.argmin()]

# initialize particles
X = np.random.uniform(low=-5, high=5, size=(args.n_dimensions, args.n_particles))
V = np.random.rand(args.n_dimensions, args.n_particles) * 0.1
pbest = X.copy()
pbest_obj = f(X[0], X[1])
gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()

# initialize the figure
fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(z, extent=[-5,5,-5,5], origin='lower', cmap='jet', alpha=0.5)
fig.colorbar(img, ax=ax)
ax.plot(x_min, y_min, marker='*', markersize=10, color='white')
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.5)
ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5)
p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='x', color='red', s=100, alpha=0.4)
ax.set_xlabel('x')
ax.set_xlim(-5, 5)
ax.set_ylabel('y')
ax.set_ylim(-5, 5)

# Function to update the particles location and velocity
def update_location():
    global X, V, pbest, pbest_obj, gbest, gbest_obj, conv

    # Updates params
    r1, r2 = np.random.rand(2)
    V = args.w * V + args.c1 * r1 * (pbest - X) + args.c2 * r2 * (gbest.reshape(-1,1) - X)
    V = np.clip(V, -args.v_max, args.v_max)
    X = X + V
    X = np.clip(X, args.low_bound, args.up_bound)
    obj = f(X[0], X[1])
    pbest[:, (obj < pbest_obj)] = X[:, (obj < pbest_obj)]
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()
    
# Function to update the plot
def animate(i):
    ax.set_title('PSO Iteration: {:02d}'.format(i))
    # Update particles params
    update_location()
    # Update plot
    pbest_plot.set_offsets(pbest.T)
    p_plot.set_offsets(X.T)
    p_arrow.set_offsets(X.T)
    p_arrow.set_UVC(V[0], V[1])
    gbest_plot.set_offsets(gbest.reshape(1, -1))
    conv.append(gbest_obj) if len(conv) < args.n_iter else None

    return ax, pbest_plot, p_plot, p_arrow, gbest_plot

# Initialize the convergence record argument
conv = []

# Run the animation
anim = FuncAnimation(fig, animate, frames=np.arange(1, args.n_iter), interval=500, blit=False, repeat=True)
anim.save('PSO.gif', writer='pillow', dpi=120, fps=60)
plt.show()

# convergence plot
plt.clf()
plt.plot(np.arange(1, args.n_iter + 1), conv)
plt.xlabel('Iteration')
plt.ylabel('Fitness value')
plt.title('PSO Convergence')
plt.savefig('PSO_conv.png')


print('PSO found best solution at f({}) = {}'.format(gbest, gbest_obj))
print('Global minimum is at f({}) = {}'.format([x_min, y_min], f(x_min, y_min)))
