import io
import imageio.v2 as imageio
import os
import numpy as np
import gmsh
import pygimli as pg
from tqdm import tqdm
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import threading
import csv

# Read settings from settings.txt
def read_settings():
    with open('settings.txt', 'r') as file:
        lines = file.readlines()
        settings = {}
        for line in lines:
            key, value = line.strip().split('=')
            settings[key.strip()] = value.strip()
        return settings

settings = read_settings()
CAD_path = settings.get('CAD_path')
gpu_rendering = settings.get('gpu_rendering', 'False') == 'True'
multithreading = settings.get('multithreading', 'False') == 'True'

# Load fluid velocity from fluid_velocity.csv
def load_fluid_velocity():
    velocities = []
    with open('fluid_velocity.csv', mode='r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            velocity = float(row[0])
            unit = int(row[1])
            # Convert velocity to m/s based on unit
            if unit == 2:  # ft/s
                velocity *= 0.3048
            elif unit == 3:  # mph
                velocity *= 0.44704
            elif unit == 4:  # km/h
                velocity *= 0.277778
            elif unit == 5:  # knots
                velocity *= 0.514444
            velocities.append(velocity)
    return velocities

fluid_velocities = load_fluid_velocity()

# Initialize gmsh and load the CAD file
gmsh.initialize()
gmsh.open(CAD_path)
gmsh.model.mesh.generate(3)
nodes = gmsh.model.mesh.getNodes()
elements = gmsh.model.mesh.getElements()

# Convert CAD mesh to pyGIMLi mesh
mesh = pg.Mesh()
for node in nodes[1]:
    mesh.createNode(pg.RVector3(node[0], node[1], node[2]))

for element in elements[2]:
    mesh.createTriangleFace(element[0], element[1], element[2])

# Simulation parameters
Nt = 1000  # Number of time steps
tau = 1.0  # Relaxation time
idxs = list(range(9))  # Example indices for 9 velocity directions
cxs = [1, 1, 0, -1, -1, -1, 0, 1]  # Velocity in x-direction
cys = [0, 1, 1, 1, 0, -1, -1, -1]  # Velocity in y-direction
weights = [4/9] + [1/9]*4 + [1/36]*4  # Weights for Lattice Boltzmann method
cylinder = mesh.boundaryMarker()  # Example cylinder boundary

# Initialize fields
F = np.zeros((mesh.nodeCount(), mesh.nodeCount(), len(idxs)))  # Distribution function

# GPU and Multithreading setup
if gpu_rendering:
    mod = SourceModule("""
    __global__ void update_F(float *F, float *rho, float *ux, float *uy, int *cxs, int *cys, float *weights, int tau, int num_nodes) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < num_nodes) {
            // GPU implementation for updating F (simplified example)
            // Note: You should implement the specific GPU logic needed here
        }
    }
    """)
    update_F = mod.get_function("update_F")

if multithreading:
    def update_distribution(F, rho, ux, uy, cxs, cys, weights, tau):
        # Multithreaded implementation for updating F
        pass

    threads = []

# Simulation loop
frames = []
for it in tqdm(range(Nt)):
    # Drift and Streaming
    for i, cx, cy in zip(idxs, cxs, cys):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
        F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

    # Set reflective boundaries
    bndryF = F[cylinder, :]
    bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

    # Calculate fluid variables
    rho = np.sum(F, 2)
    ux = np.sum(F * cxs, 2) / rho
    uy = np.sum(F * cys, 2) / rho

    if gpu_rendering:
        update_F(cuda.Out(F), cuda.In(rho), cuda.In(ux), cuda.In(uy), cuda.In(cxs), cuda.In(cys), cuda.In(weights), np.int32(tau), np.int32(mesh.nodeCount()), block=(256, 1, 1), grid=(mesh.nodeCount() // 256 + 1, 1))

    elif multithreading:
        for t in range(4):  # Example: 4 threads
            thread = threading.Thread(target=update_distribution, args=(F, rho, ux, uy, cxs, cys, weights, tau))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    else:
        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy)
                                      + 9 * (cx * ux + cy * uy) ** 2 / 2 - 3 * (ux ** 2 + uy ** 2) / 2)

        F += -(1.0 / tau) * (F - Feq)

        # Apply boundary
        F[cylinder, :] = bndryF

    # Visualization and saving frames
    if (it % 50) == 0 or (it == Nt - 1):
        plt.cla()  # Clear the current axes
        ux[cylinder] = 0
        uy[cylinder] = 0
        vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        vorticity[cylinder] = np.nan
        vorticity_masked = np.ma.array(vorticity, mask=cylinder)
        plt.imshow(vorticity_masked, cmap='gist_rainbow')
        plt.colorbar()
        plt.clim(-.02, .02)
        plt.imshow(~cylinder, cmap='gray', alpha=0.3)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')

        plt.savefig(f'frame_{it}.png', dpi=480)
        frames.append(imageio.imread(f'frame_{it}.png'))
        plt.close()

# Clean up
gmsh.finalize()
