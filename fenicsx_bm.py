from dolfinx import fem, mesh, io
from ufl import dx, grad, inner, TrialFunction, TestFunction
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import ufl
# Create mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD,
                               [np.array([0.0, 0.0]), np.array([2.0, 2.0])],
                               [2, 3],
                               mesh.CellType.quadrilateral)

# Define function space
V = fem.FunctionSpace(domain, ("CG", 1))

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)


# Define Dirichlet BC at right bottom corner
def boundary_D(x):
    return np.isclose(x[0], 2.0) & np.isclose(x[1], 0.0)

def boundary_D_top(x):
    return np.isclose(x[0], 0.0) & np.isclose(x[1], 2.0)


dofs_D = fem.locate_dofs_geometrical(V, boundary_D)
bc_value = fem.Constant(domain, ScalarType(0.0))
bc = fem.dirichletbc(bc_value, dofs_D, V)

dofs_D_top = fem.locate_dofs_geometrical(V, boundary_D_top)
bc_value_top = fem.Constant(domain, ScalarType(1.0))
bc_top = fem.dirichletbc(bc_value_top, dofs_D_top, V)



# Define negative flux on left edge
def boundary_N(x):
    return np.isclose(x[0], 0.0)


tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(tdim, fdim)
facets_left = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], 0.0)
)
tags = np.zeros(domain.topology.index_map(fdim).size_local, dtype=np.int32)
tags[facets_left] = 1
mt = mesh.meshtags(domain, fdim, np.arange(len(tags), dtype=np.int32), tags)

# Boundary measure restricted to tags
ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
g = fem.Constant(domain, ScalarType(0))  # Negative flux

# Define weak form
a = inner(grad(u), grad(v)) * dx
L = g * v * ds(1)

# Solve problem
problem = fem.petsc.LinearProblem(a, L, bcs=[bc, bc_top])
uh = problem.solve()

with io.XDMFFile(MPI.COMM_WORLD, "solution_fenicsx.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

# make scatter plot of solution using matplotlib
import matplotlib.pyplot as plt
u_vals = uh.x.array
coords = domain.geometry.x
plt.scatter(coords[:, 0], coords[:, 1], c=u_vals, cmap='viridis')
plt.colorbar(label='Temperature')
plt.title('Temperature Distribution - FEniCSx')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
for i in range(len(coords)):
    plt.text(coords[i, 0], coords[i, 1]+0.02, f'{u_vals[i]:.4f}', fontsize=9)
plt.show()
