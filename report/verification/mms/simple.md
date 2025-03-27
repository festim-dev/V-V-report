---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: vv-festim-report-env-festim-2
  language: python
  name: python3
---

# Simple diffusion case

```{tags} 2D, MMS, steady state
```

This is a simple MMS example.
We will only consider diffusion of hydrogen in a unit square domain $\Omega$ at steady state with an homogeneous diffusion coefficient $D$.
Moreover, a Dirichlet boundary condition will be assumed on the boundaries $\partial \Omega $.

The problem is therefore:
```{math}
:label: problem_simple
\begin{align}
    &\nabla \cdot (D \ \nabla{c}) = -S  \quad \text{on }  \Omega  \\
    & c = c_0 \quad \text{on }  \partial \Omega
\end{align}
```

The exact solution for mobile concentration is:

$$
\begin{equation}
    c_\mathrm{exact} = 1 + 2 x^2 + 3 y^2
\end{equation}
$$(c_exact_simple)

Injecting {eq}`c_exact_simple` in {eq}`problem_simple`, we obtain the expressions of $S$ and $c_0$:

\begin{align}
    & S = -10 D \\
    & c_0 = c_\mathrm{exact}
\end{align}

We can then run a FESTIM model with these values and compare the numerical solution with $c_\mathrm{exact}$.

+++

## FESTIM code

```{code-cell} ipython3
import festim as F
import ufl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import dolfinx
from mpi4py import MPI

# Create and mark the mesh
nx = ny = 10
fenics_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)

# Create the FESTIM model
my_model = F.HydrogenTransportProblem()

H = F.Species("H")
my_model.species = [H]
my_model.mesh = F.Mesh(fenics_mesh)

D = 2.0
material = F.Material(D_0=D, E_D=0)

volume = F.VolumeSubdomain(id=1, material=material)
boundary = F.SurfaceSubdomain(id=1)

my_model.subdomains = [boundary, volume]


exact_solution = lambda x: 1 + 2 * x[0] ** 2 + 3 * x[1] ** 2

my_model.sources = [
    F.ParticleSource(-10 * D, volume=volume, species=H),
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=boundary, value=exact_solution, species=H),
]


my_model.temperature = 500

my_model.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=False,
)

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
def error_L2(u_computed, u_exact, degree_raise=3):
    # Create higher order function space
    degree = u_computed.function_space.ufl_element().degree
    family = u_computed.function_space.ufl_element().family_name
    mesh = u_computed.function_space.mesh
    W = dolfinx.fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = dolfinx.fem.Function(W)
    u_W.interpolate(u_computed)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = dolfinx.fem.Function(W)
    if isinstance(u_exact, ufl.core.expr.Expr):
        u_expr = dolfinx.fem.Expression(u_exact, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_exact)

    # Compute the error in the higher order function space
    e_W = dolfinx.fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = dolfinx.fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)
```

```{code-cell} ipython3
:tags: [hide-input]

computed_solution = H.solution

E = error_L2(computed_solution, exact_solution)
print(f"L2 error: {E:.2e}")
```

<!-- ```{code-cell} ipython3
import pyvista
from dolfinx.plot import vtk_mesh

pyvista.OFF_SCREEN = True

u_topology, u_cell_types, u_geometry = vtk_mesh(computed_solution.function_space)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["c"] = computed_solution.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=False)
contours = u_grid.contour(50)
u_plotter.add_mesh(contours)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.show(screenshot="simple_concentration.png")
```

```{code-cell} ipython3
exact_solution_function = dolfinx.fem.Function(computed_solution.function_space)
exact_solution_function.interpolate(exact_solution)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["c_exact"] = exact_solution_function.x.array.real
u_grid.set_active_scalars("c_exact")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=False)
contours = u_grid.contour(50)
u_plotter.add_mesh(contours)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.show(screenshot="simple_exact.png")
``` -->

## Compute convergence rates

It is also possible to compute how the numerical error decreases as we increase the number of cells.
By iteratively refining the mesh, we find that the error exhibits a second order convergence rate.
This is expected for this particular problem as first order finite elements are used.

```{code-cell} ipython3
:tags: [hide-input]

errors = []
ns = [5, 10, 20, 30, 50, 100, 150]

for n in ns:
    nx = ny = n
    fenics_mesh = fenics_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)

    new_model = F.HydrogenTransportProblem()
    new_model.mesh = F.Mesh(fenics_mesh)

    new_model.species = my_model.species
    new_model.subdomains = my_model.subdomains
    new_model.sources = my_model.sources
    new_model.boundary_conditions = my_model.boundary_conditions
    new_model.temperature = my_model.temperature
    new_model.settings = my_model.settings


    new_model.initialise()
    new_model.run()

    computed_solution = H.solution
    errors.append(error_L2(computed_solution, exact_solution))

h = 1 / np.array(ns)

plt.loglog(h, errors, marker="o")
plt.xlabel("Element size")
plt.ylabel("L2 error")

plt.loglog(h, 2 * h**2, linestyle="--", color="black")
plt.annotate(
    "2nd order", (h[0], 2 * h[0] ** 2), textcoords="offset points", xytext=(10, 0)
)

plt.grid(alpha=0.3)
plt.gca().spines[["right", "top"]].set_visible(False)
```
