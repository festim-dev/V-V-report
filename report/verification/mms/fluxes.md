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

# Diffusion with dissociation flux

```{tags} 2D, MMS, DissociationFlux, steady state
```

This example is a diffusion problem with one boundary under a dissociation flux (`DissociationFlux`).

The problem is:

$$
\begin{align}
    \nabla \cdot (D \ \nabla{c}) = -S &  \quad \text{on }  \Omega  \\
    c = c_0 & \quad \text{on }  x = 0 \\
    -D \ \nabla c \cdot \mathbf{n} = 2 K_\mathrm{d} \ P & \quad \text{on }  x = 1 \\
\end{align}
$$(problem_fluxes)

The exact solution for mobile concentration is:

$$
\begin{equation}
    c_\mathrm{exact} = 10 + 2 x^2
\end{equation}
$$(c_exact_fluxes)

Injecting {eq}`c_exact_fluxes` in {eq}`problem_fluxes`, we obtain the expressions of $S$, $c_0$, and $K_\mathrm{d}$:

\begin{align}
    & S = -4 D \\
    & c_0 = c_\mathrm{exact}\\
    & P = \frac{2D}{K_\mathrm{d}} 
\end{align}

We can then run a FESTIM model with these values and compare the numerical solution with $c_\mathrm{exact}$.

+++

## FESTIM code

```{code-cell} ipython3
import festim as F
import matplotlib.pyplot as plt
import numpy as np
import ufl
import dolfinx

from mpi4py import MPI

# Create and mark the mesh
nx = ny = 10
fenics_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=dolfinx.mesh.CellType.quadrilateral)


class LefSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 0)
        )


class RightSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 1)
        )


# Create the FESTIM model
my_model = F.HydrogenTransportProblem()

H = F.Species("H")
my_model.species = [H]
my_model.mesh = F.Mesh(fenics_mesh)

left_surface = LefSurface(id=1)
right_surface = RightSurface(id=2)

D = 20
K_d = 10
material = F.Material(D_0=D, E_D=0)

volume = F.VolumeSubdomain(id=1, material=material)
my_model.subdomains = [
    left_surface,
    right_surface,
    volume,
]

# Variational formulation
exact_solution = lambda x: 10 + 2 * x[0] ** 2  # exact solution


my_model.sources = [F.ParticleSource(-4 * D, volume=volume, species=H)]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_surface, value=exact_solution, species=H),
    F.SurfaceReactionBC(
        reactant=[H, H],
        gas_pressure=2 * D / K_d,
        k_d0=K_d,
        E_kd=0,
        k_r0=0,
        E_kr=0,
        subdomain=right_surface,
    ),
]

my_model.temperature = 500.0  # ignored in this problem

my_model.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=False,
)

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

First, we compute the $L^2$-norm of the error, defined by $E=\sqrt{\int_\Omega (c-c_\mathrm{exact})^2\mathrm{d} x}$. Secondly, we compute the maximum error at any degree of freedom.

```{code-cell} ipython3
:tags: [hide-input]

def error_L2(u_computed, u_exact, degree_raise=3):
    # Create higher order function space
    degree = u_computed.function_space.ufl_element().degree
    family = u_computed.function_space.ufl_element().family_name
    mesh = u_computed.function_space.mesh
    W = dolfinx.fem.functionspace(mesh, (family, degree + degree_raise))

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = dolfinx.fem.Function(W)
    if isinstance(u_exact, ufl.core.expr.Expr):
        u_expr = dolfinx.fem.Expression(u_exact, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_exact)

    # Integrate the error
    error = dolfinx.fem.form(ufl.inner(u_computed - u_ex_W, u_computed - u_ex_W) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)
```

```{code-cell} ipython3
:tags: [hide-input]

computed_solution = H.solution

E_l2 = error_L2(computed_solution, exact_solution)

exact_solution_function = dolfinx.fem.Function(computed_solution.function_space)
exact_solution_function.interpolate(exact_solution)
E_max = np.max(np.abs(exact_solution_function.x.array-computed_solution.x.array))

print(f"L2 error: {E_l2:.2e}")
print(f"Max error: {E_max:.2e}")
```

The concentration fields can be visualised using `pyvista`.

```{code-cell} ipython3
:tags: [hide-input]

import pyvista
from dolfinx.plot import vtk_mesh

pyvista.start_xvfb()
pyvista.set_jupyter_backend('html')

u_topology, u_cell_types, u_geometry = vtk_mesh(computed_solution.function_space)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["c"] = computed_solution.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("diss_concentration.png")
```

```{code-cell} ipython3
:tags: [hide-input]

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["c_exact"] = exact_solution_function.x.array.real
u_grid.set_active_scalars("c_exact")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("diss_exact.png")
```

## Compute convergence rates

It is also possible to compute how the numerical error decreases as we increase the number of cells.
By iteratively refining the mesh, we find that the error exhibits a second order convergence rate.
This is expected for this particular problem as first order finite elements are used.

```{code-cell} ipython3
errors = []
ns = [5, 10, 20, 30, 50, 100, 150]

for n in ns:
    nx = ny = n
    fenics_mesh = fenics_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=dolfinx.mesh.CellType.quadrilateral)

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
