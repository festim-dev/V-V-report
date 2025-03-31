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

# Simple transient diffusion case

```{tags} 2D, MMS, transient
```

This is a simple transient MMS example.
We will only consider diffusion of hydrogen in a unit square domain $\Omega$ at steady state with an homogeneous diffusion coefficient $D$.
Moreover, a Dirichlet boundary condition will be assumed on the boundaries $\partial \Omega $.

The problem is therefore:

$$
\begin{align}
    &\nabla \cdot (D \ \nabla{c}) - \frac{\partial c}{\partial t} = -S  \quad \text{on }  \Omega  ; \ t\geq 0 \\
    & c = c_0 \quad \text{on }  \partial \Omega ; \ t\geq 0 \\
    & c = c_\mathrm{initial} \quad \text{on } \partial \Omega ; \ \text{at } t=0
\end{align}
$$(problem_simple_transient)

The exact solution for mobile concentration is:

$$
\begin{equation}
    c_\mathrm{exact} = 1 + 2 x^2 + 3 y^2 t + 2t
\end{equation}
$$(c_exact_simple_transient)

```{note}
We use a manufactured solution that varies linearly with time ($t^1$), as the backward Euler scheme provides an exact solution in this case.
```

Injecting {eq}`c_exact_simple_transient` in {eq}`problem_simple_transient`, we obtain the expressions of $S$, $c_0$, and $c_\mathrm{initial}$:

\begin{align}
    & S = 2 + 3 y^2 - (4 + 6t) D \\
    & c_0 = c_\mathrm{exact} \\
    & c_\mathrm{initial} = c_\mathrm{exact}(t=0)
\end{align}

We can then run a FESTIM model with these values and compare the numerical solution with $c_\mathrm{exact}$.

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-cell]

from mpi4py import MPI
import festim as F
from dolfinx.mesh import create_unit_square
import numpy as np

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 100, 100)

boundary = F.SurfaceSubdomain(id=1)
H = F.Species("mobile", mobile=True)
D = 2
my_mat = F.Material(D_0=D, E_D=0)
volume = F.VolumeSubdomain(id=1, material=my_mat)
boundary = F.SurfaceSubdomain(id=1)

exact_solution = (
    lambda x, t: 1 + 2 * x[0] ** 2 + 3 * t * x[1] ** 2 + 2 * t
)  # exact solution

f = lambda x, t: 2 + 3 * x[1] ** 2 - (4 + 6 * t) * D

final_time = 17
slices = 4
slice_size = final_time / slices
my_milestones = list(np.linspace(slice_size, final_time, slices))


my_subdomains = [volume, boundary]
my_species = [H]
my_sources = [F.ParticleSource(value=f, volume=volume, species=H)]
my_boundary_conditions = [
    F.FixedConcentrationBC(subdomain=boundary, value=exact_solution, species=H)
]
my_settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=True,
    final_time=final_time,
    stepsize=F.Stepsize(
        initial_value=0.25,
        growth_factor=1.0,
        target_nb_iterations=30,
        milestones=my_milestones,
    ),
)

my_model = F.HydrogenTransportProblem(
    mesh=F.Mesh(fenics_mesh),
    subdomains=my_subdomains,
    species=my_species,
    sources=my_sources,
    boundary_conditions=my_boundary_conditions,
    temperature=500,
    settings=my_settings,
    exports=[
        F.VTXSpeciesExport(
            field=H,
            filename="simple_transient_mobile.bp",
            subdomain=volume,
            checkpoint=True,
        )
    ],
)

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input]

import pyvista
from dolfinx.plot import vtk_mesh
from dolfinx import fem

def read_function_from_file(
    filename: str, name: str, timestamp: int | float, family="P", order: int = 1
) -> fem.Function:
    """
    Read a function from a file

    note::
        The function is read from a file using adios4dolfinx. For more information
        see the [adios4dolfinx documentation](https://jsdokken.com/adios4dolfinx/README.html).

    Args:
        filename: the filename
        name: the name of the function
        timestamp: the timestamp of the function
        family: the family of the function space
        order: the order of the function space

    Returns:
        the function
    """
    mesh_in = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD)
    V_in = fem.functionspace(mesh_in, (family, order))
    u_in = fem.Function(V_in)
    adios4dolfinx.read_function(
        filename=filename,
        u=u_in,
        name=name,
        time=timestamp,
    )
    return u_in


def error_L2(u_computed, u_exact, degree_raise=3):
    # Create higher order function space
    degree = u_computed.function_space.ufl_element().degree
    family = u_computed.function_space.ufl_element().family_name
    mesh = u_computed.function_space.mesh
    W = fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = fem.Function(W)
    u_W.interpolate(u_computed)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = fem.Function(W)
    if isinstance(u_exact, ufl.core.expr.Expr):
        u_expr = fem.Expression(u_exact, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_exact)

    # Compute the error in the higher order function space
    e_W = fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def get_u_grid(computed_solution: fem.Function, label: str):
    u_topology, u_cell_types, u_geometry = vtk_mesh(computed_solution.function_space)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data[label] = computed_solution.x.array.real
    u_grid.set_active_scalars(label)
    return u_grid


pyvista.start_xvfb()
pyvista.set_jupyter_backend('html')

computed_solution = read_function_from_file(
    "simple_transient_mobile.bp", "mobile", my_milestones[0]
)
u_grid_mobile_computed = get_u_grid(computed_solution, "c_mobile")

c_exact = fem.Function(computed_solution.function_space)
c_exact.interpolate(lambda x: exact_solution(x, my_milestones[0]))
u_grid_mobile_exact = get_u_grid(c_exact, "c_mobile_exact")

# Get the range of values for both computed and exact solutions
combined_range = [
    min(u_grid_mobile_computed.point_data["c_mobile"].min(), u_grid_mobile_exact.point_data["c_mobile_exact"].min()),
    max(u_grid_mobile_computed.point_data["c_mobile"].max(), u_grid_mobile_exact.point_data["c_mobile_exact"].max()),
]


pyvista.start_xvfb()
pyvista.set_jupyter_backend('html')

u_plotter = pyvista.Plotter(shape=(1, 2))

u_plotter.subplot(0, 0)
u_plotter.add_mesh(u_grid_mobile_exact, show_edges=False, cmap="inferno", clim=combined_range)
u_plotter.view_xy()

u_plotter.subplot(0, 1)
u_plotter.add_mesh(u_grid_mobile_computed, show_edges=False, cmap="inferno", clim=combined_range)
u_plotter.view_xy()


if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("computed_transient.png")
```

## Compute convergence rates

It is also possible to compute how the numerical error decreases as we increase the number of cells.
By iteratively refining the mesh, we find that the error exhibits a second order convergence rate.
This is expected for this particular problem as first order finite elements are used.

```{code-cell} ipython3
:tags: [hide-cell]

def error_L2(u_computed, u_exact, degree_raise=3):
    # Create higher order function space
    degree = u_computed.function_space.ufl_element().degree
    family = u_computed.function_space.ufl_element().family_name
    mesh = u_computed.function_space.mesh
    W = fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = fem.Function(W)
    u_W.interpolate(u_computed)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = fem.Function(W)
    if isinstance(u_exact, ufl.core.expr.Expr):
        u_expr = fem.Expression(u_exact, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_exact)

    # Compute the error in the higher order function space
    e_W = fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)

errors = []
ns = np.geomspace(5, 150, num=7, dtype=int)

for n in ns:
    new_model = F.HydrogenTransportProblem()

    new_model.mesh = F.Mesh(create_unit_square(MPI.COMM_WORLD, n, n))
    new_model.subdomains = my_subdomains
    new_model.species = my_species
    new_model.sources = my_sources
    new_model.boundary_conditions = my_boundary_conditions
    new_model.temperature = 500
    new_model.settings = my_settings

    new_model.initialise()
    new_model.run()

    computed_solution = H.post_processing_solution
    exact = lambda x: exact_solution(x, t=final_time)
    L2_error = error_L2(computed_solution, exact)
    errors.append(L2_error)
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

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
