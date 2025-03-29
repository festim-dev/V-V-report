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

# Radioactive decay 2D

```{tags} 2D, MMS, RadioactiveDecay, transient
```

This MMS case verifies the implementation of radioactive decay in FESTIM.
We will only consider diffusion of hydrogen in a unit square domain $\Omega$ at steady state with a homogeneous diffusion coefficient $D$.
We will impose a radioactive decay (`RadioactiveDecay`) source over the whole domain
with a decay constant $\lambda$.
Moreover, a Dirichlet boundary condition will be assumed on the boundaries $\partial \Omega $.

The problem is therefore:

$$
\begin{align}
    &\nabla \cdot (D \ \nabla{c}) -\lambda c = -S  \quad \text{on }  \Omega  \\
    & c = c_0 \quad \text{on }  \partial \Omega
\end{align}
$$(problem)

The manufactured exact solution for mobile concentration is:

$$
\begin{equation}
    c_\mathrm{exact} = 1 + 2 x^2 + 3 y^2
\end{equation}
$$(c_exact)

Injecting {eq}`c_exact` in {eq}`problem`, we obtain the expressions of $S$ and $c_0$:

$$
\begin{align}
    & S = \lambda \left(1 + 2 x^2 + 3 y^2 \right) -10 D \\
    & c_0 = c_\mathrm{exact}
\end{align}
$$

We can then run a FESTIM model with these values and compare the numerical solution with $c_\mathrm{exact}$.

```{code-cell} ipython3
import festim as F
import numpy as np

import dolfinx
from mpi4py import MPI
import ufl

decay_constant = 3
D = 2

# Create the FESTIM model
my_model = F.HydrogenTransportProblem()

# Create and mark the mesh
nx = ny = 20
fenics_mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, nx, ny, cell_type=dolfinx.mesh.CellType.quadrilateral
)
my_model.mesh = F.Mesh(fenics_mesh)

my_mat = F.Material(D_0=D, E_D=0)
volume = F.VolumeSubdomain(id=1, material=my_mat)
boundary = F.SurfaceSubdomain(id=1)

my_model.subdomains = [volume, boundary]

H = F.Species("H")
my_model.species = [H]

# Variational formulation
exact_solution = lambda x: 1 + 2 * x[0] ** 2 + 3 * x[1] ** 2  # exact solution

def S(x):
    return decay_constant * exact_solution(x) - 10 * D

my_model.sources = [
    F.ParticleSource(
        value=S, volume=volume, species=H
    ),
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=boundary, value=exact_solution, species=H),
]

decay_reaction = F.Reaction(reactant=H, k_0=decay_constant, E_k=0, volume=volume)

my_model.reactions = [decay_reaction]


my_model.temperature = 500.0  # ignored in this problem

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

First, we compute the $L^2$-norm of the error, defined by $E=\sqrt{\int_\Omega (c-c_\mathrm{exact})^2\mathrm{d} x}$. Secondly, we compute the maximum error at any degree of freedom.

```{code-cell} ipython3
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
    error = dolfinx.fem.form(
        ufl.inner(u_computed - u_ex_W, u_computed - u_ex_W) * ufl.dx
    )
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

```{code-cell} ipython3
import pyvista
from dolfinx.plot import vtk_mesh

pyvista.start_xvfb()
pyvista.set_jupyter_backend('html')

u_topology, u_cell_types, u_geometry = vtk_mesh(computed_solution.function_space)

u_grid_computed = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid_computed.point_data["c"] = computed_solution.x.array.real
u_grid_computed.set_active_scalars("c")

u_grid_exact = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid_exact.point_data["c_exact"] = exact_solution_function.x.array.real
u_grid_exact.set_active_scalars("c_exact")


u_plotter = pyvista.Plotter(shape=(1, 2))

u_plotter.subplot(0, 0)
u_plotter.add_mesh(u_grid_computed, show_edges=True)
u_plotter.view_xy()

u_plotter.subplot(0, 1)
u_plotter.add_mesh(u_grid_exact, show_edges=True)
u_plotter.view_xy()


if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("decay.png")
```
