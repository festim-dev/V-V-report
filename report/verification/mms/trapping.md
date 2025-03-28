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

# Single trap

```{tags} 2D, MMS, trapping, steady state
```

The MMS case verifies the implementation of trapping in FESTIM.
Only one trap is considered for simplicity, but the same principle applies for more traps.
Exact solutions are defined for both the mobile concentration and the trapped concentration.

\begin{align}
    c_\mathrm{m, exact} &= 5 + \sin{\left(2 \pi x \right)} + \cos{\left(2 \pi y \right)} \\
    c_\mathrm{t, exact} &= 5 + \cos{\left(2 \pi x \right)} + \sin{\left(2 \pi y \right)}
\end{align}

The trap density is $n = 2 \ c_\mathrm{t,exact} $, the trapping rate is $k = 0.1$, the detrapping rate is $p = 0.2$, and the diffusivity is $D=5$.
MMS sources are obtained for both the mobile concentration and the trapped concentration:

\begin{align}
    S_\mathrm{m} &= -D \nabla^2 c_\mathrm{m, exact} + k c_\mathrm{m, exact} (n - c_\mathrm{t, exact}) - p c_\mathrm{t, exact} \\
    S_\mathrm{t} &= - k c_\mathrm{m, exact} (n - c_\mathrm{t, exact}) + p c_\mathrm{t, exact}
\end{align}

Again, the computed solution agrees very well with the exact solution.

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import festim as F
import matplotlib.pyplot as plt
import numpy as np

import dolfinx
from mpi4py import MPI
import ufl

# Create and mark the mesh
nx = ny = 20
fenics_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=dolfinx.mesh.CellType.quadrilateral)


boundary = F.SurfaceSubdomain(id=1)
# Create the FESTIM model
my_model = F.HydrogenTransportProblem()



my_model.mesh = F.Mesh(
    fenics_mesh
)

my_mat = F.Material(D_0=5, E_D=0)
volume = F.VolumeSubdomain(id=1, material=my_mat)

my_model.subdomains = [volume, boundary]

# Variational formulation

def exact_solution_mobile(mod=np):
    return lambda x: 5 + mod.sin(2 * mod.pi * x[0]) + mod.cos(2 * mod.pi * x[1])

def exact_solution_trapped(mod=np):
    return lambda x: 5 + mod.cos(2 * mod.pi * x[0]) + mod.sin(2 * mod.pi *  x[1])

exact_solution_mobile_ufl = exact_solution_mobile(mod=ufl)
exact_solution_trapped_ufl = exact_solution_trapped(mod=ufl)
exact_solution_mobile_np = exact_solution_mobile(mod=np)
exact_solution_trapped_np = exact_solution_trapped(mod=np)

my_model.temperature = 500.0

H = F.Species("H")
trapped_H = F.Species("trapped_H", mobile=False)
trap_density = lambda x: 2 * exact_solution_mobile_ufl(x)
empty_trap = F.ImplicitSpecies(n=trap_density, others=[trapped_H])

my_model.species = [H, trapped_H]

trapping_reaction = F.Reaction(
    reactant=[H, empty_trap],
    product=[trapped_H],
    k_0=0.1,
    E_k=0,
    p_0=0.2,
    E_p=0,
    volume=volume,
)
my_model.reactions = [trapping_reaction]


# source term left
k = trapping_reaction.k_0 * ufl.exp(-trapping_reaction.E_k / my_model.temperature)
p = trapping_reaction.p_0 * ufl.exp(-trapping_reaction.E_p / my_model.temperature)
D = my_mat.D_0 * ufl.exp(-my_mat.E_D / my_model.temperature)

f_mobile = lambda x: (
    -ufl.div(D * ufl.grad(exact_solution_mobile_ufl(x)))
    + k * exact_solution_mobile_ufl(x) * (trap_density(x) - exact_solution_trapped_ufl(x))
    - p * exact_solution_trapped_ufl(x)
)
f_trap = lambda x:(
    -k * exact_solution_mobile_ufl(x) * (trap_density(x) - exact_solution_trapped_ufl(x))
    + p * exact_solution_trapped_ufl(x)
)

my_model.sources = [
    F.ParticleSource(f_mobile, volume=volume, species=H),
    F.ParticleSource(f_trap, volume=volume, species=trapped_H),
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=boundary, value=exact_solution_mobile_ufl, species=H),
    F.FixedConcentrationBC(subdomain=boundary, value=exact_solution_trapped_ufl, species=trapped_H),
]

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
exact_functions = {}

for species, exact_solution, label in zip([H, trapped_H], [exact_solution_mobile_np, exact_solution_trapped_np], ["Mobile", "Trapped"]):
    computed_solution = species.post_processing_solution

    E_l2 = error_L2(computed_solution, exact_solution)

    exact_solution_function = dolfinx.fem.Function(computed_solution.function_space)
    exact_solution_function.interpolate(exact_solution)

    exact_functions[label] = exact_solution_function
    E_max = np.max(np.abs(exact_solution_function.x.array-computed_solution.x.array))

    print(f"{label}:")
    print(f"L2 error: {E_l2:.2e}")
    print(f"Max error: {E_max:.2e}")
```

The concentration fields can be visualised using `pyvista`.

```{code-cell} ipython3
import pyvista
from dolfinx.plot import vtk_mesh

pyvista.start_xvfb()
pyvista.set_jupyter_backend('html')

def get_u_grid(computed_solution: dolfinx.fem.Function, label: str):
    u_topology, u_cell_types, u_geometry = vtk_mesh(computed_solution.function_space)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data[label] = computed_solution.x.array.real
    u_grid.set_active_scalars(label)
    return u_grid

u_grid_mobile = get_u_grid(H.post_processing_solution, "c_mobile")
u_grid_trapped = get_u_grid(trapped_H.post_processing_solution, "c_trapped")


u_grid_mobile_exact = get_u_grid(exact_functions["Mobile"], "c_mobile_exact")
u_grid_trapped_exact = get_u_grid(exact_functions["Trapped"], "c_trapped_exact")

u_plotter = pyvista.Plotter(shape=(2, 2))
u_plotter.subplot(0, 0)
u_plotter.add_mesh(u_grid_mobile, show_edges=False)
u_plotter.view_xy()

u_plotter.subplot(0, 1)
u_plotter.add_mesh(u_grid_trapped, show_edges=False)
u_plotter.view_xy()

u_plotter.subplot(1, 0)
u_plotter.add_mesh(u_grid_mobile_exact, show_edges=False)
u_plotter.view_xy()

u_plotter.subplot(1, 1)
u_plotter.add_mesh(u_grid_trapped_exact, show_edges=False)
u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("computed_trapping.png")
```
