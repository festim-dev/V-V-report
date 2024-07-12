---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: vv-festim-report-env
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
    & c = c_\mathrm{initial} \quad \text{on } \partial \Omega ; \\text{at } t=0
\end{align}
$$(problem_simple_transient)

The exact solution for mobile concentration is:

$$
\begin{equation}
    c_\mathrm{exact} = 1 + 2 x^2 + 3 y^2 t + 2t
\end{equation}
$$(c_exact_simple_transient)

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

import festim as F
import sympy as sp
import fenics as f
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Create and mark the mesh
nx = ny = 100
fenics_mesh = f.UnitSquareMesh(nx, ny)


volume_markers = f.MeshFunction("size_t", fenics_mesh, fenics_mesh.topology().dim())
volume_markers.set_all(1)

surface_markers = f.MeshFunction(
    "size_t", fenics_mesh, fenics_mesh.topology().dim() - 1
)
surface_markers.set_all(0)

class Boundary(f.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

boundary = Boundary()
boundary.mark(surface_markers, 1)

# Create the FESTIM model
my_model = F.Simulation()

my_model.mesh = F.Mesh(
    fenics_mesh, volume_markers=volume_markers, surface_markers=surface_markers
)

# Variational formulation
exact_solution = (
    1 + 2 * F.x**2 + 3 * F.t * F.y**2 + 2 * F.t
)  # exact solution

exact_solution_copy = exact_solution.subs(F.x, F.x)

D = 2

my_model.sources = [
    F.Source(2 + 3 * F.y**2 - (4 + 6 * F.t) * D, volume=1, field="solute"),
]

my_model.boundary_conditions = [
    F.DirichletBC(surfaces=[1], value=exact_solution, field="solute"),
]

my_model.materials = F.Material(id=1, D_0=D, E_D=0)

my_model.T = F.Temperature(500)  # ignored in this problem

xdmf_file_name = "simple_transient_mobile.xdmf"
my_model.exports = [F.XDMFExport(field="solute", filename=xdmf_file_name, checkpoint=True)]

final_time = 17
slices = 4
slice_size = final_time / slices
milestones = list(np.linspace(slice_size, final_time, slices))

my_model.dt = F.Stepsize(
    initial_value=1,
    milestones=milestones
)

my_model.settings = F.Settings(
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
    final_time=final_time,
)

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input]

from fenics import XDMFFile, FunctionSpace, Function, plot
def load_xdmf(mesh, filename, field, element="CG", counter=-1):
    """Loads a XDMF file and store its content to a fenics.Function

    Args:
        mesh (fenics.mesh): the mesh of the function
        filename (str): the XDMF filename
        field (str): the name of the field in the XDMF file
        element (str, optional): Finite element of the function.
            Defaults to "CG".
        counter (int, optional): timestep in the file, -1 is the
            last timestep. Defaults to -1.

    Returns:
        fenics.Function: the content of the XDMF file as a Function
    """

    V = FunctionSpace(mesh, element, 1)
    u = Function(V)

    XDMFFile(filename).read_checkpoint(u, field, counter)
    return u

fig, axs = plt.subplots(slices, 3, figsize=(slices*2.2, slices*2.5 + 1)) # tweak figsize if needed
fig.tight_layout()

def compute_arc_length(xs, ys):
    """Computes the arc length of x,y points based
    on x and y arrays
    """
    points = np.vstack((xs, ys)).T
    distance = np.linalg.norm(points[1:] - points[:-1], axis=1)
    arc_length = np.insert(np.cumsum(distance), 0, [0.0])
    return arc_length

def exists_close(x, list):
    return any(np.isclose(x, t) for t in list)

xdmf_times = F.extract_xdmf_times(xdmf_file_name)
counters = [i for (i, time) in enumerate(xdmf_times) if exists_close(time, milestones)]

for i, counter in enumerate(counters):
    time = xdmf_times[counter]
    
    c_exact = f.Expression(sp.printing.ccode(exact_solution_copy.subs(F.t, time)), degree=4)
    c_exact = f.project(c_exact, f.FunctionSpace(my_model.mesh.mesh, "CG", 1))

    computed_solution = load_xdmf(fenics_mesh, xdmf_file_name, "mobile_concentration", "CG", counter)
    E = f.errornorm(computed_solution, c_exact, "L2")

    # plot exact solution and computed solution
    plt.sca(axs[i, 0])
    plt.title(f"Exact solution at t={time}s")
    CS1 = f.plot(c_exact, cmap="inferno")
    plt.sca(axs[i, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Computed solution at t={time}s")
    CS2 = f.plot(computed_solution, cmap="inferno")

    plt.colorbar(CS1, ax=[axs[i, 0]], shrink=0.8)
    plt.colorbar(CS2, ax=[axs[i, 1]], shrink=0.8)

    axs[i, 0].sharey(axs[i, 1])
    plt.setp(axs[i, 1].get_yticklabels(), visible=False)

    for CS in [CS1, CS2]:
        CS.set_edgecolor("face")


    # define the profiles
    profiles = [
        {"start": (0.0, 0.0), "end": (1.0, 1.0)},
        {"start": (0.2, 0.8), "end": (0.7, 0.2)},
        {"start": (0.2, 0.6), "end": (0.8, 0.8)},
    ]

    # plot the profiles on the right subplot
    for profile in profiles:
        start_x, start_y = profile["start"]
        end_x, end_y = profile["end"]
        plt.sca(axs[i, 1])
        (l,) = plt.plot([start_x, end_x], [start_y, end_y])

        plt.sca(axs[i, 2])

        points_x_exact = np.linspace(start_x, end_x, num=30)
        points_y_exact = np.linspace(start_y, end_y, num=30)
        arc_length_exact = compute_arc_length(points_x_exact, points_y_exact)
        u_values = [c_exact(x, y) for x, y in zip(points_x_exact, points_y_exact)]

        points_x = np.linspace(start_x, end_x, num=100)
        points_y = np.linspace(start_y, end_y, num=100)
        arc_lengths = sorted(compute_arc_length(points_x, points_y))
        computed_values = [computed_solution(x, y) for x, y in zip(points_x, points_y)]

        (exact_line,) = plt.plot(
            arc_length_exact, u_values, color=l.get_color(), marker="o", linestyle="None", alpha=0.3
        )
        (computed_line,) = plt.plot(arc_lengths, computed_values, color=l.get_color())

    plt.sca(axs[i, 2])
    if(i == 0):
        plt.xlabel("Arc length")

    if i == 0:
        legend_marker = mpl.lines.Line2D(
            [],
            [],
            color="black",
            marker=exact_line.get_marker(),
            linestyle="None",
            label="Exact",
        )
        legend_line = mpl.lines.Line2D([], [], color="black", label="Computed")
        plt.legend(
            [legend_marker, legend_line], [legend_marker.get_label(), legend_line.get_label()]
        )
    plt.grid(alpha=0.3)
    plt.gca().spines[["right", "top"]].set_visible(False)
    
plt.show()
```

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
    fenics_mesh = f.UnitSquareMesh(nx, ny)

    volume_markers = f.MeshFunction("size_t", fenics_mesh, fenics_mesh.topology().dim())
    volume_markers.set_all(1)

    surface_markers = f.MeshFunction(
        "size_t", fenics_mesh, fenics_mesh.topology().dim() - 1
    )
    surface_markers.set_all(0)

    class Boundary(f.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    boundary = Boundary()
    boundary.mark(surface_markers, 1)

    my_model.mesh = F.Mesh(
        fenics_mesh, volume_markers=volume_markers, surface_markers=surface_markers
    )

    my_model.initialise()
    my_model.run()
    
    computed_solution = my_model.h_transport_problem.mobile.post_processing_solution
    errors.append(f.errornorm(computed_solution, c_exact, "L2"))

h = 1 / np.array(ns)

plt.loglog(h, errors, marker="o")
plt.xlabel("Element size")
plt.ylabel("L2 error")

plt.loglog(h, 2 * h**2, linestyle="--", color="black")
plt.annotate("2nd order", (h[0], 2 * h[0]**2), textcoords="offset points", xytext=(10, 0))

plt.grid(alpha=0.3)
plt.gca().spines[["right", "top"]].set_visible(False)
```
