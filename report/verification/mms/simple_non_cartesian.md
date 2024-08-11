---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: vv-festim-report-env
  language: python
  name: python3
---

# Simple non-cartesian diffusion cases

```{tags} 2D, MMS, steady state, cylindrical, spherical
```

This is a simple MMS example on both cylindrical and spherical meshes.
We will only consider diffusion of hydrogen in a unit disk domain $\Omega$ at steady state with an homogeneous diffusion coefficient $D$.
Moreover, a Dirichlet boundary condition will be assumed on the boundaries $\partial \Omega $.

The problem is therefore:

$$
\begin{align}
    & \nabla \cdot (D \ \nabla{c}) = -S  \quad \text{on }  \Omega  \\
    & c = c_0 \quad \text{on }  \partial \Omega
\end{align}
$$(problem_simple_cylindrical)

The exact solution for mobile concentration is:

$$
\begin{equation}
    c_\mathrm{exact} = 1 + r^2
\end{equation}
$$(c_exact_simple_cylindrical)

Injecting {eq}`c_exact_simple_cylindrical` in {eq}`problem_simple_cylindrical`, we obtain the expressions of $S$ and $c_0$:

$$
\begin{align}
    & S = -4D \\
    & c_0 = c_\mathrm{exact}
\end{align}
$$

We can then run a FESTIM model with these values and compare the numerical solution with $c_\mathrm{exact}$.

+++

## FESTIM code

```{code-cell} ipython3
import festim as F
import sympy as sp
import fenics as f
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Create and mark the mesh
R = 1.0
nx = ny = 100
fenics_mesh = f.RectangleMesh(f.Point(0, 0), f.Point(R, 2 * np.pi), nx, ny)

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

r = F.x
exact_solution = r**2
D = 2


def run_model(type: str):

    # Create the FESTIM model
    my_model = F.Simulation()

    my_model.mesh = F.Mesh(
        fenics_mesh,
        volume_markers=volume_markers,
        surface_markers=surface_markers,
        type=type,
    )

    my_model.sources = [
        F.Source(-4 * D, volume=1, field="solute"),
    ]

    my_model.boundary_conditions = [
        F.DirichletBC(surfaces=[1], value=exact_solution, field="solute"),
    ]

    my_model.materials = F.Material(id=1, D_0=D, E_D=0)

    my_model.T = F.Temperature(500)  # ignored in this problem

    my_model.settings = F.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        transient=False,
    )

    my_model.initialise()
    my_model.run()

    return my_model.h_transport_problem.mobile.post_processing_solution
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input]

c_exact = f.Expression(sp.printing.ccode(exact_solution), degree=4)
c_exact = f.project(c_exact, f.FunctionSpace(fenics_mesh, "CG", 1))

cylindrical_solution = run_model("cylindrical")
spherical_solution = run_model("spherical")

E_cyl = f.errornorm(cylindrical_solution, c_exact, "L2")
E_sph = f.errornorm(spherical_solution, c_exact, "L2")
print(f"L2 error, cylindrical: {E_cyl:.2e}")
print(f"L2 error, spherical: {E_sph:.2e}")

# plot exact solution and computed solution
fig, axs = plt.subplots(
    1,
    4,
    figsize=(15, 5),
    sharey=True,
)

plt.sca(axs[0])
plt.title("Exact solution")
plt.xlabel("r")
plt.ylabel("$\\theta$")
CS1 = f.plot(c_exact, cmap="inferno")
plt.gca().set_aspect(1 / (2 * np.pi)) # TODO: change this
plt.sca(axs[1])
plt.xlabel("r")
plt.title("Cylindrical solution")
CS2 = f.plot(cylindrical_solution, cmap="inferno")
plt.gca().set_aspect(1 / (2 * np.pi)) # TODO: change this
plt.sca(axs[2])
plt.xlabel("r")
plt.title("Spherical solution")
CS3 = f.plot(spherical_solution, cmap="inferno")
plt.gca().set_aspect(1 / (2 * np.pi)) # TODO: change this

plt.colorbar(CS1, ax=[axs[0]], shrink=0.8)
plt.colorbar(CS2, ax=[axs[1]], shrink=0.8)
plt.colorbar(CS3, ax=[axs[2]], shrink=0.8)

# axs[0].sharey(axs[1])
plt.setp(axs[1].get_yticklabels(), visible=False)
plt.setp(axs[2].get_yticklabels(), visible=False)

for CS in [CS1, CS2, CS3]:
    CS.set_edgecolor("face")


def compute_arc_length(xs, ys):
    """Computes the arc length of x,y points based
    on x and y arrays
    """
    points = np.vstack((xs, ys)).T
    distance = np.linalg.norm(points[1:] - points[:-1], axis=1)
    arc_length = np.insert(np.cumsum(distance), 0, [0.0])
    return arc_length


# define the profiles
profiles = [
    {"start": (0.0, 0.0), "end": (1.0, 2 * np.pi)},
    {"start": (0.2, 2 * np.pi / 3), "end": (0.7, np.pi)},
    {"start": (0.5, 1), "end": (0.8, 4)},
]

# plot the profiles on the right subplot
for i, profile in enumerate(profiles):
    start_x, start_y = profile["start"]
    end_x, end_y = profile["end"]
    plt.sca(axs[1])
    (l,) = plt.plot([start_x, end_x], [start_y, end_y])

    plt.sca(axs[-1])

    points_x_exact = np.linspace(start_x, end_x, num=30)
    points_y_exact = np.linspace(start_y, end_y, num=30)
    arc_length_exact = compute_arc_length(points_x_exact, points_y_exact)
    u_values = [c_exact(x, y) for x, y in zip(points_x_exact, points_y_exact)]

    points_x = np.linspace(start_x, end_x, num=100)
    points_y = np.linspace(start_y, end_y, num=100)
    arc_lengths = compute_arc_length(points_x, points_y)
    computed_values = [cylindrical_solution(x, y) for x, y in zip(points_x, points_y)]

    (exact_line,) = plt.plot(
        arc_length_exact,
        u_values,
        color=l.get_color(),
        marker="o",
        linestyle="None",
        alpha=0.3,
    )
    (computed_line,) = plt.plot(arc_lengths, computed_values, color=l.get_color())

plt.sca(axs[-1])
plt.xlabel("Arc length")
plt.ylabel("Solution")

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
