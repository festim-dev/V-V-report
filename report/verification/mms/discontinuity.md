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

# Diffusion multi-material

```{tags} 2D, MMS, Multi-material, steady state
```

The first MMS problem has two materials (denoted, respectively, by left and right).
In material left, the solubility is $K_{S,\mathrm{left}} = 3$ and the diffusivity is $D_\mathrm{left} = 2$.
In material right, the solubility is $K_{S,\mathrm{right}} = 6$ and the diffusivity is $D_\mathrm{right} = 5$.
Two exact solutions for mobile concentration of hydrogen are manufactured for both subdomains:

\begin{align}
    c_\mathrm{left,exact} &= 1 + \sin{\left(\pi \left(2 x + 0.5\right) \right)} + \cos{\left(2 \pi y \right)} \\
    c_\mathrm{right,exact} &= \dfrac{K_{S,\mathrm{right}}}{K_{S,\mathrm{left}}} \ c_\mathrm{left,exact}
\end{align}

````{margin}
```{note}
The manufactured solutions were chosen so that the particle flux $J = -D \nabla c_\mathrm{m} \cdot \textbf{n}$ is continuous across the materials interface. 
```
````

MMS sources are derived in each material: 

\begin{align}
    S_\mathrm{left} &= 8 \pi^{2} \left(\cos{\left(2 \pi x \right)} + \cos{\left(2 \pi y \right)}\right) \\
    S_\mathrm{right} &= 40 \pi^{2} \left(\cos{\left(2 \pi x \right)} + \cos{\left(2 \pi y \right)}\right)
\end{align}


These exact solutions can then determine the MMS fluxes and boundary conditions.

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import festim as F
import numpy as np
import dolfinx
import ufl
from mpi4py import MPI

# Create and mark the mesh
nx = ny = 100
fenics_mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, nx, ny
)

# Create the FESTIM model


class LeftSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0)
        )


class RightSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 1.0)
        )


class TopRightSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 1.0) & (x[0] > 0.5)
        )


class TopLeftSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 1.0) & (x[0] < 0.5)
        )


class BottomRightSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 0.0) & (x[0] > 0.5)
        )


class BottomLeftSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 0.0) & (x[0] < 0.5)
        )


class LeftSubdomain(F.VolumeSubdomain):
    def locate_subdomain_entities(self, mesh):
        return dolfinx.mesh.locate_entities(
            mesh, mesh.topology.dim, lambda x: x[0] < 0.5 + 1e-10
        )


class RightSubdomain(F.VolumeSubdomain):
    def locate_subdomain_entities(self, mesh):
        return dolfinx.mesh.locate_entities(
            mesh, mesh.topology.dim, lambda x: x[0] >= 0.5 - 1e-10
        )


left_surface = LeftSurface(id=1)
right_surface = RightSurface(id=2)
top_left_surface = TopLeftSurface(id=3)
bottom_left_surface = BottomLeftSurface(id=4)
top_right_surface = TopRightSurface(id=5)
bottom_right_surface = BottomRightSurface(id=6)


# Create the FESTIM model
my_model = F.HTransportProblemDiscontinuous()

my_model.mesh = F.Mesh(fenics_mesh)

D_left, D_right = 2, 5  # diffusion coeffs
S_left = 3
S_right = 6

mat_left = F.Material(D_0=D_left, E_D=0, K_S_0=S_left, E_K_S=0)
mat_right = F.Material(D_0=D_right, E_D=0, K_S_0=S_right, E_K_S=0)
left_volume = LeftSubdomain(id=1, material=mat_left)
right_volume = RightSubdomain(id=2, material=mat_right)

my_model.subdomains = [
    left_volume,
    right_volume,
    left_surface,
    right_surface,
    top_left_surface,
    bottom_left_surface,
    top_right_surface,
    bottom_right_surface,
]

my_model.surface_to_volume = {
    left_surface: left_volume,
    right_surface: right_volume,
    top_left_surface: left_volume,
    bottom_left_surface: left_volume,
    top_right_surface: right_volume,
    bottom_right_surface: right_volume,
}

my_model.interfaces = [
    F.Interface(id=7, subdomains=[left_volume, right_volume])
]

H = F.Species("H", subdomains=my_model.volume_subdomains)
my_model.species = [H]

def exact_solution_left(mod):
    return lambda x: (
        1 + mod.sin(2 * mod.pi * (x[0] + 0.25)) + mod.cos(2 * mod.pi * x[1])
    )

exact_solution_left_ufl = exact_solution_left(ufl)

def exact_solution_right(mod):
    return lambda x: S_right / S_left * exact_solution_left(mod)(x)

exact_solution_right_ufl = exact_solution_right(ufl)


# source term left
f_left = lambda x: -ufl.div(D_left * ufl.grad(exact_solution_left_ufl(x)))
f_right = lambda x: -ufl.div(D_right * ufl.grad(exact_solution_right_ufl(x)))


my_model.sources = [
    F.ParticleSource(f_left, volume=left_volume, species=H),
    F.ParticleSource(f_right, volume=right_volume, species=H),
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=surface, value=exact_solution_left_ufl, species=H)
    for surface in [left_surface, top_left_surface, bottom_left_surface]
] + [
    F.FixedConcentrationBC(subdomain=surface, value=exact_solution_right_ufl, species=H)
    for surface in [right_surface, top_right_surface, bottom_right_surface]
]


my_model.temperature = 500.0

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

The computed and exact solutions agree very well:

```{code-cell} ipython3
import pyvista
from dolfinx.plot import vtk_mesh

pyvista.start_xvfb()
pyvista.set_jupyter_backend('html')

def get_ugrid(computed_solution, label):
    u_topology, u_cell_types, u_geometry = vtk_mesh(computed_solution.function_space)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data[label] = computed_solution.x.array.real
    u_grid.set_active_scalars(label)
    return u_grid

u_plotter = pyvista.Plotter(shape=(1, 2))
u_grid_left = get_ugrid(H.subdomain_to_post_processing_solution[left_volume], "c")
u_grid_right = get_ugrid(H.subdomain_to_post_processing_solution[right_volume], "c")

u_plotter.subplot(0, 0)
u_plotter.add_mesh(u_grid_left, show_edges=False)
u_plotter.add_mesh(u_grid_right, show_edges=False)
u_plotter.view_xy()

u_plotter.subplot(0, 1)
exact_left = dolfinx.fem.Function(
    H.subdomain_to_post_processing_solution[left_volume].function_space
)
exact_left.interpolate(exact_solution_left(np))
exact_right = dolfinx.fem.Function(
    H.subdomain_to_post_processing_solution[right_volume].function_space
)
exact_right.interpolate(exact_solution_right(np))
u_grid_exact_left = get_ugrid(exact_left, "c_exact")
u_grid_exact_right = get_ugrid(exact_right, "c_exact")


u_plotter.add_mesh(u_grid_exact_left, show_edges=False)
u_plotter.add_mesh(u_grid_exact_right, show_edges=False)
u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("discontinuity_concentration.png")
```
