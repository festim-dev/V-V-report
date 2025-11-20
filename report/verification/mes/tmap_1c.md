---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: vv-festim-report-env
  language: python
  name: python3
---

# Pre-loaded semi-infinite slab

```{tags} 1D, MES, transient
```

This verification case from case Val-1c of TMAP7's V&V report {cite}`ambrosek_verification_2008` consists of a semi-infinite slab with no traps under a constant concentration $c_0$ on the first $10 \ \mathrm{m}$ of the slab. The concentration at the boundaries is set to zero.

+++

## FESTIM Code

```{code-cell} ipython3
import festim as F

from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points


class PointValue(F.VolumeQuantity):
    def __init__(self, field, volume, x0, filename=None):
        super().__init__(field, volume, filename)
        self.x0 = x0

    def compute(self):
        u = self.field.solution
        mesh = u.function_space.mesh
        tree = bb_tree(mesh, mesh.geometry.dim)
        cell_candidates = compute_collisions_points(tree, self.x0)
        cell = compute_colliding_cells(mesh, cell_candidates, self.x0).array
        assert len(cell) > 0
        first_cell = cell[0]

        self.value = self.field.solution.eval(self.x0, first_cell)
        self.data.append(self.value)

class Profile(F.Profile1DExport):
    def __init__(self, field, volume, times=None):
        super().__init__(field=field, subdomain=volume, times=times)
        self.data = []   
        self.t = []      
        self.x = None    

    def compute(self, *args, **kwargs):
        super().compute(*args, **kwargs)

        last_profile = self.data[-1].copy()
        self.data[-1] = last_profile     
```

```{code-cell} ipython3
:tags: [hide-cell]

import festim as F
import numpy as np
import ufl
from matplotlib import pyplot as plt

preloaded_length = 10  # m
C_0 = 1  # atom m^-3
D = 1  # 1 m^2 s^-1

model = F.HydrogenTransportProblem()

H = F.Species("H")
model.species = [H]

### Mesh Settings ###
vertices = np.concatenate(
    [
        np.linspace(0, 10, 400),
        np.linspace(10, 100, 1000),
    ]
)

model.mesh = F.Mesh1D(vertices)

material = F.Material(D_0=D, E_D=0)

left_boundary = F.SurfaceSubdomain1D(id=1, x=0)
volume = F.VolumeSubdomain1D(id=1, borders=[0, 100], material=material)

model.subdomains = [left_boundary, volume]

model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_boundary, value=0, species=H)
]

initial_concentration = lambda x: ufl.conditional(x[0] <= preloaded_length, C_0, 0)
model.initial_conditions = [F.InitialConcentration(value=initial_concentration, species=H, volume=volume)]


model.temperature = 500  # ignored in this problem


test_points = [0.5, preloaded_length, 12]  # m
profile_times = [0.1] + np.linspace(0, 100, num=10).tolist()[1:]


profile_export = F.Profile1DExport(field=H, times=profile_times)

model.exports = [
    PointValue(field=H, volume=volume, x0=np.array([v, 0, 0])) for v in test_points
] + [profile_export
] 


model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=100)
model.settings.stepsize = F.Stepsize(
    initial_value=0.01,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=10,
    milestones=profile_times,
)

model.initialise()
model.run()

```

## Comparison with exact solution

This is a comparison of the computed concentration profiles at different times with the exact analytical solution (shown in dashed lines).

The analytical solution is given by

$$
    c(x, t) = \frac{c_0}{2}\left[ 
        2\mathrm{erf}\left(\frac{x}{2 \sqrt{Dt}}\right)
        - \mathrm{erf}\left(\frac{x - h}{2 \sqrt{Dt}}\right)
        - \mathrm{erf}\left(\frac{x + h}{2 \sqrt{Dt}}\right)
     \right]
$$

where $h$ is the thickness of the pre-loaded region.

+++

```{warning}
There is an existing bug with `ProfileExport1D` class. See (this associated issue for more details)[https://github.com/festim-dev/FESTIM/issues/1046].
```

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.special import erf


def exact_solution(x, t):
    sqrt_term = np.sqrt(4 * D * t)
    return (
        C_0
        / 2
        * (
            2 * erf(x / sqrt_term)
            - erf((x - preloaded_length) / sqrt_term)
            - erf((x + preloaded_length) / sqrt_term)
        )
    )


profile_export = model.exports[-1]
times = np.array(profile_export.t)

profiles_1d = [np.ravel(p) for p in profile_export.data]
data = np.stack(profiles_1d, axis=0)

x = profile_export.x
order = np.argsort(x)
x = x[order]
data = data[:, order]

norm = Normalize(vmin=times.min(), vmax=times.max())
cmap = cm.viridis

plt.figure()


for i, t in enumerate(profile_times):
    label = "exact" if i == 0 else ""
    y_name = f"t{t:.2e}s".replace("+", "").replace("-", "").replace(".", "")

    idx = np.argmin(np.abs(times - t))
    t = times[idx]
    y = data[idx, :]

    exact_y = exact_solution(x, t)

    plt.plot(x, exact_y, linestyle="dashed", color="tab:grey", linewidth=3, label=label)
    plt.plot(x, y, color=cmap(norm(t)))


plt.xlabel("x")
plt.ylabel("Concentration (atom / m^3)")

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # You can also set this to the range of your data
plt.colorbar(sm, label="Time (s)", ax=plt.gca())

plt.legend()
plt.show()
```

The results can also be compared with the results obtained by TMAP7.

```{code-cell} ipython3
:tags: [hide-input]

fig, axs = plt.subplots(1, len(test_points), sharey=True)

for i, x in enumerate(test_points):
    plt.sca(axs[i])

    # plotting computed data
    computed_solution = model.exports[i].data
    t = np.array(model.exports[i].t)
    plt.plot(t, computed_solution, label="FESTIM", linewidth=3)

    # plotting exact solution
    plt.plot(t, exact_solution(x, t), label="Exact", color="orange", linestyle="--")

    # plotting TMAP data
    tmap_data = np.genfromtxt(
        f"./tmap_1c_data/tmap_point_data_{i}.txt", delimiter=" ", names=True
    )
    tmap_t = tmap_data["t"]
    tmap_solution = tmap_data["tmap"]
    plt.scatter(tmap_t, tmap_solution, label="TMAP7", color="purple")

    plt.title(f"x={x} m")
    if i == 0:
        plt.ylabel("Concentration (atom / m$^3$)")
    plt.xlabel("t (s)")

fig.tight_layout()
plt.legend()
plt.show()
```
