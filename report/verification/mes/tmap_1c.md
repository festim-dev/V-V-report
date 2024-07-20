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

# Pre-loaded semi-infinite slab

```{tags} 1D, MES, transient
```

This verification case from case Val-1c of TMAP7's V&V report {cite}`ambrosek_verification_2008` consists of a semi-infinite slab with no traps under a constant concentration $c_0$ on the first $10 \ \mathrm{m}$ of the slab. The concentration at the boundaries is set to zero.

+++

## FESTIM Code

```{code-cell} ipython3
:tags: [hide-cell]

import festim as F
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

preloaded_length = 10  # m
C_0 = 1  # atom m^-3
D = 1  # 1 m^2 s^-1

model = F.Simulation()

### Mesh Settings ###
vertices = np.concatenate(
    [
        np.linspace(0, 10, 400),
        np.linspace(10, 100, 1000),
    ]
)

model.mesh = F.MeshFromVertices(vertices)

model.boundary_conditions = [F.DirichletBC(surfaces=[1], value=0, field="solute")]

initial_concentration = sp.Piecewise((C_0, F.x <= preloaded_length), (0, True))
model.initial_conditions = [
    F.InitialCondition(field="solute", value=initial_concentration)
]

model.materials = F.Material(id=1, D_0=D, E_D=0)

model.T = F.Temperature(500)  # ignored in this problem

model.dt = F.Stepsize(
    initial_value=0.01,
    stepsize_change_ratio=1.1,
)

test_points = [0.5, preloaded_length, 12]  # m
final_times = [100, 100, 50]
profile_times = [0.1] + np.linspace(0, 100, num=10).tolist()[1:]
derived_quantities = F.DerivedQuantities(
    [F.PointValue("solute", x=v) for v in test_points]
)
model.exports = [
    derived_quantities,
    F.TXTExport(
        field="solute", filename="./tmap_1c_data/c_profiles.txt", times=profile_times
    ),
]

model.settings = F.Settings(
    absolute_tolerance=1e-10, relative_tolerance=1e-10, final_time=max(final_times)
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

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.special import erf

norm = Normalize(vmin=0, vmax=max(profile_times))
cmap = cm.viridis

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

plt.figure()
filename = model.exports[1].filename
data = np.genfromtxt(filename, delimiter=",", names=True)
for i, t in enumerate(profile_times):
    label = "exact" if i == 0 else ""
    x = data["x"]
    y_name = f"t{t:.2e}s".replace("+", "").replace("-", "").replace(".", "")
    y = data[y_name]
    x, y = zip(*sorted(zip(x, y)))
    
    exact_y = exact_solution(np.array(x), t)

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
    computed_solution = derived_quantities[i].data
    t = np.array(derived_quantities[i].t)
    plt.plot(t, computed_solution, label="FESTIM", linewidth=3)

    # plotting exact solution
    plt.plot(t, exact_solution(x, t), label="Exact", color="green", linestyle="--")

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
