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

# TMAP7 V&V Val-1c

```{tags} 1D, MES, transient
```

This verification case {cite}`ambrosek_verification_2008` from TMAP7's V&V document consists of a semi-infinite slab with no traps under a constant concentration $C_0$ on the first $10 m$ of the slab.

+++

## FESTIM Code

```{code-cell} ipython3
:tags: [hide-cell]

import festim as F
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

preloaded_length = 10 # m
C_0 = 1 # atom m^-3
D = 1 # 1 m^2 s^-1
exact_solution = C_0/2 * (
    2*sp.erf(F.x / sp.sqrt(4*D*F.t)) 
    - sp.erf((F.x - preloaded_length) / sp.sqrt(4*D*F.t))
    - sp.erf((F.x + preloaded_length) / sp.sqrt(4*D*F.t))
)

model = F.Simulation()

### Mesh Settings ###
vertices = np.concatenate([
    np.linspace(0, 10, 200),
    np.linspace(10, 100, 300),
])

model.mesh = F.MeshFromVertices(vertices)

model.boundary_conditions = [
    F.DirichletBC(surfaces=[1], value=0, field="solute")
]

initial_concentration = sp.Piecewise(
    (C_0, F.x <= preloaded_length),
    (0, True)
)
model.initial_conditions = [
    F.InitialCondition(field="solute", value=initial_concentration)
]

model.materials = [F.Material(id=1, D_0=D, E_D=0)]

model.T = F.Temperature(500)  # ignored in this problem

model.dt = F.Stepsize(
    initial_value=0.01,
    stepsize_change_ratio=1.1,
)

test_points = [0.5, preloaded_length, 12] #m
final_times = [100, 100, 50]
derived_quantities = F.DerivedQuantities(
    [F.PointValue("solute", x=v) for v in test_points]
)
model.exports = [derived_quantities]

model.settings = F.Settings(
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
    final_time=max(final_times)
)

model.initialise()
model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input]

fig, axs = plt.subplots(1, len(test_points))

for i, x in enumerate(test_points):
    plt.sca(axs[i])

    # plotting computed data
    computed_solution = derived_quantities[i].data
    t = derived_quantities[i].t
    plt.plot(t, computed_solution, label="FESTIM", linewidth = 3)

    # plotting exact solution
    exact_y = [exact_solution.subs({F.x : x, F.t : time}) for time in t]
    plt.plot(t, exact_y, label="Exact", color="green", linestyle="--")

    # plotting TMAP data
    tmap_data = np.genfromtxt(f"./tmap_1c_data/tmap_point_data_{i}.txt", delimiter=" ", skip_header=1)
    tmap_t = tmap_data[:, 0]
    tmap_solution = tmap_data[:, 1]
    plt.scatter(tmap_t, tmap_solution, label="TMAP7", color="purple")

    plt.title(f"x={x}m")
    if(i == 0):
        plt.ylabel("Concentration (atom / m^3)")
    plt.xlabel("t (s)")

fig.tight_layout()
plt.legend()
plt.show()
```

```{code-cell} ipython3

```
