---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: vv-festim-report-env
  language: python
  name: python3
---

# Diffusion through a semi-infinite slab

```{tags} 1D, MES, transient
```

This verification case from TMAP7's V&V report {cite}`ambrosek_verification_2008` consists of a semi-infinite slab with no traps under a constant concentration $C_0$ boundary condition on the left.

+++

## FESTIM Code

```{code-cell} ipython3
:tags: [hide-cell]

import festim as F
import numpy as np
import sympy as sp
from scipy.special import erf
from matplotlib import pyplot as plt

C_0 = 1 # atom m^-3
D = 1 # m^2 s^-1
profile_time = 25 # s
exact_solution = lambda x, t: C_0 * (1 - erf(x / np.sqrt(4*D*t)))

model = F.Simulation()

### Mesh Settings ###
vertices = np.concatenate([
    np.linspace(0, 1, 100),
    np.linspace(1, 20, 200),
])

model.mesh = F.MeshFromVertices(vertices)

model.boundary_conditions = [
    F.DirichletBC(surfaces=[1], value=C_0, field="solute")
]

model.materials = [F.Material(id=1, D_0=D, E_D=0)]

model.T = F.Temperature(500)  # ignored in this problem

model.dt = F.Stepsize(
    initial_value=0.01,
    stepsize_change_ratio=1.1,
    milestones=[profile_time]
)

model.settings = F.Settings(
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
    final_time=30
)

test_point_x = 0.45
derived_quantities = F.DerivedQuantities(
    [F.PointValue("solute", x=test_point_x)]
)
model.exports = [
    derived_quantities, 
    F.TXTExport(
        field="solute", 
        filename="./tmap_1b_concentration.txt",
        times=[profile_time]
    ),
]

model.initialise()
model.run()
```

## Comparison with exact solution

The exact solution is given by

$$
    c(x, t) = c_0 \left( 1 - \mathrm{erf}\left( \frac{x}{2\sqrt{Dt}} \right) \right)
$$

```{code-cell} ipython3
:tags: [hide-input]

# plotting computed data
computed_data = np.genfromtxt("./tmap_1b_concentration.txt", delimiter=",", skip_header=1)
computed_x = computed_data[:, 0]
computed_solution = computed_data[:, 1]
plt.plot(computed_x, computed_solution, label="FESTIM", linewidth = 3)

# plotting exact solution
exact_y = exact_solution(np.array(computed_x), profile_time)
plt.plot(computed_x, exact_y, label="Exact", color="green", linestyle="--")

plt.title(f"Concentration profile at t={profile_time}s")
plt.ylabel("Concentration (atom / m^3)")
plt.xlabel("x (m)")

plt.legend()
plt.show()
```

```{code-cell} ipython3
:tags: [hide-input]

# plotting computed data
computed_solution = derived_quantities[0].data
t = derived_quantities[0].t
plt.plot(t, computed_solution, label="FESTIM", linewidth = 3)

# plotting exact solution
exact_y = exact_solution(test_point_x, np.array(t))

plt.plot(t, exact_y, label="Exact", color="green", linestyle="--")

# plotting TMAP data
tmap_data = np.genfromtxt("./tmap_point_data.txt", delimiter=" ", names=True)
tmap_t = tmap_data["t"]
tmap_solution = tmap_data["tmap"]
plt.scatter(tmap_t, tmap_solution, label="TMAP7", color="purple")

plt.title(f"Concentration profile at x={test_point_x}m")
plt.ylabel("Concentration (atom / m^3)")
plt.xlabel("t (s)")

plt.legend()
plt.show()
```

```{code-cell} ipython3

```
