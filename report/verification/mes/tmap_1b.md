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

# TMAP7 V&V Val-1b

```{tags} 1D, MES, transient
```

This verification case {cite}`ambrosek_verification_2008` from TMAP7's V&V document consists of a semi-infinite slab with no traps under a constant concentration $C_0$ boundary condition on the left.

+++

## FESTIM Code

```{code-cell}
:tags: [hide-cell]

import festim as F
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

C_0 = 1 # atom m^-3
D = 1 # 1 m^2 s^-1
profile_time = 25 # s
exact_solution = C_0 * (1 - sp.erf(F.x / sp.sqrt(4*D*F.t)))

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

```{code-cell}
:tags: [hide-input]

# plotting computed data
computed_data = np.genfromtxt("./tmap_1b_concentration.txt", delimiter=",", skip_header=1)
computed_x = computed_data[:, 0]
computed_solution = computed_data[:, 1]
plt.plot(computed_x, computed_solution, label="FESTIM", linewidth = 3)

# plotting exact solution
exact_y = [exact_solution.subs({F.x : x, F.t : profile_time}) for x in computed_x]
plt.plot(computed_x, exact_y, label="Exact", color="green", linestyle="--")

plt.title(f"Concentration profile at t={profile_time}s")
plt.ylabel("Concentration (atom / m^3)")
plt.xlabel("x (m)")

plt.legend()
plt.show()
```

```{code-cell}
:tags: [hide-input]

# plotting computed data
computed_solution = derived_quantities[0].data
t = derived_quantities[0].t
plt.plot(t, computed_solution, label="FESTIM", linewidth = 3)

# plotting exact solution
exact_y = [exact_solution.subs({F.x : test_point_x, F.t : time}) for time in t]
plt.plot(t, exact_y, label="Exact", color="green", linestyle="--")

# plotting TMAP data
tmap_data = np.genfromtxt("./tmap_point_data.txt", delimiter=" ")
tmap_t = tmap_data[:, 0]
tmap_solution = tmap_data[:, 1]
plt.scatter(tmap_t, tmap_solution, label="TMAP7", color="purple")

plt.title(f"Concentration profile at x={test_point_x}s")
plt.ylabel("Concentration (atom / m^3)")
plt.xlabel("t (s)")

plt.legend()
plt.show()
```
