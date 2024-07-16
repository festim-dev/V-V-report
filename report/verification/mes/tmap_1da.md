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

# TMAP7 V&V Val-1da

```{tags} 1D, MES, transient
```

This verification case {cite}`ambrosek_verification_2008` from TMAP7's V&V document consists of a slab of depth $1 \times 10^(-3) \ m$ with one trap under the effective diffusivity regime.

TODO: Add expression for the permeation transient (analytical flux).

+++

## FESTIM Code

```{code-cell} ipython3
:tags: [hide-cell]

# referenced https://github.com/gabriele-ferrero/Titans_TT_codecomparison/blob/main/Festim_models/WeakTrap.py

import festim as F
import numpy as np
import matplotlib.pyplot as plt

D_0 = 1.9e-7
N_A = 6.0221408e23
rhow = 6.3382e28
N_Tis = 6 * rhow
T = 1000
D = D_0 * np.exp(-0.2 / F.k_B / T)
S = 2.9e-5 * np.exp(-1 / F.k_B / T)
c_0 = (1e5) ** 0.5 * S * 1.0525e5
zeta = (N_Tis * np.exp((0.2 - 1) / (F.k_B * T)) + c_0 * N_A) / (rhow * 1e-3)
Deff = D / (1 + 1 / zeta)

sample_depth = 1e-3

model = F.Simulation()
model.mesh = F.MeshFromVertices(vertices=np.linspace(0, sample_depth, num=1001))
model.materials = F.Material(id=1, D_0=D_0, E_D=0.2)
model.T = F.Temperature(value=T)

model.boundary_conditions = [
    F.DirichletBC(surfaces=1, value=0.0088 * N_A, field=0),
    F.DirichletBC(surfaces=2, value=0, field=0),
]

rho_n = 6.338e28
trap = F.Trap(
    k_0=1.58e7 / N_A,
    E_k=0.2,
    p_0=1e13,
    E_p=1,
    density=1e-3 * rho_n,
    materials=model.materials[0],
)

model.traps = [trap]

model.settings = F.Settings(
    absolute_tolerance=1e10, relative_tolerance=1e-10, final_time=100  # s
)

model.dt = F.Stepsize(0.5)

derived_quantities = F.DerivedQuantities([F.HydrogenFlux(surface=2)])
model.exports = [derived_quantities]

model.initialise()
model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input]

# plot computed solution
t = np.array(derived_quantities.t)
computed_solution = derived_quantities.filter(surfaces=2).data
plt.plot(t, np.abs(computed_solution) / 2, label="FESTIM", linewidth=3)

# plot exact solution
t = np.array(t)
m = np.arange(1, 10001)

# Calculate the exponential part for all m values at once
exp_part = np.exp(-m ** 2* np.pi**2 * Deff * t[:, None] / sample_depth**2)

# Calculate the 'add' part for all m values and sum them up for each t
add = 2 * (-1) ** m * exp_part
exact_solution = 1 + add.sum(axis=1)  # Sum along the m dimension and add 1 for the initial ones

exact_solution = N_A * exact_solution * c_0 * D / sample_depth / 2

plt.plot(t, exact_solution, linestyle="--", color="green")

plt.xlabel("Time (s)")
plt.ylabel("Downstream flux (H/m2/s)")

plt.legend()
plt.show()
```
