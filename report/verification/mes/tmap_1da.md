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

# Effective diffusivity regime

```{tags} 1D, MES, transient, trapping
```

This verification case consists of a slab of depth $l = 1 \times 10^{-3} \ \mathrm{m}$ with one trap under the effective diffusivity regime.

A trapping parameter $\zeta$ is defined by

$$
    \zeta = \frac{\lambda ^ 2 \nu}{D_0 \rho} \exp \left(\frac{E_k - E_p}{k_B T}\right) + \frac{c_m}{\rho}
$$

where

$\lambda \ \mathrm{(m)}$ is the lattice parameter, \
$\nu \ (\mathrm{s}^{-1})$, \
$\rho$ is the trapping site fraction, \
$c_m (\text{atom} \ \mathrm{m}^{-3})$ is the mobile atom concentration,

and the effective diffusivity $\mathrm{D_\text{eff}}$ is defined by

$$
    D_\text{eff} = \frac{D}{1 + \frac{1}{\zeta}}
$$

Then with a breakthrough time $\tau = \frac{l^2}{2\pi^2 D_\text{eff}}$, the exact solution for flux is

$$
    J = \frac{}{} \left[ 1 + 2\sum_{m=1}^\infty (-1)^m \exp \left( -m^2 \frac{t}{\tau} \right) \right]
$$

This analytical solution was obtained from TMAP7's V&V report {cite}`ambrosek_verification_2008`, case Val-1da.

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
rho_w = 6.3382e28
E_k = 0.2
E_p = 1
T = 1000
D = D_0 * np.exp(-E_k / (F.k_B * T))
S = 2.9e-5 * np.exp(-1 / (F.k_B * T))
sample_depth = 1e-3

c_m = (1e5) ** 0.5 * S * 1.0525e5
zeta = (6 * rho_w * np.exp((E_k - E_p) / (F.k_B * T)) + c_m * N_A) / (rho_w * 1e-3)
D_eff = D / (1 + 1 / zeta)

model = F.Simulation()
model.mesh = F.MeshFromVertices(vertices=np.linspace(0, sample_depth, num=100))
model.materials = F.Material(id=1, D_0=D_0, E_D=E_k)
model.T = F.Temperature(value=T)

model.boundary_conditions = [
    F.DirichletBC(surfaces=1, value=c_m * N_A, field=0),
    F.DirichletBC(surfaces=2, value=0, field=0),
]

trap = F.Trap(
    k_0=1.58e7 / N_A,
    E_k=E_k,
    p_0=1e13,
    E_p=E_p,
    density=1e-3 * rho_w,
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
tau = sample_depth**2 / (np.pi**2 * D_eff)
exp_part = np.exp(-m ** 2 * t[:, None] / tau)

# Calculate the 'add' part for all m values and sum them up for each t
add = 2 * (-1) ** m * exp_part
exact_solution = 1 + add.sum(axis=1)  # Sum along the m dimension and add 1 for the initial ones

exact_solution = N_A * exact_solution * c_m * D / (2 * sample_depth)

plt.plot(t, exact_solution, linestyle="--", color="green", label="exact")

plt.xlabel("Time (s)")
plt.ylabel("Downstream flux (H/m2/s)")

plt.legend()
plt.show()
```
