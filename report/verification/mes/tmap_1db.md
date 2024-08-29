---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: festim-report
  language: python
  name: python3
---

# Strong Trapping Regime

```{tags} 1D, MES, transient, trapping
```

This verification case consists of a slab of depth $l = 1 \times 10^{-3} \ \mathrm{m}$ with one trap under the strong trapping regime.

A trapping parameter $\zeta$ is defined by

$$
    \zeta = \frac{\lambda ^ 2 \nu}{D_0 \rho} \exp \left(\frac{E_k - E_p}{k_B T}\right) + \frac{c_m}{n}
$$

where

$\lambda \ \mathrm{(m)}$ is the lattice parameter, \
$\nu \ (\mathrm{s}^{-1})$, the Debye frequency \
$n \ (\mathrm{s}^{-1})$ is the trapping site fraction, \
$c_\mathrm{m} \ (\text{H} \ \mathrm{m}^{-3})$ is the mobile concentration,

and the effective diffusivity $\mathrm{D_\text{eff}}$ is defined by

$$
    D_\text{eff} = \frac{D}{1 + \frac{1}{\zeta}}
$$

Then the breakthrough time is given by

$$
    \tau = \frac{l^2}{2\pi^2 D_\text{eff}}
$$

This analytical solution was obtained from TMAP7's V&V report {cite}`ambrosek_verification_2008`, case Val-1db.

+++

## FESTIM Code

```{code-cell} ipython3
:tags: [hide-cell]

import festim as F
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

l = 0.1e-3
D_0 = 1.9e-7
T = 1000
n_t = 1e-3
w_atom_density = 6.3382e28
N_A = const.N_A

my_model = F.Simulation()
vertices = np.concatenate(
    [np.linspace(0, 0.9 * l, num=200), np.linspace(0.9 * l, l, num=100)]
)
my_model.mesh = F.MeshFromVertices(vertices)
my_model.materials = F.Material(id=1, D_0=D_0, E_D=0.2)
my_model.T = T

my_model.boundary_conditions = [
    F.DirichletBC(surfaces=1, value=0.0088 * N_A, field=0),
    F.DirichletBC(surfaces=2, value=0, field=0),
]

trap = F.Trap(
    k_0=1.58e7 / N_A,
    E_k=0.2,
    p_0=1e13,
    E_p=2.5,
    density=n_t * w_atom_density,
    materials=my_model.materials[0],
)

my_model.traps = [trap]
my_model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-10,
    final_time=4000,  # s
)

my_model.dt = F.Stepsize(
    initial_value=1e-3,
    dt_min=1e-3,
    stepsize_change_ratio=1.1,
    max_stepsize=lambda t: 1 if 3250 >= t >= 3100 else None,
)

derived_quantities = F.DerivedQuantities([F.HydrogenFlux(surface=2)], show_units=True)

my_model.exports = [derived_quantities, F.XDMFExport("1", checkpoint=False)]

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input]

times = derived_quantities.t
computed_flux = derived_quantities.filter(surfaces=2).data

plt.plot(times, np.abs(computed_flux) / 2, label="FESTIM")

D_eff = D_0 * np.exp(-0.2 / F.k_B / T)
S = 2.9e-5 * np.exp(-1 / F.k_B / T)
c0m = np.sqrt(1e5) * S * 1.0525e5

time_exact = l**2 * n_t / (2 * c0m * D_eff) * w_atom_density / N_A

plt.axvline(x=time_exact, color="r", label="analytical")

plt.ylim(bottom=0)
plt.xlabel("Time (s)")
plt.ylabel("Downstream flux (H2/m2/s)")
# plt.ylim([1e-6, 0.5e17])
plt.xlim([3000, 3500])

plt.legend()
plt.show()
```
