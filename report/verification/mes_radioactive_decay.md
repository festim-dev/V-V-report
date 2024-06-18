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

# Radioactive decay 1D

This example is a radioactive decay (`RadioactiveDecay`) problem on simple unit interval with a uniform mobile concentration and no boundary condition.


In this problem, for simplicity, we don't set any traps and we model an isolated domain (no flux boundary conditions) to mimick a simple 0D case. Diffusion can therefore be neglected and the problem is:

$$
\begin{align}
    \frac{\partial c}{\partial t} = - \lambda \ c &  \quad \text{on }  \Omega  \\
\end{align}
$$(problem_decay)

The exact solution for mobile concentration is:

$$
\begin{equation}
    c_\mathrm{exact} = c_0 e^{-\lambda t}
\end{equation}
$$(c_exact_decay)

Here, $c_0$ is the initial concentration and $\lambda$ is the decay constant (in $s^{-1}$). We can then run a FESTIM model with these conditions and compare the numerical solution with $c_\mathrm{exact}$.

We can then run a FESTIM model with these conditions and compare the numerical solution with $c_\mathrm{exact}$.

+++

## FESTIM Code

```{code-cell} ipython3
:tags: [hide-cell]

import festim as F
import numpy as np
import matplotlib.pyplot as plt


initial_concentration = 3


def run_model(half_life):
    my_model = F.Simulation()

    my_model.initial_conditions = [
        F.InitialCondition(value=initial_concentration, field="solute")
    ]

    my_model.dt = F.Stepsize(
        initial_value=0.05,
        stepsize_change_ratio=1.01,
        dt_min=1e-05,
    )

    my_model.materials = F.Material(id=1, D_0=1, E_D=0)
    my_model.T = F.Temperature(value=300)

    my_model.boundary_conditions = []  # no BCs to have 0D case

    my_model.mesh = F.MeshFromVertices(np.linspace(0, 1, 1001))

    my_model.sources = [
        F.RadioactiveDecay(decay_constant=np.log(2) / half_life, volume=1)
    ]

    derived_quantities = F.DerivedQuantities([F.TotalVolume("solute", volume=1)])
    my_model.exports = [derived_quantities]

    my_model.settings = F.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        final_time=5 * half_life,  # s
    )

    my_model.initialise()
    my_model.run()

    time = derived_quantities.t
    concentration = derived_quantities[0].data
    return time, concentration
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-cell]

tests = []
for half_life in np.linspace(1, 100, 5):
    tests.append((*run_model(half_life), half_life))
```

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import cm
from matplotlib.colors import LogNorm

norm = LogNorm(vmin=1e-2, vmax=100)

for time, concentration, half_life in tests:
    plt.plot(
        time,
        concentration,
        label=f"half-life = {half_life:.2f} s",
        color=cm.Blues(norm(half_life)),
        linewidth=3,
    )
    exact = initial_concentration * np.exp(-np.log(2) / half_life * np.array(time))
    plt.plot(time, exact, "--", color="tab:grey")

plt.legend()
plt.ylim(bottom=0)
plt.xlabel("Time (s)")
plt.ylabel("Concentration (H/m3)")
```
