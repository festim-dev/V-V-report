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

# Deuterium retention in tungsten

```{tags} 1D, TDS, trapping, transient
```

This validation case is a thermo-desorption spectrum measurement perfomed by Ogorodnikova et al. {cite}`ogorodnikova_deuterium_2003`.

Deuterium ions at 500 eV were implanted in a 0.5 mm thick sample of tungsten.

For the first case, the ion beam with an incident flux of $2.5 \times 10^{19} \ \mathrm{D \ m^{-2} \ s^{-1}}$ was turned on for 400 s which corresponds to a fluence of $1.0 \times 10^{22} \ \mathrm{D \ m^{-2}}$

For the second case, the ion beam was turned on for 4000 s which corresponds to a fluence of $1.0 \times 10^{23} \ \mathrm{D \ m^{-2}}$

The diffusivity of tungsten in the FESTIM model is as measured by Frauenfelder {cite}`frauenfelder_permeation_1968`.

To reproduce this experiment, three traps are needed: 2 intrinsic traps and 1 extrinsic trap.
The extrinsic trap represents the defects created during the ion implantation.

The time evolution of extrinsic traps density $n_i$ expressed in $\text{m}^{-3}$ is defined as:
\begin{equation}
    \frac{dn_i}{dt} = \varphi_0\:\left[\left(1-\frac{n_i}{n_{a_{max}}}\right)\:\eta_a \:f_a(x)+\left(1-\frac{n_i}{n_{b_{max}}}\right)\:\eta_b \:f_b(x)\right]
\end{equation}

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-cell]

import festim as F
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

model = F.Simulation()

vertices = np.concatenate([
    np.linspace(0, 30e-9, num=600),
    np.linspace(30e-9, 3e-6, num=300),
    np.linspace(3e-6, 5e-4, num=200),
])

model.mesh = F.MeshFromVertices(vertices)
# Material Setup, only W
tungsten = F.Material(
    id=1,
    D_0=2.9e-07,  # m2/s
    E_D=0.39,  # eV
)

model.materials = tungsten
import sympy as sp

imp_fluence = 5e21
incident_flux = 1e18  # beam strength from paper

imp_time = imp_fluence / incident_flux  # s

ion_flux = sp.Piecewise((incident_flux, F.t <= imp_time), (0, True))

source_term = F.ImplantationFlux(
    flux=ion_flux, imp_depth=10e-9, width=1e-9, volume=1  # H/m2/s  # m  # m
)

model.sources = [source_term]
# trap settings
w_atom_density = 6.3e28  # atom/m3
k_0 = tungsten.D_0 / (1.1e-10**2 * 6 * w_atom_density)
density = 3.8e29
damage_dist = 1 / (1 + sp.exp((F.x - 1e-06) / 2e-07))

trap_1 = F.Trap(
    k_0=k_0,
    E_k=tungsten.E_D,
    p_0=1e13,
    E_p=0.65,
    density=density * damage_dist,
    materials=tungsten,
)

trap_2 = F.Trap(
    k_0=k_0,
    E_k=tungsten.E_D,
    p_0=1e13,
    E_p=1.25,
    density=density * damage_dist,
    materials=tungsten,
)

trap_3 = F.Trap(
    k_0=k_0,
    E_k=tungsten.E_D,
    p_0=1e13,
    E_p=1.55,
    density=density * damage_dist,
    materials=tungsten
)

model.traps = [trap_2, trap_3]

# boundary conditions
model.boundary_conditions = [F.DirichletBC(surfaces=[1, 2], value=0, field=0)]
implantation_temp = 300  # K
temperature_ramp = 0.5  # K/s

start_tds = imp_time + 50  # s

model.T = F.Temperature(
    value=sp.Piecewise(
        (implantation_temp, F.t < start_tds),
        (implantation_temp + temperature_ramp * (F.t - start_tds), True),
    )
)

min_temp, max_temp = implantation_temp, 1173

model.dt = F.Stepsize(
    initial_value=0.5,
    stepsize_change_ratio=1.1,
    max_stepsize=lambda t: 2 if t > start_tds else None,
    dt_min=1e-05,
    milestones=[start_tds],
)

model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-09,
    final_time=start_tds
    + (max_temp - implantation_temp) / temperature_ramp,  # time to reach max temp
)

derived_quantities = F.DerivedQuantities(
    [
        F.TotalVolume("solute", volume=1),
        F.TotalVolume("retention", volume=1),
        F.HydrogenFlux(surface=1),
        F.HydrogenFlux(surface=2),
    ] + 
    [F.TotalVolume(f"{i + 1}", volume=1) for i in range(0,len(model.traps))]
)

model.exports = [derived_quantities]

model.initialise()
model.run()
```

## Comparison with experimental data

The results produced by FESTIM are in good agreement with the experimental data. The grey areas represent the contribution of each trap to the global TDS spectrum.

```{code-cell} ipython3
:tags: [hide-input]

t = derived_quantities.t
flux_left = derived_quantities.filter(fields="solute", surfaces=1).data
flux_right = derived_quantities.filter(fields="solute", surfaces=2).data
flux_total = -np.array(flux_left) - np.array(flux_right)

t = np.array(t)
temp = implantation_temp + temperature_ramp * (t - start_tds)

# plotting simulation data
plt.plot(temp, flux_total, linewidth=3, label="FESTIM")

# plotting trap contributions
traps = [derived_quantities.filter(fields=f"{i}").data for i in range(1, len(model.traps) + 1)]
contributions = [-np.diff(trap) / np.diff(t) for trap in traps]
for cont in contributions:
    plt.plot(temp[1:], cont, linestyle="--", color="grey")
    plt.fill_between(temp[1:], 0, cont, facecolor="grey", alpha=0.1)

# plotting original data
""" experimental_tds = np.genfromtxt(experimental_data_path[i], delimiter=",")
experimental_temp = experimental_tds[:, 0]
experimental_flux = experimental_tds[:, 1]
axs[i].scatter(experimental_temp, experimental_flux, color="green", label="original", s=16) """

plt.legend()
plt.xlim(min_temp, max_temp)
#plt.ylim(top=2e17)
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")

plt.show()
```

```{note}
The experimental data was taken from Figure 5 of the original experiment paper {cite}`ogorodnikova_deuterium_2003` using [WebPlotDigitizer](https://automeris.io/)
```
