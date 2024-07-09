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

Deuterium ions at 200 eV were implanted in a 0.5 mm thick sample of high purity tungsten foil (PCW).

The ion beam with an incident flux of $2.5 \times 10^{19} \ \mathrm{D \ m^{-2} \ s^{-1}}$ was turned on for 400 s which corresponds to a fluence of $1.0 \times 10^{22} \ \mathrm{D \ m^{-2}}$

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
    D_0=4.1e-07,  # m2/s
    E_D=0.39,  # eV
)

model.materials = tungsten
import sympy as sp

imp_fluence = 1e22
incident_flux = 2.5e19  # beam strength from paper

imp_time = imp_fluence / incident_flux  # s

ion_flux = sp.Piecewise((incident_flux, F.t <= imp_time), (0, True))

source_term = F.ImplantationFlux(
    flux=ion_flux, imp_depth=4.5e-9, width=2.5e-9, volume=1  # H/m2/s  # m  # m
)

model.sources = [source_term]
# trap settings
w_atom_density = 6.3e28  # atom/m3

densities = [1.3e-3 * w_atom_density, 4e-4 * w_atom_density, None]
trap_1 = F.Trap(
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=0.87,
    density=densities[0],
    materials=tungsten,
)

trap_2 = F.Trap(
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=1.0,
    density=densities[1],
    materials=tungsten,
)

center = 4.5e-9
width = 2.5e-9
distribution = (
    1 / (width * (2 * sp.pi) ** 0.5) * sp.exp(-0.5 * ((F.x - center) / width) ** 2)
)

trap_3 = F.ExtrinsicTrap(
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=1.5,
    phi_0=ion_flux,
    n_amax=1e-01 * w_atom_density,
    f_a=distribution,
    eta_a=6e-4,
    n_bmax=1e-02 * w_atom_density,
    f_b=sp.Piecewise((1e6, F.x < 1e-6), (0, True)),
    eta_b=2e-4,
    materials=tungsten,
)

model.traps = [trap_1, trap_2, trap_3]
## Glueing parameters for table
from myst_nb import glue
for i, trap in enumerate(model.traps):
    for key, value in trap.__dict__.items():
        glue(f"small_trap{i}{key}", value, display=False)
    glue(f"small_trap{i}density", densities[i], display=False)
##

# boundary conditions
model.boundary_conditions = [F.DirichletBC(surfaces=[1, 2], value=0, field=0)]
implantation_temp = 293  # K
temperature_ramp = 8  # K/s

start_tds_0 = imp_time + 50  # s

model.T = F.Temperature(
    value=sp.Piecewise(
        (implantation_temp, F.t < start_tds_0),
        (implantation_temp + temperature_ramp * (F.t - start_tds_0), True),
    )
)

min_temp, max_temp_0 = implantation_temp, 700

model.dt = F.Stepsize(
    initial_value=0.5,
    stepsize_change_ratio=1.1,
    max_stepsize=lambda t: 0.5 if t > start_tds_0 else None,
    dt_min=1e-05,
    milestones=[start_tds_0],
)

model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-09,
    final_time=start_tds_0
    + (max_temp_0 - implantation_temp) / temperature_ramp,  # time to reach max temp
)

derived_quantities_small = F.DerivedQuantities(
    [
        F.TotalVolume("solute", volume=1),
        F.TotalVolume("retention", volume=1),
        F.TotalVolume("1", volume=1),
        F.TotalVolume("2", volume=1),
        F.TotalVolume("3", volume=1),
        F.HydrogenFlux(surface=1),
        F.HydrogenFlux(surface=2),
    ],
)

model.exports = [derived_quantities_small]

model.initialise()
model.run()
```

```{code-cell} ipython3
:tags: [hide-cell]

# referenced https://github.com/gabriele-ferrero/Titans_TT_codecomparison/blob/main/Festim%20models/TDS_Tungsten.py

# Material Setup, only W
tungsten = F.Material(
    id = 1,
    D_0 = 1.9e-07/(2)**0.5,  # m2/s
    E_D = 0.2,  # eV
)

model.materials = tungsten

incident_flux = 1.25e19  # beam strength from paper
imp_time = 4000 # s
ion_flux = sp.Piecewise((incident_flux, F.t <= imp_time), (0, True))

source_term = F.ImplantationFlux(
    flux=ion_flux, imp_depth=4.216e-9, width=2.5e-9, volume=1  # H/m2/s  # m  # m
)

model.sources = [source_term]
# trap settings
w_atom_density = 6.3e28  # atom/m3
densities = [1.364e-3 * w_atom_density, 3.639e-4 * w_atom_density, None]

trap_1 = F.Trap(
    k_0 = 1e13 / (6 * w_atom_density),
    E_k = 0.2,
    p_0 = 1e13,
    E_p = 0.834,
    density = densities[0],
    materials = tungsten,
)

trap_2 = F.Trap(
    k_0 = 1e13 / (6 * w_atom_density),
    E_k = 0.2,
    p_0 = 1e13,
    E_p = 0.959,
    density = densities[1],
    materials = tungsten,
)

center = 10e-9
width = 2e-9
distribution = 1/(1+sp.exp((F.x-center)/width))

trap_3 = F.Trap(
    k_0 = 1e13 / (6 * w_atom_density),
    E_k = 0.2,
    p_0 = 1e13,
    E_p = 1.496,
    density = 9.742e-2 * w_atom_density * distribution ,
    materials = tungsten
)

model.traps = [trap_1, trap_2, trap_3]

## Glueing parameters for table
for i, trap in enumerate(model.traps):
    for key, value in trap.__dict__.items():
        glue(f"big_trap{i}{key}", value, display=False)
    glue(f"big_trap{i}density", densities[i], display=False)
##

# boundary conditions
implantation_temp = 300  # K
temperature_ramp = 8  # K/s

start_tds_1 = imp_time + 990  # s

model.T = F.Temperature(
    value=sp.Piecewise(
        (implantation_temp, F.t < start_tds_1),
        (implantation_temp + temperature_ramp * (F.t - start_tds_1), True),
    )
)

model.boundary_conditions = [F.DirichletBC(surfaces=[1, 2], value=0, field=0)]

min_temp, max_temp_1 = implantation_temp, 800

model.dt = F.Stepsize(
    initial_value=0.1,
    stepsize_change_ratio=1.01,
    max_stepsize=lambda t: 0.5 if t > start_tds_1 - 10 else None,
    dt_min=1e-08,
    milestones=[start_tds_1],
)

model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-05,
    final_time=start_tds_1
    + (max_temp_1 - implantation_temp) / temperature_ramp,  # time to reach max temp
)

derived_quantities_big = F.DerivedQuantities(
    [
        F.TotalVolume("solute", volume=1),
        F.TotalVolume("retention", volume=1),
        F.TotalVolume("1", volume=1),
        F.TotalVolume("2", volume=1),
        F.TotalVolume("3", volume=1),
        F.HydrogenFlux(surface=1),
        F.HydrogenFlux(surface=2),
    ],
)

model.exports = [derived_quantities_big]

model.initialise()
model.run()
```

## Comparison with experimental data

The results produced by FESTIM are in good agreement with the experimental data. The grey areas represent the contribution of each trap to the global TDS spectrum.

```{code-cell} ipython3
:tags: [hide-input]

start_times = [start_tds_0, start_tds_1]
dqs = [derived_quantities_small, derived_quantities_big]
experimental_data_path = ["ogorodnikova-original.csv", "ogorodnikova-original-big.csv"]
max_temp = [max_temp_0, max_temp_1]

fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
fig.tight_layout()

for i, derived_quantities, start_tds in zip([0, 1], dqs, start_times):
    t = derived_quantities.t
    flux_left = derived_quantities.filter(fields="solute", surfaces=1).data
    flux_right = derived_quantities.filter(fields="solute", surfaces=2).data
    flux_total = -np.array(flux_left) - np.array(flux_right)

    t = np.array(t)
    temp = implantation_temp + 8 * (t - start_times[i])

    # plotting simulation data
    axs[i].plot(temp, flux_total, linewidth=3, label="FESTIM")

    # plotting trap contributions
    traps = [derived_quantities.filter(fields=f"{i}").data for i in range(1, 4)]
    contributions = [-np.diff(trap) / np.diff(t) for trap in traps]
    for cont in contributions:
        axs[i].plot(temp[1:], cont, linestyle="--", color="grey")
        axs[i].fill_between(temp[1:], 0, cont, facecolor="grey", alpha=0.1)

    # plotting original data
    experimental_tds = np.genfromtxt(experimental_data_path[i], delimiter=",")
    experimental_temp = experimental_tds[:, 0]
    experimental_flux = experimental_tds[:, 1]
    axs[i].scatter(experimental_temp, experimental_flux, color="green", label="original", s=16)

    axs[i].legend()
    axs[i].set_xlim(min_temp, max_temp[i])
    axs[i].set_ylim(bottom=[-1.25e18, -3e18][i], top=[0.6e19, 1.3e19][i])
    axs[i].set_ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
    axs[i].set_xlabel(r"Temperature (K)")

plt.show()
```

```{note}
The experimental data was taken from Figure 5 of the original experiment paper {cite}`ogorodnikova_deuterium_2003` using [WebPlotDigitizer](https://automeris.io/)
```

### Trap parameters for the small fluence case

|Trap|$k_0 \ (m^3 s^{-1})$|$E_k \ (\mathrm{eV})$|$E_p \ (\mathrm{eV})$|$p_0 \ (s^{-1})$|$n_i \ (m^{-3})$|
|:---|:--|:--|:--|:--|:------|
|1|{glue:text}`small_trap0k_0:.2e`|{glue:text}`small_trap0E_k:.2e`|{glue:text}`small_trap0E_p:.2e`|{glue:text}`small_trap0p_0:.2e`|{glue:text}`small_trap0density:.2e`|
|2|{glue:text}`small_trap1k_0:.2e`|{glue:text}`small_trap1E_k:.2e`|{glue:text}`small_trap1E_p:.2e`|{glue:text}`small_trap1p_0:.2e`|{glue:text}`small_trap1density:.2e`|
|3|{glue:text}`small_trap2k_0:.2e`|{glue:text}`small_trap2E_k:.2e`|{glue:text}`small_trap2E_p:.2e`|{glue:text}`small_trap2p_0:.2e`|**Extrinsic**|

### Trap parameters for the big fluence case

|Trap|$k_0 \ (m^3 s^{-1})$|$E_k \ (\mathrm{eV})$|$E_p \ (\mathrm{eV})$|$p_0 \ (s^{-1})$|$n_i \ (m^{-3})$|
|:---|:--|:--|:--|:--|:------|
|1|{glue:text}`big_trap0k_0:.2e`|{glue:text}`big_trap0E_k:.2e`|{glue:text}`big_trap0E_p:.2e`|{glue:text}`big_trap0p_0:.2e`|{glue:text}`big_trap0density:.2e`|
|2|{glue:text}`big_trap1k_0:.2e`|{glue:text}`big_trap1E_k:.2e`|{glue:text}`big_trap1E_p:.2e`|{glue:text}`big_trap1p_0:.2e`|{glue:text}`big_trap1density:.2e`|
|3|{glue:text}`big_trap2k_0:.2e`|{glue:text}`big_trap2E_k:.2e`|{glue:text}`big_trap2E_p:.2e`|{glue:text}`big_trap2p_0:.2e`|**Extrinsic**|
