---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: vv-festim-report-env
  language: python
  name: python3
---

# Deuterium retention in Fe-damaged tungsten

```{tags} 1D, TDS, trapping, transient
```

This validation case is a thermo-desorption spectrum measurement perfomed by Oya et al. {cite}`oya_thermal_2015`.

Deuterium ions at 500 eV were implanted in a 0.5 mm thick sample of tungsten damaged by Fe ions at different _dpa_ doses.

The ion beam with an incident flux of $1.0 \times 10^{18} \ \mathrm{D \ m^{-2} \ s^{-1}}$ was turned on for 5000 s which corresponds to a fluence of $5.0 \times 10^{21} \ \mathrm{D \ m^{-2}}$

The diffusivity of tungsten in the FESTIM model is as measured by Frauenfelder {cite}`frauenfelder_permeation_1968`.

To reproduce this experiment, eight traps are needed: 1 intrinsic trap and 7 damage-induced traps.

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-cell]

import festim as F
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

energies = [0.87, 1.0, 1.2, 1.35, 1.55, 1.73, 1.9]
dpa_to_densities = {
    0 : [],
    0.0003 : [1.8e-4, 0,      2.9e-5, 7.5e-5, 0,    0,      0],
    0.03   : [8.8e-5, 6.0e-5, 3.8e-5, 1.3e-4, 8.0e-5, 1.2e-4, 0],
    0.3    : [8.4e-5, 2.2e-4, 9.0e-5, 1.5e-4, 1.5e-4, 2.2e-4, 4.8e-4],
    1      : [8.5e-5, 5.2e-4, 2.8e-4, 2.2e-4, 2.4e-4, 3.4e-4, 5.9e-4],          
}

sample_depth = 5e-4
vertices = np.concatenate([
    np.linspace(0, 30e-9, num=700),
    np.linspace(30e-9, 3e-6, num=400),
    np.linspace(3e-6, sample_depth, num=200),
])      

# TODO: try with James' diffusivity
tungsten = F.Material(
    id=1,
    D_0=4.1e-07, # m2/s
    E_D=0.39, # eV
)

import sympy as sp

imp_fluence = 5e21
incident_flux = 1e18  # beam strength from paper

imp_time = imp_fluence / incident_flux  # s

print(imp_time)

ion_flux = sp.Piecewise((incident_flux, F.t <= imp_time), (0, True))

source_term = F.ImplantationFlux(
    flux=ion_flux, imp_depth=10e-9, width=1e-9, volume=1  # H/m2/s  # m  # m
)

implantation_temp = 300  # K
temperature_ramp = 0.5  # K/s
start_tds = imp_time + 60  # s

min_temp, max_temp = implantation_temp, 1173

def TDS(dpa):
    model = F.Simulation()

    model.mesh = F.MeshFromVertices(vertices)
    model.materials = tungsten
    model.sources = [source_term]

    # trap settings
    w_atom_density = 6.3e28  # atom/m3
    k_0 = tungsten.D_0 / (1.1e-10**2 * 6 * w_atom_density)

    damage_depth = 1.1e-6
    damage_dist = sp.Piecewise(
        (1 , F.x <= damage_depth),
        (0, True)
    )

    intrinsic_trap = F.Trap(
        k_0=k_0,
        E_k=tungsten.E_D,
        p_0=1e13,
        E_p=0.85,
        density=1.2e-5* w_atom_density,
        materials=tungsten,
    )

    densities = dpa_to_densities[dpa]

    damage_induced_traps = [
        F.Trap(
            k_0=k_0,
            E_k=tungsten.E_D,
            p_0=1e13,
            E_p=e,
            density=d * w_atom_density * damage_dist,
            materials=tungsten,
        ) for e, d in zip(energies, densities)
    ]

    model.traps = [intrinsic_trap] + damage_induced_traps

    # boundary conditions
    model.boundary_conditions = [F.DirichletBC(surfaces=[1, 2], value=0, field=0)]
    # SiervertsBC doesn't change the result much, and the pressure isn't specified
    # model.boundary_conditions = [F.SievertsBC(surfaces=[1, 2], S_0=4.52e21, E_S=0.3, pressure=1e-8)]
    
    model.T = F.Temperature(
        value=sp.Piecewise(
            (implantation_temp, F.t < start_tds),
            (implantation_temp + temperature_ramp * (F.t - start_tds), True),
        )
    )

    model.dt = F.Stepsize(
        initial_value=0.5,
        stepsize_change_ratio=1.2,
        max_stepsize=lambda t: 10 if t >= start_tds else None,
        dt_min=1e-05,
        milestones=[start_tds],
    )

    model.settings = F.Settings(
        absolute_tolerance=1e7,
        relative_tolerance=1e-10,
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
        [F.TotalVolume(f"{i + 1}", volume=1) for i, _ in enumerate(model.traps)],
        show_units=True
    )

    model.exports = [derived_quantities]

    model.initialise()
    model.run()

    return derived_quantities

dpa_to_dq = {}
for dpa in dpa_to_densities:
    dpa_to_dq[dpa] = TDS(dpa)
```

```{code-cell} ipython3
for col, energy in enumerate(energies):
    densities = []
    for dpa in dpa_to_densities:
        if(dpa == 0):
            continue
        densities.append(dpa_to_densities[dpa][col])
    plt.plot(list(dpa_to_densities.keys())[1:],densities, label=energies[col])
plt.legend()
```

## Comparison with experimental data

The results produced by FESTIM are in good agreement with the experimental data.

```{code-cell} ipython3
:tags: [hide-input]

""" # plotting trap contributions
traps = [derived_quantities.filter(fields=f"{i}").data for i in range(1, len(model.traps) + 1)]
contributions = [-np.diff(trap) / np.diff(t) for trap in traps]
for i, cont in enumerate(contributions):
    plt.plot(temp[1:], cont, linestyle="--", color=colors[i], label=f"trap{i}")
    plt.fill_between(temp[1:], 0, cont, facecolor="grey", alpha=0.1) """

dpa_values = dpa_to_densities.keys()

# color setup
colors = [(0.9 * (i % 2), 0.2 * (i % 4), 0.4 * (i % 3)) for i in range(1, len(dpa_values) + 1)]

experimental_tds = np.genfromtxt("oya_data.csv", delimiter=",", names=True)                                     
data = list(enumerate(zip(experimental_tds["T"], experimental_tds["flux"])))
experiment_dpa = experimental_tds["dpa"]
    
for j, dpa in enumerate(dpa_values):

    # plotting simulation data
    derived_quantities = dpa_to_dq[dpa]         

    t = np.array(derived_quantities.t)

    flux_left = derived_quantities.filter(fields="solute", surfaces=1).data
    flux_right = derived_quantities.filter(fields="solute", surfaces=2).data
    flux_total = -np.array(flux_left) - np.array(flux_right)

    temp = implantation_temp + temperature_ramp * (t - start_tds)

    plt.plot(temp, flux_total, linewidth=3, label="FESTIM", color=colors[j])


    # plot experimental
    x, y = list(zip(*((T, flux) for (i, (T, flux)) in data if np.isclose(experiment_dpa[i], dpa))))
    plt.scatter(x, y, color=colors[j], label=f"{dpa} dpa", s=16)

plt.legend()
plt.xlim(min_temp, max_temp)
plt.ylim(bottom=0, top=2e17)
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")

plt.show()
```

```{note}
The experimental data was taken from Figure 3 of the original experiment paper {cite}`oya_thermal_2015` using [WebPlotDigitizer](https://automeris.io/)
```

## Trap Parameters

### Damage-induced trap parameters

This table displays the neutron-induced traps' detrapping energy $E_p$ and their density per dpa dose in $m^{-3}$.

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd

dpa_no_zero = dpa_to_densities | {}
dpa_no_zero.pop(0)
data = {"E_p (eV)" : energies} | dpa_no_zero
dpa_frame = pd.DataFrame(data)

dpa_frame.columns = dpa_frame.columns.map(lambda s: f"{s:.1e} dpa" if not isinstance(s, str) else s)
dpa_frame.style \
    .relabel_index([f"Trap D{i}" for i,_ in enumerate(energies, 1)], axis=0) \
    .format("{:.2e}".format)
```
