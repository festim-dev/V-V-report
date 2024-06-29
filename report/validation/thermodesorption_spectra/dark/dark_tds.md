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

# Deuterium retention in neutron-damaged tungsten

```{tags} 1D, TDS, trapping, transient
```

+++

This validation case is a thermo-desorption spectrum measurement perfomed by {cite}`dark_modelling_2024`.

Several 0.8 mm thick samples of tungsten were damaged via annealing before being used to perform a TDS measurement.

An ion beam with an incident flux of $5.79 \times 10^{19} \ \mathrm{D \ m^{-2} \ s^{-1}}$ was turned on for $72\mathrm{h}$ with an implantation temperature of $370\mathrm{K}$. The sample then rested for $12\mathrm{h}$ at $295\mathrm{K}$ before beggining the TDS measurement at $300\mathrm{K}$ with a temperature ramp of $0.05\mathrm{K}/s$.

To reproduce this experiment, six traps are needed: 1 intrinsic trap and 5 neutron induced traps.
The trap densities for the neutron induced traps were fitted by {cite}`dark_modelling_2024` for each _dpa_ amount using FESTIM's `NeutronInducedTrap`.

The damage distribution for the neutron-induced traps is as follows:
$$
    f(x) = \frac{1}{1 + \exp{ \frac{\left( x - x_0 \right)}{\Delta x} }}
$$

Tables with the relevant trap parameters are at the bottom of the page. TODO: ADD TRAP PARAMETERS

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-input, hide-output]

# Setup
import festim as F
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Parameters


# Deuterium in Tungsten [35]
D_0 = 1.6 * 1e-7 # m^2 s^-1
E_D = 0.28 # eV
w_atom_density = 6.3222e28

sample_depth = 0.8e-3 # m

# Table 2
T_imp = 370 # K
T_rest = 295 # K
R_p = 0.7e-9 # m
sigma = 0.5e-9 # m
t_imp = 72 * 3600 # s
t_rest = 12 * 3600 # s
Beta = 0.05 # K s^-1

fluence = 1.5e25
flux = fluence / t_imp

# Deuterium Beam Profile (S = flux * f(x))
distribution = (1 / (sigma * (2 * np.pi) ** 0.5) * sp.exp(-0.5 * ((F.x - R_p) / sigma) ** 2))

model = F.Simulation()

vertices = np.concatenate(
    [
        np.linspace(0, 3e-9, num=100),
        np.linspace(3e-9, 8e-6, num=100),
        np.linspace(8e-6, 8e-5, num=100),
        np.linspace(8e-5, 8e-4, num=100),
    ]
)
model.mesh = F.MeshFromVertices(vertices)

# material
damaged_tungsten = F.Material(1, D_0, E_D, borders=[0, sample_depth])
model.materials = damaged_tungsten

# hydrogen source
ion_flux = sp.Piecewise((flux*distribution, F.t < t_imp), (0, True))
source_term = F.Source(value=ion_flux, volume=1, field=0)
model.sources = [source_term]

# boundary conditions
model.boundary_conditions = [F.DirichletBC(surfaces=[1, 2], value=0, field="solute")]

# temperature
start_tds = t_imp + t_rest  # s
min_temp, max_temp = 300, 1000

model.T = F.Temperature(
    value=sp.Piecewise(
        (T_imp, F.t < t_imp),
        (T_rest, F.t < start_tds),
        (min_temp + Beta * (F.t - start_tds), True),
    )
)

# trap settings
k_0 = D_0 / (1.1e-10**2 * 6 * w_atom_density)

trap_1 = F.Trap(
    E_k = 0.28,
    k_0 = k_0,
    E_p = 1.04,
    p_0 = 1e13,
    density = 2.4e22,
    materials = damaged_tungsten
)

damage_dist = 1 / (1 + sp.exp((F.x - 2.5e-06) / 5e-07))
n_i = np.array([4.8, 3.8, 2.6, 3.6, 1.1]) * 1e25
neutron_induced_traps = []
E_p = [1.15, 1.35, 1.65, 1.85, 2.05]

for i in range(5):
    neutron_induced_traps.append(F.Trap(
        k_0 = k_0,
        E_k = E_D,
        p_0 = 1e13,
        E_p = E_p[i],
        density = n_i[i] * damage_dist,
        materials = damaged_tungsten,
    ))


model.traps = [trap_1] + neutron_induced_traps

model.dt = F.Stepsize(
    initial_value=1,
    stepsize_change_ratio=1.1,
    t_stop=t_imp + t_rest * 0.5,
    dt_min=1e-1,
    stepsize_stop_max=50,
)

model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-10,
    final_time=start_tds + (max_temp - min_temp) / Beta,  # time to reach max temp
)

derived_quantities = F.DerivedQuantities(
    [
        F.AverageVolume("T", volume=1),
        F.TotalVolume("solute", volume=1),
        F.TotalVolume("retention", volume=1),
        F.TotalVolume("1", volume=1),
        F.TotalVolume("2", volume=1),
        F.TotalVolume("3", volume=1),
        F.TotalVolume("4", volume=1),
        F.TotalVolume("5", volume=1),
        F.TotalVolume("6", volume=1),
        F.HydrogenFlux(surface=1),
        F.HydrogenFlux(surface=2),
    ],
)

model.exports = [derived_quantities]

model.initialise()
model.run()
```

```{note}
Implementation details from {cite}`dark_modelling_2024_code`.
```

+++

## Comparison with experimental data

The results produced by FESTIM are in good agreement with the experimental data. The grey areas represent the contribution of each trap to the global TDS spectrum.

```{code-cell} ipython3
:tags: [hide-input]

t = derived_quantities.t
flux_left = derived_quantities.filter(fields="solute", surfaces=1).data
flux_right = derived_quantities.filter(fields="solute", surfaces=2).data
flux_total = -np.array(flux_left) - np.array(flux_right)

t = np.array(t)
temp = min_temp + Beta * (t - start_tds)

# plotting simulation data
plt.plot(temp, flux_total, linewidth=3, label="FESTIM")

# plotting trap contributions
traps = [derived_quantities.filter(fields=f"{i}").data for i in range(1, 7)]
contributions = [-np.diff(trap) / np.diff(t) for trap in traps]

colors = [(0.9*(i % 2), 0.2*(i % 4), 0.4*(i % 3)) for i in range(6)]

for i, cont in enumerate(contributions):
    if(i == 0):
        label = "Trap 1"
    else:
        label = f"Trap D{i}"

    plt.plot(temp[1:], cont, linestyle="--", color=colors[i], label=label)
    plt.fill_between(temp[1:], 0, cont, facecolor="grey", alpha=0.1)


# plotting original data
""" experimental_tds = np.genfromtxt("dark-original.csv", delimiter=",")
experimental_temp = experimental_tds[:, 0]
experimental_flux = experimental_tds[:, 1]
plt.scatter(experimental_temp, experimental_flux, color="green", label="original", s=16)"""

plt.legend()
plt.xlim(min_temp, max_temp)
plt.ylim(bottom=0, top=1e17)
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")

plt.show()
```

```{note}
The experimental data was taken from {cite}`dark_modelling_2024` using [WebPlotDigitizer](https://automeris.io/).
```
