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

# Deuterium retention in self-damaged tungsten

```{tags} 1D, TDS, trapping, transient
```

+++

This validation case is a thermo-desorption spectrum measurement on damaged tungsten. The complete description is available in {cite}`dark_modelling_2024`.

Several 0.8 mm thick samples of tungsten were self-damaged and annealing before being used to perform a TDS measurement.

An ion beam with an incident flux of $5.79 \times 10^{19} \ \mathrm{D \ m^{-2} \ s^{-1}}$ was turned on for $72 \ \mathrm{h}$ with an implantation temperature of $370 \ \mathrm{K}$. The sample then rested for $12 \ \mathrm{h}$ at $295 \ \mathrm{K}$ before beginning the TDS measurement at $300 \ \mathrm{K}$ with a temperature ramp of $0.05 \ \mathrm{K}/s$.

To reproduce this experiment, six traps are needed: 1 intrinsic trap and 5 neutron induced traps.
The trap densities for the neutron induced traps were fitted by {cite}`dark_modelling_2024` for each _dpa_ dose using FESTIM's `NeutronInducedTrap`.

The damage distribution for the damage-induced traps is as follows:

$$
    f(x) = \frac{1}{1 + \exp{ \frac{\left( x - x_0 \right)}{\Delta x} }}
$$

Tables with the relevant trap parameters are at the bottom of the page.

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
D_0 = 1.6e-7 # m^2 s^-1
E_D = 0.28 # eV
w_atom_density = 6.3222e28

sample_thickness = 0.8e-3 # m

# Table 2 from Dark et al 10.1088/1741-4326/ad56a0
T_imp = 370 # K
T_rest = 295 # K
R_p = 0.7e-9 # m
sigma = 0.5e-9 # m
t_imp = 72 * 3600 # s
t_rest = 12 * 3600 # s
Beta = 0.05 # K s^-1

fluence = 1.5e25
flux = fluence / t_imp

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

# ### Material ###
damaged_tungsten = F.Material(1, D_0, E_D, borders=[0, sample_thickness])
model.materials = damaged_tungsten

# ### Source ###

# Deuterium Beam Profile (S = flux * f(x))
distribution = (1 / (sigma * (2 * np.pi) ** 0.5) * sp.exp(-0.5 * ((F.x - R_p) / sigma) ** 2))

ion_flux = sp.Piecewise((flux*distribution, F.t < t_imp), (0, True))
source_term = F.Source(value=ion_flux, volume=1, field=0)
model.sources = [source_term]


# ### Boundary Conditions ###
model.boundary_conditions = [F.DirichletBC(surfaces=[1, 2], value=0, field="solute")]


# ### Temperature ###
start_tds = t_imp + t_rest  # s
min_temp, max_temp = 300, 1000

model.T = F.Temperature(
    value=sp.Piecewise(
        (T_imp, F.t < t_imp),
        (T_rest, F.t < start_tds),
        (min_temp + Beta * (F.t - start_tds), True),
    )
)

# ### Trap Settings ###
k_0 = D_0 / (1.1e-10**2 * 6 * w_atom_density)
damage_dist = 1 / (1 + sp.exp((F.x - 2.5e-06) / 5e-07))
n_i = np.array([2.4e-3, 4.8, 3.8, 2.6, 3.6, 1.1]) * 1e25
neutron_induced_traps = []
E_p = [1.04, 1.15, 1.35, 1.65, 1.85, 2.05]

trap_1 = F.Trap(
    E_k = 0.28,
    k_0 = k_0,
    E_p = 1.04,
    p_0 = 1e13,
    density = n_i[0],
    materials = damaged_tungsten
)

for i in range(1, 6):
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
    max_stepsize=50,
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
trap_data = [derived_quantities.filter(fields=f"{i}").data for i in range(1, 7)]
contributions = [-np.diff(trap) / np.diff(t) for trap in trap_data]

colors = [(0.9*(i % 2), 0.2*(i % 4), 0.4*(i % 3)) for i in range(6)]

for i, cont in enumerate(contributions):
    if(i == 0):
        label = "Trap 1"
    else:
        label = f"Trap D{i}"

    plt.plot(temp[1:], cont, linestyle="--", color=colors[i], label=label)
    plt.fill_between(temp[1:], 0, cont, facecolor="grey", alpha=0.1)


# plotting original data
experimental_tds = np.genfromtxt("dark-original.csv", delimiter=",")
experimental_temp = experimental_tds[:, 0]
experimental_flux = experimental_tds[:, 1]
plt.scatter(experimental_temp, experimental_flux, color="black", label="original", s=16)

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

```{code-cell} ipython3
:tags: [hide-cell]

from myst_nb import glue

for i, trap in enumerate(model.traps):
    for key, value in trap.__dict__.items():
        glue(f'trap{i}{key}', value, display=False)
    glue(f'ni{i}', n_i[i], display=False)
```

The density distribution of the neutron-induced traps is $n_i \ f(x)$.

|Trap|k_0|E_k|E_p|p_0|n_i|
|:---|:--|:--|:--|:--|:------|
|1|{glue}`trap0k_0`|{glue}`trap0E_k`|{glue}`trap0E_p`|{glue}`trap0p_0`|{glue}`ni0`|
|D1|{glue}`trap1k_0`|{glue}`trap1E_k`|{glue}`trap1E_p`|{glue}`trap1p_0`|{glue}`ni1`|
|D2|{glue}`trap2k_0`|{glue}`trap2E_k`|{glue}`trap2E_p`|{glue}`trap2p_0`|{glue}`ni2`|
|D3|{glue}`trap3k_0`|{glue}`trap3E_k`|{glue}`trap3E_p`|{glue}`trap3p_0`|{glue}`ni3`|
|D4|{glue}`trap4k_0`|{glue}`trap4E_k`|{glue}`trap4E_p`|{glue}`trap4p_0`|{glue}`ni4`|
|D5|{glue}`trap5k_0`|{glue}`trap5E_k`|{glue}`trap5E_p`|{glue}`trap5p_0`|{glue}`ni5`|
