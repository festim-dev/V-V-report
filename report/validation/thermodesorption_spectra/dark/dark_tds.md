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

The diffusivity of tungsten in the FESTIM model is as measured by Holzner et al. {cite}`holzner_2020`.

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

# ### Parameters ###

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
    dt_min=1e-1,
    max_stepsize=lambda t: 50 if t > t_imp + t_rest * 0.5 else None
)

model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-10,
    final_time=start_tds + (max_temp - min_temp) / Beta,  # time to reach max temp
)

derived_quantities = F.DerivedQuantities(
    [
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
sample_area = 12e-03 * 15e-03
experimental_tds = np.genfromtxt("tds_data/0.1_dpa.csv", delimiter=",")
experimental_temp = experimental_tds[:, 0]
experimental_flux = experimental_tds[:, 1] / sample_area
plt.scatter(experimental_temp, experimental_flux, color="black", label="original", s=16)

plt.legend()
plt.xlim(min_temp, max_temp)
plt.ylim(bottom=0, top=1e17)
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")
plt.title("0.1 dpa")

plt.show()
```

```{code-cell} ipython3
:tags: [hide]
dpa_values = [0, 0.001, 0.005, 0.023, 0.1, 0.23, 0.5, 2.5]
dpa_n_i = {
    0: [0, 0, 0, 0, 0],
    0.001: [1e+24, 2.5e+24, 1e+24, 1e+24, 2e+23], 
    0.005: [3.5e+24, 5e+24, 2.5e+24, 1.9e+24, 1.6e+24], 
    0.023: [2.2e+25, 1.5e+25, 6.5e+24, 2.1e+25, 6e+24], 
    0.1: [4.8e+25, 3.8e+25, 2.6e+25, 3.6e+25, 1.1e+25], 
    0.23: [5.4e+25, 4.4e+25, 3.6e+25, 3.9e+25, 1.4e+25], 
    0.5: [5.5e+25, 4.6e+25, 4e+25, 4.5e+25, 1.7e+25], 
    2.5: [6.8e+25, 6.1e+25, 5e+25, 5e+25, 2e+25],
}

## Gluing for making tables

from myst_nb import glue

for i, trap in enumerate(model.traps):
    for key, value in trap.__dict__.items():
        if (not isinstance(value, float)) and (not isinstance(value, int)):
            continue
        glue(f"trap{i}{key}", value, display=False)
    glue(f"ni{i}", n_i[i], display=False)

table = "" # useful for making the table
for i in range(0, 5):
    output = ""
    for j, dpa in enumerate(dpa_values[1:]):
        glue(f"ni_{i}_{j}", dpa_n_i[dpa][i], display=False)
        output += f"{{glue:text}}`ni_{i}_{j}:.2e`|"
    table += output + "\n"
print(table)

##


results = dict()
for dpa in reversed(dpa_values):
    neutron_induced_traps = []
    if dpa != 0:
        for i in range(1, 6):
            neutron_induced_traps.append(F.Trap(
                k_0 = k_0,
                E_k = E_D,
                p_0 = 1e13,
                E_p = E_p[i],
                density = dpa_n_i[dpa][i - 1] * damage_dist,
                materials = damaged_tungsten,
            ))

    ## can remove once data empty fix is live
    model.traps = [trap_1] + neutron_induced_traps
    derived_quantities = F.DerivedQuantities(
        [
            F.TotalVolume("solute", volume=1),
            F.TotalVolume("retention", volume=1),
            F.HydrogenFlux(surface=1),
            F.HydrogenFlux(surface=2),
        ],
    )
    ##

    model.exports = [derived_quantities]

    model.initialise()
    model.run()

    results[dpa] = {
        "t" : np.array(derived_quantities.t),
        "flux_left" : np.array(derived_quantities.filter(fields="solute", surfaces=1).data),
        "flux_right" : np.array(derived_quantities.filter(fields="solute", surfaces=2).data),
    }
```

```{code-cell} ipython3
:tags: [hide-input]
# Color Bar
from matplotlib import cm, colors

norm = colors.LogNorm(vmin=min(dpa_values[1:]), vmax=max(dpa_values))
colorbar = cm.viridis
sm = plt.cm.ScalarMappable(cmap=colorbar, norm=norm)

colors = ["black"] + [colorbar(norm(dpa)) for dpa in dpa_values[1:]]
for dpa, color in zip(dpa_values, colors):
    experimental_tds = np.genfromtxt(f"tds_data/{dpa}_dpa.csv", delimiter=",")
    experimental_temp = experimental_tds[:, 0]
    experimental_flux = experimental_tds[:, 1] / sample_area

    label = "undamaged" if color == "black" else ""
    plt.plot(experimental_temp, experimental_flux, color=color, linewidth=3, label=label)

    t = np.array(results[dpa]["t"])
    flux_total = -np.array(results[dpa]["flux_left"]) - np.array(results[dpa]["flux_right"])
    temp = min_temp + Beta * (t - start_tds)

    # plotting simulation data
    label = "FESTIM" if color == "black" else ""
    plt.plot(temp, flux_total, linewidth=2, color="grey", linestyle="--", label=label)

plt.legend()
plt.xlim(min_temp, max_temp)
plt.ylim(bottom=0, top=1e17)
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")

# Plotting color bar
from mpl_toolkits.axes_grid1 import make_axes_locatable
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(sm, cax=cax, label="Damage (dpa)")

plt.show()
```

```{note}
The experimental data was taken from {cite}`dark_modelling_2024_code`.
```

+++

The density distribution of the neutron-induced traps is $n_i \ f(x)$.

### Trap parameters, and densities for 0.1 dpa dose

|Trap|$k_0 \ (m^3 s^{-1})$|$E_k \ (\mathrm{eV})$|$E_p \ (\mathrm{eV})$|$p_0 \ (s^{-1})$|$n_i \ (m^{-3})$|
|:---|:--|:--|:--|:--|:------|
|1|{glue:text}`trap0k_0:.2e`|{glue:text}`trap0E_k:.2e`|{glue:text}`trap0E_p:.2e`|{glue:text}`trap0p_0:.2e`|{glue:text}`ni0:.2e`|
|D1|{glue:text}`trap1k_0:.2e`|{glue:text}`trap1E_k:.2e`|{glue:text}`trap1E_p:.2e`|{glue:text}`trap1p_0:.2e`|{glue:text}`ni1:.2e`|
|D2|{glue:text}`trap2k_0:.2e`|{glue:text}`trap2E_k:.2e`|{glue:text}`trap2E_p:.2e`|{glue:text}`trap2p_0:.2e`|{glue:text}`ni2:.2e`|
|D3|{glue:text}`trap3k_0:.2e`|{glue:text}`trap3E_k:.2e`|{glue:text}`trap3E_p:.2e`|{glue:text}`trap3p_0:.2e`|{glue:text}`ni3:.2e`|
|D4|{glue:text}`trap4k_0:.2e`|{glue:text}`trap4E_k:.2e`|{glue:text}`trap4E_p:.2e`|{glue:text}`trap4p_0:.2e`|{glue:text}`ni4:.2e`|
|D5|{glue:text}`trap5k_0:.2e`|{glue:text}`trap5E_k:.2e`|{glue:text}`trap5E_p:.2e`|{glue:text}`trap5p_0:.2e`|{glue:text}`ni5:.2e`|

### Density per induced trap for each dpa dose

|Trap|$0.001$|$0.005$|$0.023$|$0.1$|$0.23$|$0.5$|$2.5$|
|:---|:------|:------|:------|:----|:-----|:----|:----|
|D1|{glue:text}`ni_0_0:.2e`|{glue:text}`ni_0_1:.2e`|{glue:text}`ni_0_2:.2e`|{glue:text}`ni_0_3:.2e`|{glue:text}`ni_0_4:.2e`|{glue:text}`ni_0_5:.2e`|{glue:text}`ni_0_6:.2e`|
|D2|{glue:text}`ni_1_0:.2e`|{glue:text}`ni_1_1:.2e`|{glue:text}`ni_1_2:.2e`|{glue:text}`ni_1_3:.2e`|{glue:text}`ni_1_4:.2e`|{glue:text}`ni_1_5:.2e`|{glue:text}`ni_1_6:.2e`|
|D3|{glue:text}`ni_2_0:.2e`|{glue:text}`ni_2_1:.2e`|{glue:text}`ni_2_2:.2e`|{glue:text}`ni_2_3:.2e`|{glue:text}`ni_2_4:.2e`|{glue:text}`ni_2_5:.2e`|{glue:text}`ni_2_6:.2e`|
|D4|{glue:text}`ni_3_0:.2e`|{glue:text}`ni_3_1:.2e`|{glue:text}`ni_3_2:.2e`|{glue:text}`ni_3_3:.2e`|{glue:text}`ni_3_4:.2e`|{glue:text}`ni_3_5:.2e`|{glue:text}`ni_3_6:.2e`|
|D5|{glue:text}`ni_4_0:.2e`|{glue:text}`ni_4_1:.2e`|{glue:text}`ni_4_2:.2e`|{glue:text}`ni_4_3:.2e`|{glue:text}`ni_4_4:.2e`|{glue:text}`ni_4_5:.2e`|{glue:text}`ni_4_6:.2e`|
