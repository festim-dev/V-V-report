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

The density distribution of the neutron-induced traps is $n_i \ f(x)$.

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import festim as F
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# # ### Parameters ###
D_0 = 1.6e-7  # m^2 s^-1
E_D = 0.28  # eV
w_atom_density = 6.3222e28
sample_thickness = 0.8e-3  # m
sample_area = 12e-03 * 15e-03

detrapping_energies = [1.15, 1.35, 1.65, 1.85, 2.05]
dpa_n_i = {
    0.001: [1e24, 2.5e24, 1e24, 1e24, 2e23],
    0.005: [3.5e24, 5e24, 2.5e24, 1.9e24, 1.6e24],
    0.023: [2.2e25, 1.5e25, 6.5e24, 2.1e25, 6e24],
    0.1: [4.8e25, 3.8e25, 2.6e25, 3.6e25, 1.1e25],
    0.23: [5.4e25, 4.4e25, 3.6e25, 3.9e25, 1.4e25],
    0.5: [5.5e25, 4.6e25, 4e25, 4.5e25, 1.7e25],
    2.5: [5.8e25, 6.5e25, 4.5e25, 5.5e25, 2e25],  # re-fit
}

# Table 2 from Dark et al 10.1088/1741-4326/ad56a0
T_imp = 370  # K
T_rest = 295  # K
R_p = 0.7e-9  # m
sigma = 0.5e-9  # m
t_imp = 72 * 3600  # s
implantation_time = t_imp
t_rest = 12 * 3600  # s
resting_time = t_rest
Beta = 3 / 60  # K s^-1
fluence = 1.5e25
flux = fluence / t_imp
start_tds = t_imp + t_rest  # s
min_temp, max_temp = 300, 1000


center = 0.7e-9
width = 0.5e-9


def festim_sim(densities):

    model = F.Simulation()
    vertices = np.concatenate(
        [
            np.linspace(0, 3e-9, num=100),
            np.linspace(3e-9, 8e-6, num=100),
            np.linspace(8e-6, 8e-5, num=100),
            np.linspace(8e-5, sample_thickness, num=100),
        ]
    )
    model.mesh = F.MeshFromVertices(vertices)
    # ### Material ###
    damaged_tungsten = F.Material(1, D_0, E_D, borders=[0, sample_thickness])
    model.materials = damaged_tungsten
    # ### Source ###
    # Deuterium Beam Profile (S = flux * f(x))
    distribution = (
        1 / (sigma * (2 * np.pi) ** 0.5) * sp.exp(-0.5 * ((F.x - R_p) / sigma) ** 2)
    )
    ion_flux = sp.Piecewise((flux * distribution, F.t < t_imp), (0, True))
    source_term = F.Source(value=ion_flux, volume=1, field=0)
    model.sources = [source_term]
    # ### Boundary Conditions ###
    model.boundary_conditions = [
        F.DirichletBC(surfaces=[1, 2], value=0, field="solute")
    ]
    # ### Temperature ###

    model.T = F.Temperature(
        value=sp.Piecewise(
            (T_imp, F.t < t_imp),
            (T_rest, F.t < start_tds),
            (min_temp + Beta * (F.t - start_tds), True),
        )
    )
    # ### Trap Settings ###
    k_0 = D_0 / (1.1e-10**2 * 6 * w_atom_density)

    trap_1 = F.Trap(
        E_k=damaged_tungsten.E_D,
        k_0=k_0,
        E_p=1.04,
        p_0=1e13,
        density=2.4e22,
        materials=damaged_tungsten,
    )
    neutron_induced_traps = []
    if densities != []:
        damage_dist = 1 / (1 + sp.exp((F.x - 2.5e-06) / 5e-07))
        for E_p, density in zip(detrapping_energies, densities):
            neutron_induced_traps.append(
                F.Trap(
                    k_0=k_0,
                    E_k=damaged_tungsten.E_D,
                    p_0=1e13,
                    E_p=E_p,
                    density=density * damage_dist,
                    materials=damaged_tungsten,
                )
            )
    model.traps = [trap_1] + neutron_induced_traps

    model.dt = F.Stepsize(
        initial_value=1,
        stepsize_change_ratio=1.1,
        dt_min=1e-1,
        max_stepsize=lambda t: 50 if t > t_imp + t_rest * 0.5 else None,
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
            F.HydrogenFlux(surface=1),
            F.HydrogenFlux(surface=2),
        ],
    )
    derived_quantities += [
        F.TotalVolume(f"{i}", volume=1) for i, _ in enumerate(model.traps, start=1)
    ]
    model.exports = [derived_quantities]
    model.initialise()
    model.run()

    return derived_quantities

dpa_to_quantities = {}
for dpa, densities in dpa_n_i.items():
    dpa_to_quantities[dpa] = festim_sim(densities)
```

## Comparison with experimental data

The results produced by FESTIM are in good agreement with the experimental data. The grey areas represent the contribution of each trap to the global TDS spectrum.

```{code-cell} ipython3
:tags: [hide-input]

def plot_tds(derived_quantities, trap_contributions=False, **kwargs):
    t = np.array(derived_quantities.t)
    flux_left = np.array(derived_quantities.filter(fields="solute", surfaces=1).data)
    flux_right = np.array(derived_quantities.filter(fields="solute", surfaces=2).data)
    flux_total = -flux_left - flux_right
    temp = min_temp + Beta * (t - start_tds)

    idx = np.where(t > start_tds)
    plt.plot(temp[idx], flux_total[idx], **kwargs)

    if trap_contributions:
        colors = [(0.9 * (i % 2), 0.2 * (i % 4), 0.4 * (i % 3)) for i in range(6)]
        trap_data = [derived_quantities.filter(fields=f"{i}").data for i in range(1, 7)]
        contributions = [
            -np.diff(np.array(trap)[idx]) / np.diff(t[idx]) for trap in trap_data
        ]

        for i, cont in enumerate(contributions):
            if i == 0:
                label = "Trap 1"
            else:
                label = f"Trap D{i}"

            plt.plot(temp[idx][1:], cont, linestyle="--", color=colors[i], label=label)
            plt.fill_between(temp[idx][1:], 0, cont, facecolor="grey", alpha=0.1)

from matplotlib import cm, colors

norm = colors.LogNorm(vmin=min(dpa_values[1:]), vmax=max(dpa_values))
colorbar = cm.viridis
sm = plt.cm.ScalarMappable(cmap=colorbar, norm=norm)

for dpa, derived_quantities in dpa_to_quantities.items():
    filename = f"tds_data/{dpa}_dpa.csv"
    experimental_tds = np.genfromtxt(filename, delimiter=",")
    experimental_temp = experimental_tds[:, 0]
    experimental_flux = experimental_tds[:, 1] / sample_area

    if dpa == 0.1:
        plt.figure(1)
        plt.title("Damage = 0.1 dpa")
        plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
        plt.xlabel(r"Temperature (K)")
        plot_tds(
            derived_quantities, linewidth=3, label="FESTIM", trap_contributions=True
        )
        plt.scatter(
            experimental_temp, experimental_flux, color="black", label="experiment", s=16
        )

    plt.figure(2)
    plot_tds(derived_quantities, linestyle="dashed", color="tab:grey", linewidth=2)
    plt.plot(experimental_temp, experimental_flux, color=colorbar(norm(dpa)), linewidth=3)

plt.figure(1)
plt.ylim(bottom=0, top=1e17)
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")
plt.legend()
plt.figure(2)
plt.ylim(bottom=0, top=1e17)
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")

# Plotting color bar
from mpl_toolkits.axes_grid1 import make_axes_locatable

for i in [1, 2]:
    plt.figure(i)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.figure(2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(sm, cax=cax, label="Damage (dpa)")

plt.show()
```

```{note}
The experimental data was taken from {cite}`dark_modelling_2024_code`.
```

+++

