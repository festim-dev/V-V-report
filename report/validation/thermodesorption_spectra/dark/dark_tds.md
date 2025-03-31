---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: vv-festim-report-env-festim-2
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
The trap densities for the neutron induced traps were fitted by {cite}`dark_modelling_2024` for each _dpa_ dose.

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

import ufl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # ### Parameters ###
D_0 = 1.6e-7  # m^2 s^-1
E_D = 0.28  # eV
w_atom_density = 6.3222e28
sample_thickness = 0.8e-3  # m
sample_area = 12e-03 * 15e-03

detrapping_energies = [1.15, 1.35, 1.65, 1.85, 2.05]
dpa_n_i = {
    0: [],
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

    model = F.HydrogenTransportProblem()
    vertices = np.concatenate(
        [
            np.linspace(0, 3e-9, num=100),
            np.linspace(3e-9, 8e-6, num=100),
            np.linspace(8e-6, 8e-5, num=100),
            np.linspace(8e-5, sample_thickness, num=100),
        ]
    )
    model.mesh = F.Mesh1D(vertices)
    # ### Material ###
    damaged_tungsten = F.Material(D_0, E_D)

    volume = F.VolumeSubdomain1D(id=1, borders=[0, sample_thickness], material=damaged_tungsten)
    left_boundary = F.SurfaceSubdomain1D(id=1, x=0)
    right_boundary = F.SurfaceSubdomain1D(id=2, x=sample_thickness)
    model.subdomains = [volume, left_boundary, right_boundary]


    H = F.Species("H")
    trapped_species = [F.Species(f"trapped_{i+1}", mobile=False) for i in range(len(densities) + 1)]

    damage_dist = lambda x: 1 / (1 + ufl.exp((x[0] - 2.5e-06) / 5e-07))
    empty_traps = [F.ImplicitSpecies(n=2.4e22, others=[trapped_species[0]])] + [
        F.ImplicitSpecies(
            n=lambda x: densities[i] * damage_dist(x),
            others=[spe],
        )
        for i, spe in enumerate(trapped_species[1:])
    ]

    assert len(empty_traps) == len(trapped_species)

    model.species = [H] + trapped_species

    # # ### Source ###
    # # Deuterium Beam Profile (S = flux * f(x))
    # distribution = (
    #     1 / (sigma * (2 * np.pi) ** 0.5) * sp.exp(-0.5 * ((F.x - R_p) / sigma) ** 2)
    # )
    # ion_flux = sp.Piecewise((flux * distribution, F.t < t_imp), (0, True))
    # source_term = F.Source(value=ion_flux, volume=1, field=0)
    # model.sources = [source_term]
    # ### Boundary Conditions ###
    model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left_boundary, value=0, species=H),
        F.FixedConcentrationBC(subdomain=right_boundary, value=0, species=H),
    ]
    # ### Temperature ###

    def temp_fun(t):
        if t < t_imp:
            return T_imp
        elif t < start_tds:
            return T_rest + (T_imp - T_rest) * (t - t_imp) / (start_tds - t_imp)
        else:
            return min_temp + Beta * (t - start_tds)
    model.temperature = temp_fun

    # ### Trap Settings ###
    k_0 = D_0 / (1.1e-10**2 * 6 * w_atom_density)

    trapping_reaction_1 = F.Reaction(
        reactant=[H, empty_traps[0]],
        product=[trapped_species[0]],
        k_0=k_0,
        E_k=damaged_tungsten.E_D,
        p_0=1e13,
        E_p=1.04,
        volume=volume,
    )

    model.reactions = [trapping_reaction_1]

    for i, _ in enumerate(densities):
        model.reactions.append(
            F.Reaction(
                reactant=[H, empty_traps[i+1]],
                product=[trapped_species[i + 1]],
                k_0=k_0,
                E_k=damaged_tungsten.E_D,
                p_0=1e13,
                E_p=detrapping_energies[i],
                volume=volume,
            )
        )
    

    
    model.settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        final_time=start_tds + (max_temp - min_temp) / Beta,  # time to reach max temp
    )
    model.settings.stepsize = F.Stepsize(
        initial_value=1,
        growth_factor=1.1,
        cutback_factor=0.9,
        target_nb_iterations=4,
        max_stepsize=lambda t: 50 if t > t_imp + t_rest * 0.5 else None,
    )
    derived_quantities = [F.TotalVolume(field=spe, volume=volume) for spe in model.species]
            # F.HydrogenFlux(surface=1),
            # F.HydrogenFlux(surface=2),
        
    
    model.exports = derived_quantities
    model.initialise()
    model.run()

    return derived_quantities
```

```{code-cell} ipython3
dpa_to_quantities = {}
for dpa, densities in dpa_n_i.items():
    dpa_to_quantities[dpa] = festim_sim(densities)
```

## Comparison with experimental data

The results produced by FESTIM are in good agreement with the experimental data. The grey areas represent the contribution of each trap to the global TDS spectrum.

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import cm, colors
norm = colors.LogNorm(vmin=min(list(dpa_n_i.keys())[1:]), vmax=max(dpa_n_i.keys())) #using [1:] indexing to ignore 0
colorbar = cm.viridis
sm = plt.cm.ScalarMappable(cmap=colorbar, norm=norm)

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
    plt.plot(
        experimental_temp,
        experimental_flux,
        color=colorbar(norm(dpa)) if dpa != 0 else "black",
        linewidth=3,
    )
    if dpa == 0:
        max_curve_y = np.max(experimental_flux)
        max_curve_x = experimental_temp[experimental_flux == max_curve_y][0]
        plt.annotate(
            "undamaged",
            xy=(max_curve_x, max_curve_y),  # Point to annotate
            xytext=(300, 0.4e17),  # Location of text
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", facecolor='black'),  # Arrow properties
        )

for i in [1, 2]:
    plt.figure(i)
    plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
    plt.xlabel(r"Temperature (K)")
    plt.ylim(bottom=0, top=1e17)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.figure(1)
plt.legend()

# Plotting color bar
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.figure(2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(sm, cax=cax, label="Damage (dpa)")

plt.show()
```

```{note}
The experimental data was taken from {cite}`dark_modelling_2024_code`.
```

## Trap Parameters

### Damage-induced trap parameters

This table displays the neutron-induced traps' detrapping energy $E_p$ and their density per dpa dose in $m^{-3}$.

```{code-cell} ipython3
:tags: [hide-input]

dpa_no_zero = dpa_n_i | {}
dpa_no_zero.pop(0)
data = {"E_p (eV)" : detrapping_energies} | dpa_no_zero
dpa_frame = pd.DataFrame(data)

dpa_frame.columns = dpa_frame.columns.map(lambda s: f"{s:.1e} dpa" if not isinstance(s, str) else s)
dpa_frame.style \
    .relabel_index([f"Trap D{i}" for i in range(1, 6)], axis=0) \
    .format("{:.2e}".format)
```
