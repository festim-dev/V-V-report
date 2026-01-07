---
jupyter:
  jupytext:
    default_lexer: ipython3
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: vv-festim-report-env
    language: python
    name: python3
---

<!-- #region -->
# Co-permeation of H and D through Pd

```{tags} 1D, transient, multi-isotopes
```


This case is taken and adapted from {cite}`ambrosek_verification_2008` based on the experimental data reporeted in {cite}`kizu2001co`.
The system is first simulated with pure D<sub>2</sub> permeation, then extended to the co-permeation regime, in which both D<sub>2</sub> and H<sub>2</sub> dissociate on the Pd upstream surface, diffuse through the membrane, and recombine on the downstream surface to form H<sub>2</sub>, D<sub>2</sub> and HD.
<!-- #endregion -->

## Permeation of pure D<sub>2</sub>
Below we present the implementation of pure D<sub>2</sub> permeation through a Pd membrane. Two membrane thicknesses are considered (0.025 mm and 0.05 mm), and simulations are performed at temperatures of 825 K and 865 K. The corresponding experimental setup and model parameters are described in {cite}`ambrosek_verification_2008`.

### Implementation

```python
import festim as F

import numpy as np


def make_festim_model_dlr(pd_thickness, temperature, upstream_d2_pressure):
    my_model = F.HydrogenTransportProblem()

    D = F.Species("D")
    my_model.species = [D]

    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, pd_thickness, 100))
    my_mat = F.Material(
        name="Pd",
        D_0=2.636e-4,
        E_D=1315.8 * F.k_B,
    )
    vol = F.VolumeSubdomain1D(id=1, borders=[0, pd_thickness], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=pd_thickness)

    my_model.subdomains = [vol, left, right]

    my_model.temperature = temperature

    n = 0.9297
    K_S_0 = 1.511e23
    K_S_E = 5918 * F.k_B
    surface_reaction_dd_left = F.FixedConcentrationBC(
        subdomain=left,
        species=D,
        value=K_S_0 * np.exp(-K_S_E / F.k_B / temperature) * upstream_d2_pressure**n,
    )

    my_model.boundary_conditions = [
        surface_reaction_dd_left,
        F.FixedConcentrationBC(
            species=D,
            value=0.0,
            subdomain=right,
        ),
    ]

    D_flux_right = F.SurfaceFlux(D, right)
    D_flux_left = F.SurfaceFlux(D, left)

    my_model.exports = [D_flux_left, D_flux_right]

    my_model.settings = F.Settings(atol=1e8, rtol=1e-10, transient=False)

    return my_model
```

```python
upstream_d_pressures = np.geomspace(1e-4, 3, num=6)

thicknesses = [0.025e-3, 0.05e-3]

prms = [
    # Pd thickness, temperature
    (0.025e-3, 825),
    (0.05e-3, 825),
    (0.025e-3, 865),
]
results = []


for pd_thickness, temperature in prms:
    print(f"Pd thickness: {pd_thickness:.2e} m")
    print(f"Temperature: {temperature:.2f} K")
    dd_desorption_fluxes = []
    models = []

    for d2_pressure in upstream_d_pressures:
        my_model = make_festim_model_dlr(
            pd_thickness=pd_thickness,
            temperature=temperature,
            upstream_d2_pressure=d2_pressure,
        )

        my_model.initialise()

        my_model.run()

        # ------ Post processsing ------ #
        models.append(my_model)

        # convert all data to mol
        for export in my_model.exports:
            avogadro = 6.022e23
            # convert from particles to moles
            export.data = np.array(export.data) / avogadro

            # convert from atoms to molecules
            export.data *= 1 / 2

        DD_flux = my_model.exports[1]

        dd_desorption_fluxes.append(np.abs(DD_flux.data)[-1])

    results.append(
        {
            "thickness": pd_thickness,
            "temperature": temperature,
            "upstream_d_pressures": upstream_d_pressures,
            "dd_desorption_fluxes": dd_desorption_fluxes,
            "models": models,
        }
    )
```

### Results

Below is the evolution of the downstream D<sub>2</sub> flux as a function of upstream D<sub>2</sub> pressure.
There is a good agreement between FESTIM and the experimental data. The change of the slope in the experimentally measured flux at higher pressures suggest a transition to the diffusion-limited regime.

```python tags=["hide-input"]
import pandas as pd
import matplotlib.pyplot as plt
from pypalettes import load_cmap

cmap = load_cmap("Acadia")

# Load experimental data from GitHub
commit_hash = "dc0bfc4cb3114a2c3159f5f18a0d441c4ce78b13"

url_base = f"https://raw.githubusercontent.com/idaholab/TMAP8/{commit_hash}/test/tests/val-2e/gold/"
data_thin_825 = pd.read_csv(url_base + "experiment_thin_825K.csv")
data_thick_825 = pd.read_csv(url_base + "experiment_thick_825K.csv")
data_thin_865 = pd.read_csv(url_base + "experiment_thin_865K.csv")

exp_data = [
    data_thin_825,
    data_thick_825,
    data_thin_865,
]

fig, ax = plt.subplots(figsize=(10, 6))

for i, result_dict in enumerate(results):
    # Add line plot for simulation results
    ax.plot(
        result_dict["upstream_d_pressures"],
        result_dict["dd_desorption_fluxes"],
        color=cmap.hex[i][:-2],
        linewidth=3,
        label=f"{result_dict['thickness']:.2e} m, {result_dict['temperature']} K (model)"
    )

    # Add scatter plot for experimental data
    ax.scatter(
        exp_data[i]["Pressure [Pa]"],
        exp_data[i]["Flux [mol/m^2/s]"],
        color=cmap.hex[i][:-2],
        marker="x",
        s=64,
        label=f"{result_dict['thickness']:.2e} m, {result_dict['temperature']} K (exp)"
    )


# Log scales
ax.set_xscale("log")
ax.set_yscale("log")

# Labels
ax.set_xlabel("Upstream D pressure (Pa)")
ax.set_ylabel(r"Desorption flux (mol m$^{-2}$ s$^{-1}$)")

# Match your y-range: 1e-8 to 1e-3
ax.set_ylim(1e-8, 1e-3)

# Legend + layout
ax.legend(title="Legend", fontsize=9)
fig.tight_layout()

plt.show()
```

## Co-permeation of H and D

The co-permeation of H and D through a Pd membrane is simulated using FESTIM within a one-dimensional transient framework. A Pd membrane with thicknesses of 0.025 mm and 0.05 mm is discretized using a 1D mesh, with H and D treated as distinct diffusing species within the same solid phase. Temperature-dependent bulk diffusion coefficients for each isotope are prescribed using Arrhenius relations. Surface reactions at both the upstream and downstream boundaries are explicitly modeled via surface reaction boundary conditions. At each surface, adsorption and desorption reactions are defined for the H–H, D–D, and mixed H–D, enabling the formation of H<sub>2</sub>, D<sub>2</sub> and HD molecules. The corresponding kinetic parameters and material properties are taken from the TMAP7 verification and validation report {cite}`ambrosek_verification_2008` with system-level effects such as enclosures, pumping, etc. not included.

Upstream boundary conditions are imposed through effective H<sub>2</sub>, D<sub>2</sub> gas pressures, while the downstream boundary is maintained at zero gas pressure. Surface-integrated fluxes of H, D, H<sub>2</sub>, HD, and D<sub>2</sub> are evaluated using custom surface flux exports. Fluxes of H<sub>2</sub>, HD, and D<sub>2</sub> are subsequently post-processed and converted to molar units to enable direct comparison with experimental co-permeation data at temperatures of 825 K and 865 K.

### Implementation

```python tags=["hide-cell"]
import festim as F
import dolfinx.fem as fem


class FluxFromSurfaceReaction(F.SurfaceFlux):
    def __init__(self, reaction: F.SurfaceReactionBC):
        super().__init__(
            F.Species(),  # just a dummy species here
            reaction.subdomain,
        )
        self.reaction = reaction.flux_bcs[0]

    def compute(self, u, ds, entity_maps=None):
        # u is provided by FESTIM but may be unused here, keep for API compatibility
        self.value = fem.assemble_scalar(
            fem.form(self.reaction.value_fenics * ds(self.surface.id))
        )
        self.data.append(self.value)
```

```python
pd_thickness = 0.025e-3  # m
temperature = 870  # K
upstream_effective_H_pressure = 0.063  # Pa

my_model = F.HydrogenTransportProblem()

H = F.Species("H")
D = F.Species("D")
my_model.species = [H, D]

my_model.mesh = F.Mesh1D(vertices=np.linspace(0, pd_thickness, 100))
my_mat = F.Material(
    name="Pd",
    D_0={
        H: 3.728e-4,
        D: 2.636e-4,
    },
    E_D={
        H: 1315.8 * F.k_B,
        D: 1315.8 * F.k_B,
    },
)
vol = F.VolumeSubdomain1D(id=1, borders=[0, pd_thickness], material=my_mat)
left = F.SurfaceSubdomain1D(id=1, x=0)
right = F.SurfaceSubdomain1D(id=2, x=pd_thickness)

my_model.subdomains = [vol, left, right]


my_model.temperature = temperature

E_kr = 11836 * F.k_B

surface_reaction_hd_left = F.SurfaceReactionBC(
    reactant=[H, D],
    gas_pressure=0,  # free parameter
    k_r0=2.502e-24 / (3 * temperature) ** 0.5,
    E_kr=E_kr,
    k_d0=2.1897e22 / (3 * temperature) ** 0.5,
    E_kd=0,
    subdomain=left,
)

surface_reaction_hh_left = F.SurfaceReactionBC(
    reactant=[H, H],
    gas_pressure=0,  # free parameter
    k_r0=2.502e-24 / (2 * temperature) ** 0.5,
    E_kr=E_kr,
    k_d0=2.1897e22 / (2 * temperature) ** 0.5,
    E_kd=0,
    subdomain=left,
)

surface_reaction_dd_left = F.SurfaceReactionBC(
    reactant=[D, D],
    gas_pressure=0,  # free parameter
    k_r0=2.502e-24 / (4 * temperature) ** 0.5,
    E_kr=E_kr,
    k_d0=2.1897e22 / (4 * temperature) ** 0.5,
    E_kd=0,
    subdomain=left,
)

surface_reaction_hd_right = F.SurfaceReactionBC(
    reactant=[H, D],
    gas_pressure=0,
    k_r0=2.502e-24 / (3 * temperature) ** 0.5,
    E_kr=E_kr,
    k_d0=2.1897e22 / (3 * temperature) ** 0.5,
    E_kd=0,
    subdomain=right,
)

surface_reaction_hh_right = F.SurfaceReactionBC(
    reactant=[H, H],
    gas_pressure=0,
    k_r0=2.502e-24 / (2 * temperature) ** 0.5,
    E_kr=E_kr,
    k_d0=2.1897e22 / (2 * temperature) ** 0.5,
    E_kd=0,
    subdomain=right,
)

surface_reaction_dd_right = F.SurfaceReactionBC(
    reactant=[D, D],
    gas_pressure=0,
    k_r0=2.502e-24 / (4 * temperature) ** 0.5,
    E_kr=E_kr,
    k_d0=2.1897e22 / (4 * temperature) ** 0.5,
    E_kd=0,
    subdomain=right,
)

my_model.boundary_conditions = [
    surface_reaction_hd_left,
    surface_reaction_hh_left,
    surface_reaction_dd_left,
    surface_reaction_hd_right,
    surface_reaction_hh_right,
    surface_reaction_dd_right,
]

H_flux_right = F.SurfaceFlux(H, right)
H_flux_left = F.SurfaceFlux(H, left)
D_flux_right = F.SurfaceFlux(D, right)
D_flux_left = F.SurfaceFlux(D, left)
HD_flux = FluxFromSurfaceReaction(surface_reaction_hd_right)
HH_flux = FluxFromSurfaceReaction(surface_reaction_hh_right)
DD_flux = FluxFromSurfaceReaction(surface_reaction_dd_right)


# needed to compute D_global even if not used
for flux in [HH_flux, HD_flux, DD_flux]:
    flux.field = H


my_model.exports = [
    H_flux_left,
    H_flux_right,
    D_flux_left,
    D_flux_right,
    HD_flux,
    HH_flux,
    DD_flux,
]


my_model.settings = F.Settings(atol=1e11, rtol=1e-6, final_time=10, transient=True)

my_model.settings.stepsize = 0.2
```

```python
all_d_desorption_fluxes = []
hh_desorption_fluxes = []
hd_desorption_fluxes = []
dd_desorption_fluxes = []
upstream_d_pressures = np.geomspace(4e-3, 1, num=5)

for effective_d_pressure in upstream_d_pressures:
    upstream_d2_pressure = effective_d_pressure
    upstream_h2_pressure = upstream_effective_H_pressure

    print(f"Upstream D2 pressure: {upstream_d2_pressure:.2e} Pa")
    print(f"Upstream H2 pressure: {upstream_h2_pressure:.2e} Pa")

    for flux_bc in surface_reaction_dd_left.flux_bcs:
        flux_bc.gas_pressure = effective_d_pressure

    for flux_bc in surface_reaction_hh_left.flux_bcs:
        flux_bc.gas_pressure = upstream_h2_pressure

    my_model.initialise()
    my_model.run()

    # convert all data to mol
    for export in my_model.exports:
        avogadro = 6.022e23
        export.data = np.array(export.data) / avogadro

    all_d_desorption_fluxes.append(np.abs(D_flux_right.data)[-1])
    print(
        f"Desorption flux at {effective_d_pressure:.2e} Pa: {all_d_desorption_fluxes[-1]:.2e} mol/m^2/s"
    )

    hh_desorption_fluxes.append(np.abs(HH_flux.data)[-1])
    hd_desorption_fluxes.append(np.abs(HD_flux.data)[-1])
    dd_desorption_fluxes.append(np.abs(DD_flux.data)[-1])
```

### Results
Below is the evolution of H<sub>2</sub>, HD, and D<sub>2</sub> fluxes as a function of the effective deuterium upstream pressure.

There is a reasonable agreement between the experimental data and the FESTIM simulation. 

A better agreement could potentially be obtained by setting the surface rates and diffusivities as free parameters and then perform some parametric optimisation.

Better experimental data with better measurements of the upstream partial pressures would be required to better constrain the model.

```python tags=["hide-cell"]
from scipy.interpolate import interp1d

# read experimental data
exp_data = pd.read_csv(
    "co_permeation_exp_data.csv",
    names=["H2_X", "H2_Y", "D2_X", "D2_Y", "HD_X", "HD_Y"],
    skiprows=2,
)


def RMSE(exp, sim):
    """
    Calculate the Root Mean Square Error (RMSE) between experimental and simulated data.

    Parameters:
    exp (array-like): Experimental data.
    sim (array-like): Simulated data.

    Returns:
    float: RMSE value.
    """
    return np.sqrt(np.mean((np.array(exp) - np.array(sim)) ** 2)) / np.mean(
        np.array(exp)
    )


errors = {}

for label, flux in zip(
    ["H2", "D2", "HD"],
    [hh_desorption_fluxes, dd_desorption_fluxes, hd_desorption_fluxes],
):
    fluxes_exp = exp_data[f"{label}_Y"]
    pressures_exp = exp_data[f"{label}_X"]

    sim_interp = interp1d(
        upstream_d_pressures,
        flux,
        kind="linear",
        fill_value="extrapolate",
    )

    RMSE_value = RMSE(np.log10(fluxes_exp), np.log10(sim_interp(pressures_exp)))

    errors[label] = RMSE_value


from numpy import size


cmap = load_cmap("Acadia")

# Create a matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

c0 = cmap.hex[0][:-2]
c1 = cmap.hex[1][:-2]
c2 = cmap.hex[2][:-2]

# Experimental data (markers)
ax.plot(
    exp_data["H2_X"], exp_data["H2_Y"],
    linestyle="None", marker="o", markersize=8,
    label="H2 (exp)", color=c0
)
ax.plot(
    exp_data["D2_X"], exp_data["D2_Y"],
    linestyle="None", marker="^", markersize=8,
    label="D2 (exp)", color=c1
)
ax.plot(
    exp_data["HD_X"], exp_data["HD_Y"],
    linestyle="None", marker="s", markersize=8,
    label="HD (exp)", color=c2
)

# FESTIM simulation results (lines)
ax.plot(
    upstream_d_pressures, hh_desorption_fluxes,
    linewidth=3, label="HH (FESTIM)", color=c0
)
ax.plot(
    upstream_d_pressures, dd_desorption_fluxes,
    linewidth=3, label="DD (FESTIM)", color=c1
)
ax.plot(
    upstream_d_pressures, hd_desorption_fluxes,
    linewidth=3, label="HD (FESTIM)", color=c2
)

# RMSE annotations
# Place text to the right of the last x point, at chosen y levels (data coords)
x_text = upstream_d_pressures[-1]*1.1
y_positions = [5e-6, 5e-5, 4e-4]

for (label, rmse_val), y in zip(errors.items(), y_positions):
    ax.text(
        x_text, y,
        f"log-RMSE = {abs(rmse_val):.2%}",
        ha="left", va="center",
    )


# Axes formatting (log-log)
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Upstream D pressure (Pa)")
ax.set_ylabel(r"Desorption flux (mol m$^{-2}$ s$^{-1}$)")

ax.set_ylim(1e-8, 1e-3)
ax.set_xlim(2e-3, 5)

ax.legend()
plt.tight_layout()
plt.show()

```
