---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: vv-festim-report-env-festim-2
    language: python
    name: python3
---

<!-- #region -->
# Co-permeation of H and D through Pd

```{tags} 1D, transient, multi-isotopes
```


This case is taken and adapted from {cite}`ambrosek_verification_2008` based on the experimental data of {cite}`kizu2001co`.


<!-- #endregion -->

## Calibration with pure D2

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

# import dolfinx.log
# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
for pd_thickness, temperature in prms:
    print(f"Pd thickness: {pd_thickness:.2e} m")
    print(f"Temperature: {temperature:.2f} K")
    dd_desorption_fluxes = []
    models = []

    # ------ Model ------ #
    for d2_pressure in upstream_d_pressures:
        # create the model
        my_model = make_festim_model_dlr(
            pd_thickness=pd_thickness,
            temperature=temperature,
            upstream_d2_pressure=d2_pressure,
        )

        # initialise the model
        my_model.initialise()

        # run the model
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

```python
import pandas as pd
import plotly.graph_objects as go
from pypalettes import load_cmap

cmap = load_cmap("Acadia")

data_thick_825 = pd.read_csv(
    "https://raw.githubusercontent.com/idaholab/TMAP8/dc0bfc4cb3114a2c3159f5f18a0d441c4ce78b13/test/tests/val-2e/gold/experiment_thick_825K.csv"
)
data_thin_825 = pd.read_csv(
    "https://raw.githubusercontent.com/idaholab/TMAP8/dc0bfc4cb3114a2c3159f5f18a0d441c4ce78b13/test/tests/val-2e/gold/experiment_thin_825K.csv"
)
data_thin_865 = pd.read_csv(
    "https://raw.githubusercontent.com/idaholab/TMAP8/dc0bfc4cb3114a2c3159f5f18a0d441c4ce78b13/test/tests/val-2e/gold/experiment_thin_865K.csv"
)

exp_data = [
    data_thin_825,
    data_thick_825,
    data_thin_865,
]

fig = go.Figure()

for i, result_dict in enumerate(results):
    # Add line plot for simulation results
    fig.add_trace(
        go.Scatter(
            x=result_dict["upstream_d_pressures"],
            y=result_dict["dd_desorption_fluxes"],
            mode="lines",
            name=f"{result_dict['thickness']:.2e} m, {result_dict['temperature']} K (model)",
            line=dict(color=cmap.hex[i][:-2]),
        )
    )

    # Add scatter plot for experimental data
    fig.add_trace(
        go.Scatter(
            x=exp_data[i]["Pressure [Pa]"],
            y=exp_data[i]["Flux [mol/m^2/s]"],
            mode="markers",
            name=f"{result_dict['thickness']:.2e} m, {result_dict['temperature']} K (exp)",
            marker=dict(color=cmap.hex[i][:-2], symbol="x"),
        )
    )

# Update layout for log scale and labels
fig.update_layout(
    xaxis=dict(
        title="Upstream D pressure (Pa)",
        type="log",
        exponentformat="power",
        showexponent="last",
    ),
    yaxis=dict(
        title="Desorption flux (mol/m^2/s)",
        type="log",
        range=[-8, -3],  # Corresponds to 1e-8 to 1e-3
        exponentformat="power",
        showexponent="last",
    ),
    legend=dict(title="Legend"),
    template="plotly_white",
    width=1000,  # Set the width of the figure
    height=600,  # Set the height of the figure
)

fig.write_html("./co_permeation.html")
from IPython.display import HTML, display

display(HTML("./co_permeation.html"))
```

## Co-permeation of H and D

### Implementation

```python
import festim as F
import dolfinx.fem as fem


class FluxFromSurfaceReaction(F.SurfaceFlux):
    def __init__(self, reaction: F.SurfaceReactionBC):
        super().__init__(
            F.Species(),  # just a dummy species here
            reaction.subdomain,
        )
        self.reaction = reaction.flux_bcs[0]

    def compute(self, ds):
        self.value = fem.assemble_scalar(
            fem.form(self.reaction.value_fenics * ds(self.surface.id))
        )
        self.data.append(self.value)
```

```python
pd_thickness = 0.025e-3  # m
temperature = 870  # K
upstream_effective_H_pressure = 0.063  # Pa
```

```python
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
```

```python
cmap = load_cmap("Acadia")

# Create a Plotly figure
fig = go.Figure()

# Add scatter plots for experimental data
fig.add_trace(
    go.Scatter(
        x=exp_data["H2_X"],
        y=exp_data["H2_Y"],
        mode="markers",
        name="H2 (exp)",
        marker=dict(color=cmap.hex[0][:-2], symbol="circle"),
    )
)
fig.add_trace(
    go.Scatter(
        x=exp_data["D2_X"],
        y=exp_data["D2_Y"],
        mode="markers",
        name="D2 (exp)",
        marker=dict(color=cmap.hex[1][:-2], symbol="triangle-up"),
    )
)
fig.add_trace(
    go.Scatter(
        x=exp_data["HD_X"],
        y=exp_data["HD_Y"],
        mode="markers",
        name="HD (exp)",
        marker=dict(color=cmap.hex[2][:-2], symbol="square"),
    )
)

# Add line plots for FESTIM simulation results
fig.add_trace(
    go.Scatter(
        x=upstream_d_pressures,
        y=hh_desorption_fluxes,
        mode="lines",
        name="HH (FESTIM)",
        line=dict(color=cmap.hex[0][:-2]),
    )
)
fig.add_trace(
    go.Scatter(
        x=upstream_d_pressures,
        y=dd_desorption_fluxes,
        mode="lines",
        name="DD (FESTIM)",
        line=dict(color=cmap.hex[1][:-2]),
    )
)
fig.add_trace(
    go.Scatter(
        x=upstream_d_pressures,
        y=hd_desorption_fluxes,
        mode="lines",
        name="HD (FESTIM)",
        line=dict(color=cmap.hex[2][:-2]),
    )
)

# annotate the RMSE
for RMSE_value, y in zip(errors.values(), [5e-6, 5e-5, 4e-4]):
    fig.add_annotation(
        x=np.log10(upstream_d_pressures[-1] * 3),
        y=np.log10(y),
        text=f"log-RMSE = {np.abs(RMSE_value):.2%}",
        showarrow=False,
        # font=dict(size=12),
    )

# Update layout for log scale, labels, and legend
fig.update_layout(
    xaxis=dict(title="Upstream D pressure (Pa)", type="log"),
    yaxis=dict(
        title="Desorption flux (mol/m^2/s)",
        type="log",
        range=[-8, -3],  # Corresponds to 1e-8 to 1e-3
        exponentformat="power",
        showexponent="last",
    ),
    legend=dict(title="Legend"),
    template="plotly_white",
    width=800,  # Set the width of the figure
    height=600,  # Set the height of the figure
)


fig.write_html("./co_permeation2.html")
from IPython.display import HTML, display

display(HTML("./co_permeation2.html"))
```
