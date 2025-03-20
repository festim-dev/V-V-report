---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: vv-festim-report-env
  language: python
  name: python3
---

# Isothermal absorption of H in Ti

```{tags} 1D, kinetic surface model, transient
```

+++

This validation case reproduces H absorption curves for Ti at different temperatures obtained by Hirooka et al. {cite}`hirooka_1981`. 

Absorption experiments were performed in the vacuum chamber at the base pressure of $1.3 \times 10^{4} \ \mathrm{Pa}$ with $10 \times 13 \times 1 \ \mathrm{mm}^{3}$ cold rolled Ti strips. Absorption curves were acquired at the fixed sample temperature ranging from $450 \ ^{\circ}\mathrm{C}$ to $650 \ ^{\circ}\mathrm{C}$. 

The FESTIM model is based on the work of Shimohata et al. {cite}`shimohata_2021`. By following the approach, evolution of the surface H concentration is assumed to be driven by adsorption from the gas phase and recombination. Only a half of the sample is simulated for simplicity.

The H diffusivity in Ti is taken from the report of Wille and Davis {cite}`wille_1981`. Rates of H desorption, absorption and re-absorption were fitted by Kulagin et al. {cite}`kulagin_2024`.

The FESTIM results are compared to the experimental data from {cite}`shimohata_2021`, extracted with [PlotDigitizer](https://plotdigitizer.com/).

+++

## FESTIM model

+++

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import festim as F
import fenics as f
import numpy as np
import h_transport_materials as htm

################### PARAMETERS ###################
N_A_const = 6.022e23  # Avogadro, mol^-1
e = 1.602e-19  # elementary charge, C
M_H2 = 2.016e-3 / N_A_const  # the H2 mass, kg mol^-1

# Sample size
A = 1e-2 * 1.3e-2  # Ti surface area (1cm x 1.3cm), m^2
L = 1e-3 / 2  # Ti thickness, m
V = A * L  # Ti volume (1cm x 1.3cm x 1mm), m^-3

# Ti properties
X_max = 3  # maximum number of H atoms per a Ti atom in the bulk
n_b = 9.4e4 * N_A_const  #  the number of atomic sites per unit of volume of Ti, m^-3
n_surf = (
    X_max * 2.16e-5 * N_A_const
)  # the number of atomic sites per unit of surface area of Ti, m^-2
N_Ti = n_b * V  #  the number of moles of Ti
n_IS = X_max * n_b
lambda_Ti = n_surf / n_IS

D = htm.diffusivities.filter(material=htm.TITANIUM, author="wille")[0]
D0 = D.pre_exp.magnitude  # diffusivity pre-factor, m^2 s^-1
E_diff = D.act_energy.magnitude  # diffusion activation energy, eV

k_sb = 5.1413505e9  # frequency factor of surface-to-bulk transition, s^-1
E_sb = 1.0087995  # activation energy for surface-to-bulk transition, eV
k_bs = 1.0012013e10  #  # frequency factor of bulk-to-surface transition, s^-1
E_bs = 1.0518870  # activation energy for bulk-to-surface transition, eV
k_des = 3.4115671e-11  # desorption rate, m^2 s^-1
E_des = 5.6197034e-01  # desorption activation energy, eV

# Chamber
V_ch = 2.95e-3  # the chamber volume, m^3
P0 = 1.3e4  # the initial pressure, Pa

T_list = [
    450 + 273,
    500 + 273,
    550 + 273,
    600 + 273,
    650 + 273,
]  # list of considered temperatures, reference data is given in deg C


################### FUNCTIONS ###################
def S0(T):
    # the capturing coefficient
    return 0.0143 * f.exp(F.kJmol_to_eV(1.99) / F.k_B / T)


def P_H2(T, X):
    # partial pressure of hydrogen, Pa
    X0 = 0
    return F.k_B * T * e / V_ch * (P0 * V_ch / (F.k_B * T * e) + (X0 - X) / 2)


def J_vs(T, surf_conc, X):
    J_ads = (
        2
        * S0(T)
        * (1 - surf_conc / n_surf) ** 2
        * P_H2(T, X)
        / (2 * np.pi * M_H2 * F.k_B * T * e) ** 0.5
    )
    J_des = 2 * k_des * surf_conc**2 * f.exp(-E_des / F.k_B / T)
    return J_ads - J_des


def K_sb(T, surf_conc, X):
    return k_sb * f.exp(-E_sb / F.k_B / T)


def K_bs(T, surf_conc, X):
    return k_bs * f.exp(-E_bs / F.k_B / T)


################### CUSTOM MODEL CLASS ###################
class CustomSimulation(F.Simulation):
    def iterate(self):
        super().iterate()

        # Compute Content
        surf = f.assemble(
            self.h_transport_problem.boundary_conditions[0].solutions[0]
            * self.mesh.ds(1)
        )
        X = 2 * (f.assemble(self.mobile.solution * self.mesh.dx) + surf) * A

        # Normalised content parameter
        self.h_transport_problem.boundary_conditions[0].prms["X"].assign(X)


def run_sim(T0):
    Ti_model_impl = CustomSimulation()

    # Mesh
    vertices = np.linspace(0, L, num=1000)

    Ti_model_impl.mesh = F.MeshFromVertices(vertices)

    # Materials
    Ti_model_impl.materials = F.Material(id=1, D_0=D0, E_D=E_diff)

    surf_conc = F.SurfaceKinetics(
        k_bs=K_bs,
        k_sb=K_sb,
        lambda_IS=lambda_Ti,
        n_surf=n_surf,
        n_IS=n_IS,
        J_vs=J_vs,
        surfaces=1,
        initial_condition=0,
        X=0,
    )

    # Boundary conditions
    Ti_model_impl.boundary_conditions = [surf_conc]

    # Homogenous temperature
    Ti_model_impl.T = F.Temperature(value=T0)

    # Exports
    derived_quantities = F.DerivedQuantities(
        [F.AdsorbedHydrogen(surface=1), F.TotalVolume(field="solute", volume=1)],
        show_units=True,
    )

    Ti_model_impl.exports = [derived_quantities]

    Ti_model_impl.dt = F.Stepsize(
        initial_value=1e-3, stepsize_change_ratio=1.1, max_stepsize=5, dt_min=1e-5
    )

    Ti_model_impl.settings = F.Settings(
        absolute_tolerance=1e6,
        relative_tolerance=1e-4,
        maximum_iterations=50,
        final_time=25 * 60,
    )

    Ti_model_impl.initialise()
    Ti_model_impl.run()

    return derived_quantities


results = {}

for T0 in T_list:
    results[T0] = run_sim(T0)
```

## Comparison with experimental data

+++

FESTIM reproduces the experimental data over the whole range of temperatures using one set of parameters.

```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
import plotly.express as px
import plotly

fig = go.Figure()

for i, T in enumerate(T_list):
    color = px.colors.qualitative.Plotly[i]

    FESTIM_data = results[T]
    retention = np.array(FESTIM_data[1].data) + np.array(FESTIM_data[0].data)
    t = np.array(FESTIM_data.t)

    exp = np.loadtxt(f"./content_data/{T-273}.csv", delimiter=",", skiprows=1)

    fig.add_trace(
        go.Scatter(
            x=t / 60,
            y=retention * A / N_Ti,
            mode="lines",
            line=dict(color=color, width=3),
            name=f"FESTIM: T={T} K",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=exp[:, 0],
            y=exp[:, 1],
            mode="markers",
            marker=dict(color=color, size=9),
            name=f"Exp.: T={T} K",
        )
    )


fig.update_yaxes(title_text="Content, H/Ti", range=[0, 0.8], tick0=0, dtick=0.1)
fig.update_xaxes(title_text="Time, min", range=[0, 25], tick0=0, dtick=5)
fig.update_layout(template="simple_white", height=600)
fig.write_html("./hirooka_comparison.html")

from IPython.display import HTML, display
display(HTML("./hirooka_comparison.html"))
```

## Fitted parameters

+++

The table below displays the input parameters which were determined during the optimisation procedure.

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd

pd.set_option("display.float_format", "{:.3e}".format)
prms_list = [k_sb, E_sb, k_bs, E_bs, k_des, E_des]
col_names = [
    "k_sb (s^-1)",
    "E_sb (eV)",
    "k_bs (s^-1)",
    "E_bs (eV)",
    "k_des (m^2 s^-1)",
    "E_des (eV)",
]

data = pd.DataFrame(prms_list).T
data.columns = col_names
data.style.relabel_index(["value"], axis=0).format("{:.2e}".format)

```
