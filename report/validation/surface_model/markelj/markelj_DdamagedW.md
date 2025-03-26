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

#  Deuterium atom exposure of self-damaged tungsten

```{tags} 1D, kinetic surface model, NRA, transient
```

+++

This validation case is NRA measurements of D in self-damaged W performed by Markelj et al. {cite}`markelj_2016`.

The experimental procedure included three main phases. A high-purity polycrystalline W sample of 0.8 mm thickness was pre-damaged with 20 MeV W ions. The pre-damaged sample was then exposed to low-energy (~0.3 eV) D atomic flux of $5.8\times 10^{18}\,\textrm{m}^{-2}\textrm{s}^{-1}$ at 600 K. The exposure continued until the D fluence of $1\times 10^{24}\,\mathrm{m}^{-2}$ was reached. Finally, an isothermal desorption of D for 43 h. at 600 K was conducted. 

The FESTIM model is based on the approach of Hodille et al. {cite}`hodille_2017`. Only isothermal D exposure and desorption phases are simulated omitting intermediate cooling/re-heating steps. For the surface processes, adsorption of low-energy atoms, desorption of molecules (Langmuir-Hinshelwood recombination), and recombination of an adsorbed atom with an incident atom (Eley-Rideal recombination) are considered.

The D diffusivity in W is defined by scaling the corresponding value for H (Fernandez et al. {cite}`fernandez_2015`) by a factor of $1/\sqrt{2}$. Five types of trapping sites are included to reproduce the experimental data: two intrinsic traps and three extrinsic traps with sigmoidal distribution ($f$) within the damaged layer:

$$
f(x)=\dfrac{1}{1+\exp\left(\dfrac{x-x_0}{\Delta x}\right)},
$$
where $x_0=2.2\,\mu\textrm{m}$, $\Delta x=0.154\,\mu\textrm{m}$.

The FESTIM results are compared to the experimental data and the results of MHIMS simulation, both taken from {cite}`hodille_2017`.

+++

## FESTIM model

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import festim as F
import fenics as f
import numpy as np
import sympy as sp
import h_transport_materials as htm

################### PARAMETERS ###################
# Exposure conditions
Gamma_atom = 5.8e18
T_exposure = 600
t_exposure = 1e24 / Gamma_atom
t_des = 52 * 3600
final_time = t_exposure + t_des

# Sample
L = 0.8e-3  # half thickness, m

# W properties
rho_W = 6.3e28  # W atomic concentration, m^-3
n_IS = 6 * rho_W  # concentration of interstitial sites, m^-3
n_surf = 6.9 * rho_W ** (2 / 3)  # concentration of adsorption sites, m^-2
nu0 = 1e13  # attempt frequency, s^-1
SP = 0.19

D_H = htm.diffusivities.filter(material=htm.Tungsten, author="fernandez")[0]
D0 = D_H.pre_exp.magnitude / np.sqrt(2)  # diffusivity pre-factor, m^2 s^-1
E_diff = D_H.act_energy.magnitude  # diffusion activation energy, eV

lambda_IS = 110e-12  # distance between 2 IS sites, m
sigma_exc = 1.7e-21  # Cross-section for the direct abstraction, m^2
lambda_des = 1 / np.sqrt(n_surf)

# Transitions
E_bs = E_diff  # energy barrier from bulk to surface, eV
E_sb = 1.545
E_des = 0.87


################### FUNCTIONS ###############
def k_sb(T, surf_conc, t):
    return nu0 * f.exp(-E_sb / F.k_B / T)


def k_bs(T, surf_conc, t):
    return nu0 * f.exp(-E_bs / F.k_B / T)


def J_vs_left(T, surf_conc, t):
    G_atom = Gamma_atom * f.conditional(t <= t_exposure, 1, 0)

    phi_atom = SP * G_atom * (1 - surf_conc / n_surf)

    phi_exc = G_atom * sigma_exc * surf_conc

    phi_des = 2 * nu0 * (lambda_des * surf_conc) ** 2 * f.exp(-2 * E_des / F.k_B / T)
    return phi_atom - phi_exc - phi_des


def J_vs_right(T, surf_conc, t):
    phi_des = 2 * nu0 * (lambda_des * surf_conc) ** 2 * f.exp(-2 * E_des / F.k_B / T)
    return -phi_des


################### MODEL ###################

W_model = F.Simulation(log_level=40)

# Mesh
vertices = np.concatenate(
    [
        np.linspace(0, 5e-8, num=100),
        np.linspace(5e-8, 5e-6, num=400),
        np.linspace(5e-6, L, num=500),
    ]
)

W_model.mesh = F.MeshFromVertices(vertices)

# Materials
tungsten = F.Material(id=1, D_0=D0, E_D=E_diff)
W_model.materials = tungsten

distr = 1 / (1 + sp.exp((F.x - 2.2e-6) / 1.54e-7))

traps = F.Traps(
    [
        F.Trap(
            k_0=D0 / (n_IS * lambda_IS**2),
            E_k=E_diff,
            p_0=nu0,
            E_p=0.85,
            density=1e-4 * rho_W,
            materials=tungsten,
        ),
        F.Trap(
            k_0=D0 / (n_IS * lambda_IS**2),
            E_k=E_diff,
            p_0=nu0,
            E_p=1.00,
            density=1e-4 * rho_W,
            materials=tungsten,
        ),
        F.Trap(
            k_0=D0 / (n_IS * lambda_IS**2),
            E_k=E_diff,
            p_0=nu0,
            E_p=1.65,
            density=0.19e-2 * rho_W * distr,
            materials=tungsten,
        ),
        F.Trap(
            k_0=D0 / (n_IS * lambda_IS**2),
            E_k=E_diff,
            p_0=nu0,
            E_p=1.85,
            density=0.16e-2 * rho_W * distr,
            materials=tungsten,
        ),
        F.Trap(
            k_0=D0 / (n_IS * lambda_IS**2),
            E_k=E_diff,
            p_0=nu0,
            E_p=2.06,
            density=0.02e-2 * rho_W * distr,
            materials=tungsten,
        ),
    ]
)
W_model.traps = traps

W_model.T = T_exposure

BC_left = F.SurfaceKinetics(
    k_sb=k_sb,
    k_bs=k_bs,
    lambda_IS=lambda_IS,
    n_surf=n_surf,
    n_IS=n_IS,
    J_vs=J_vs_left,
    surfaces=1,
    initial_condition=0,
    t=F.t,
)

BC_right = F.SurfaceKinetics(
    k_sb=k_sb,
    k_bs=k_bs,
    lambda_IS=lambda_IS,
    n_surf=n_surf,
    n_IS=n_IS,
    J_vs=J_vs_right,
    surfaces=2,
    initial_condition=0,
    t=F.t,
)

W_model.boundary_conditions = [BC_left, BC_right]

# Exports
export_fluences = [
    5.22e22,
    1.25e23,
    4.8e23,
    6.3e23,
    1e24,
]

export_times = [fluence / Gamma_atom for fluence in export_fluences]

export_times += [export_times[-1] + 20 * 3600, export_times[-1] + 52 * 3600]

derived_quantities = F.DerivedQuantities(
    [
        F.TotalVolume(field="retention", volume=1),
        F.AdsorbedHydrogen(surface=1),
        F.AdsorbedHydrogen(surface=2),
    ],
    show_units=True,
)

TXT = F.TXTExport(field="retention", filename="./FESTIM_sim.txt", times=export_times)

W_model.exports = [derived_quantities] + [TXT]

W_model.dt = F.Stepsize(
    initial_value=1e-5,
    stepsize_change_ratio=1.25,
    max_stepsize=500,
    dt_min=1e-6,
    milestones=export_times,
)

W_model.settings = F.Settings(
    absolute_tolerance=1e7,
    relative_tolerance=1e-13,
    final_time=final_time,
    traps_element_type="DG",
)

W_model.initialise()
W_model.run()
```

## Comparison with experimental data and MHIMS: D content

+++

The results produced by FESTIM are in good agreement with the experimental data and correlate perfectly with MHIMS.

```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d


def RMSPE(x_sim, x_exp):
    error = np.sqrt(np.mean((x_sim - x_exp) ** 2)) / np.mean(x_exp)
    return error


retention = (
    np.array(derived_quantities[0].data)
    + np.array(derived_quantities[1].data)
    + np.array(derived_quantities[2].data)
)
t = np.array(derived_quantities.t)

exp_ret = np.loadtxt("./reference_data/exp_ret.csv", delimiter=",", skiprows=1)

interp_ret = interp1d(t / 3600, retention, fill_value="extrapolate")

error = RMSPE(
    interp_ret(
        exp_ret[:, 0],
    ),
    exp_ret[:, 1],
)

print(f"RMSPE between FESTIM and experimental data is {error*100:.2f}%")

MHIMS_ret = np.loadtxt("./reference_data/MHIMS_ret.csv", delimiter=",", skiprows=1)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=t / 3600,
        y=retention / 1e20,
        mode="lines",
        line=dict(width=4, color=px.colors.qualitative.Plotly[1]),
        name="FESTIM",
    )
)

fig.add_trace(
    go.Scatter(
        x=MHIMS_ret[::100, 0],
        y=MHIMS_ret[::100, 1] / 1e20,
        mode="markers",
        marker_symbol="square",
        marker=dict(size=10, color=px.colors.qualitative.Plotly[2], opacity=0.6),
        name="MHIMS",
    )
)

fig.add_trace(
    go.Scatter(
        x=exp_ret[:, 0],
        y=exp_ret[:, 1] / 1e20,
        mode="markers",
        marker=dict(size=10, color=px.colors.qualitative.Plotly[0], opacity=0.6),
        name="Exp.",
    )
)

fig.update_yaxes(
    title_text="D inventory, 10<sup>20</sup> m<sup>-2</sup>",
    range=[0, 5],
    tick0=0,
    dtick=1,
)
fig.update_xaxes(title_text="Time, h", range=[0, 100], tick0=0, dtick=10)
fig.update_layout(template="simple_white", height=600)

# The writing-reading block below is needed to avoid the issue with compatibility
# of Plotly plots and dollarmath syntax extension in Jupyter Book
# For mode details, see https://github.com/jupyter-book/jupyter-book/issues/1528

fig.write_html("./markelj_comparison_ret.html")
from IPython.display import HTML, display

display(HTML("./markelj_comparison_ret.html"))
```

## Comparison with experimental data: D depth distribution

+++

FESTIM reproduces well the experimental NRA measurements.

```{code-cell} ipython3
:tags: [hide-input]

fig = go.Figure()

FESTIM_profiles = np.genfromtxt("./FESTIM_sim.txt", names=True, delimiter=",")
NRA_exp = np.loadtxt("./reference_data/exp_NRA.csv", delimiter=",", skiprows=1)
NRA_sto = np.loadtxt("./reference_data/sto_NRA.csv", delimiter=",", skiprows=1)


def create_slider(fig):
    steps = []
    for i in range(0, len(fig.data), 2):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": "Time: " + f"{export_times[int(i/2)]/3600:.2f}"},
            ],  # layout attribute
            label=f"{export_times[int(i/2)]/3600:.2f} h",
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][i + 1] = True  # Toggle i+1'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(active=0, currentvalue={"prefix": "Time: "}, pad={"t": 50}, steps=steps)
    ]

    return sliders


for i, t in enumerate(export_times):
    x = FESTIM_profiles["x"]
    y = FESTIM_profiles[f"t{t:.2e}s".replace(".", "").replace("+", "")]
    # order y by x
    x, y = zip(*sorted(zip(x, y)))

    color = px.colors.qualitative.Plotly[i]

    fig.add_trace(
        go.Scatter(
            x=np.array(x) / 1e-6,
            y=np.array(y) / rho_W * 100,
            mode="lines",
            line=dict(width=3.5, color=px.colors.qualitative.Plotly[i]),
            name="FESTIM",
            visible=False,
        )
    )

    if i <= 4:
        fig.add_trace(
            go.Scatter(
                x=NRA_exp[:, 0],
                y=NRA_exp[:, i + 1],
                mode="lines",
                line=dict(
                    width=3.5, color=px.colors.qualitative.Plotly[i], dash="dash"
                ),
                name="NRA",
                visible=False,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=NRA_sto[:, 0],
                y=NRA_sto[:, i - 4],
                mode="lines",
                line=dict(
                    width=3.5, color=px.colors.qualitative.Plotly[i], dash="dash"
                ),
                name="NRA",
                visible=False,
            )
        )

fig.data[0].visible = True
fig.data[1].visible = True

fig.update_yaxes(title_text="D concentration, at.%", range=[0, 0.5], tick0=0, dtick=0.1)
fig.update_xaxes(title_text="Depth, &#181;m", range=[0, 5], tick0=0, dtick=1)
fig.update_layout(template="simple_white", sliders=create_slider(fig), height=600)

# The writing-reading block below is needed to avoid the issue with compatibility
# of Plotly plots and dollarmath syntax extension in Jupyter Book
# For mode details, see https://github.com/jupyter-book/jupyter-book/issues/1528

fig.write_html("./markelj_profiles_exp.html")

display(HTML("./markelj_profiles_exp.html"))
```

## Comparison with MHIMS: D depth distribution

+++

FESTIM agrees with MHIMS. Slight differences are due to the use of the precise value of Fernandez's diffusivity pre-factor. FESTIM uses the value from [HTM](https://github.com/RemDelaporteMathurin/h-transport-materials) divided by $\sqrt{2}$: $1.93 \times 10^{-7} / \sqrt{2}$, whereas the presented diffusivity pre-factor in {cite}`hodille_2017` is $1.9 \times 10^{-7} / \sqrt{2}$.

```{code-cell} ipython3
:tags: [hide-input]

fig = go.Figure()

FESTIM_profiles = np.genfromtxt("./FESTIM_sim.txt", names=True, delimiter=",")
MHIMS_profiles = np.loadtxt(
    "./reference_data/MHIMS_profiles.csv", delimiter=",", skiprows=1
)

for i, t in enumerate(export_times):
    x = FESTIM_profiles["x"]
    y = FESTIM_profiles[f"t{t:.2e}s".replace(".", "").replace("+", "")]
    # order y by x
    x, y = zip(*sorted(zip(x, y)))

    color = px.colors.qualitative.Plotly[i]

    fig.add_trace(
        go.Scatter(
            x=np.array(x) / 1e-6,
            y=np.array(y) / rho_W * 100,
            mode="lines",
            line=dict(width=3.5, color=px.colors.qualitative.Plotly[i]),
            name="FESTIM",
            visible=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=MHIMS_profiles[:, 0],
            y=MHIMS_profiles[:, i + 1],
            mode="markers",
            marker=dict(size=10, color=px.colors.qualitative.Plotly[i], opacity=0.4),
            name="MHIMS",
            visible=False,
        )
    )

fig.data[0].visible = True
fig.data[1].visible = True

fig.update_yaxes(title_text="D concentration, at.%", range=[0, 0.5], tick0=0, dtick=0.1)
fig.update_xaxes(title_text="Depth, &#181;m", range=[0, 5], tick0=0, dtick=1)
fig.update_layout(template="simple_white", sliders=create_slider(fig), height=600)

# The writing-reading block below is needed to avoid the issue with compatibility
# of Plotly plots and dollarmath syntax extension in Jupyter Book
# For mode details, see https://github.com/jupyter-book/jupyter-book/issues/1528

fig.write_html("./markelj_profiles_MHIMS.html")

display(HTML("./markelj_profiles_MHIMS.html"))
```
