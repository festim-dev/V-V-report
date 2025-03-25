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

# Deuterium implantation in EUROFER damaged with tungsten ions

```{tags} 1D, kinetic surface model, TDS, transient
```

+++

This validation case reproduces TDS measurements of D from EUROFER performed by Schmid et al. {cite}`schmid_2023a`.

Experiments were conducted with three types of EUROFER samples (0.8 mm thick): undamaged, damaged with 20 MeV W ions, loaded with D and then damaged with 20 MeV W ions. These samples were then loaded with low-energy (5 ev/ion) D flux of $\approx9\times10^{19}\,\mathrm{m}^{-2}\mathrm{s}^{-1}$ at the gas pressure of 1 Pa and $T=370$ K. The exposure time varied between 48 h to 143 h resulting in four cases: 
1. undamaged sample loaded for 143 h with D; 
2. damaged sample loaded for 48 h with D; 
3. damaged sample loaded for 143 h with D; 
4. pre-loaded damaged sample exposed for 48 h with D. 

After exposure, the samples were stored for $\approx24$ h at $T=290 \ K$. Finally, TDS measurements up to 800 K were performed.

The FESTIM model is mainly based on the simulations of Schmid et al. {cite}`schmid_2023b`. The D diffusivity in EUROFER is taken from the work of Aiello et al. {cite}`aiello_2002`. In all cases, intrinsic trapping sites are considered with a homogeneous distribution within the sample. For pre-damaged samples, additional extrinsic traps, distributed within the damaged zone, is included in the simulations. The surface kinetics is determined by adsoption from the gas phase, recombination, and desorption due to the incidence of energetic D ions. Only the front surface is assumed to be subjected to the D flux. 

To reproduce the experimental data, the input parameters for the TESSIM-X simulations, performed by Schmid et al. {cite}`schmid_2023a`, were used. The obtained FESTIM results are then compared with experimental and TESSIM-X data.

+++

## FESTIM model

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import festim as F
import fenics as f
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import special


# Monkey patch the C99CodePrinter class
# this is to avoid the bug observed in https://github.com/festim-dev/FESTIM/issues/813
from sympy.printing.c import C99CodePrinter

original_print_function = C99CodePrinter._print_Function
log_bis = sp.Function("std::log")


def _custom_print_Function(self, expr):
    if expr.func == log_bis:
        return f"std::log({self._print(expr.args[0])})"
    return original_print_function(self, expr)


C99CodePrinter._print_Function = _custom_print_Function

################### PARAMETERS ###################
N_A_const = 6.022e23  # Avogadro, mol^-1
e = 1.602e-19
M_D2 = 4.028e-3 / N_A_const  # the D2 mass, kg mol^-1

# Exposure conditions
P_D2 = 1  # Pa
T_load = 370  # D loading temperature, K
T_storage = 290  # temperature after cooling phase, K
ramp = 3 / 60  # TDS heating rate, K/s
t_cool = 1000  # colling duration, s
t_storage = 24 * 3600  # storage time, s
t_TDS = (800 - T_storage) / ramp  # TDS duration (up to 800 K), s

# Sample
L = 0.8e-3  # half thickness, m

# EUROFER properties
n_EFe = 8.59e28  # EUROFER atomic concentration, m^-3
n_IS = 6 * n_EFe  # concentration of interstitial sites, m^-3
n_surf = n_EFe ** (2 / 3)  # concentration of adsorption sites, m^-2
lambda_lat = n_EFe ** (-1 / 3)  # Typical lattice spacing, m

D0 = 1.5e-7  # diffusivity pre-factor, m^2 s^-1
E_diff = F.kJmol_to_eV(14.470)  # diffusion activation energy, eV

# Energy landscape
E_bs = E_diff  # energy barrier from bulk to surface transition, eV
nu_bs = D0 / lambda_lat**2  # attempt frequency for b-to-s transition, s^-1
E_diss = 0.4  # energy barrier for D2 dissociation, eV
E_rec = 0.63  # energy barrier for D2 recombination, eV
E_sol = 0.238  # heat of solution, eV
S0 = 1.5e-6 * n_EFe  # solubility pre-factor, m^-3 Pa^-0.5
Xi0 = 1e-5  # adsorption rate pre-factor
chi0 = 1e-7  # recombination rate pre-factor
E_sb = (
    E_rec / 2 - E_diss / 2 + E_sol + E_diff
)  # energy barrier from bulk to surface transition, eV

# Trap properties
nu_tr = D0 / lambda_lat**2  # trapping attempt frequency, s^-1
nu_dt = 4.0e13  # detrapping attempt frequency, s^-1
E_tr = E_diff
E_dt_intr = 0.9  # detrapping energy for intrinsic traps, eV
E_dt_dpa = 1.08  # detrapping energy for DPA traps, eV

# Implantation parameters
indicent_flux = 9e19  # irradiation flux, m^-2 s^-1
R_impl = -1.0e-10  # implantation range, m
sigma = 7.5e-10 / np.sqrt(2)
refl_coeff = 0.612  # reflection coefficient


################### FUNCTIONS ###################
def Xi(T):
    # unitless
    return Xi0 * f.exp(-E_diss / F.k_B / T)


def chi(T):
    # in m^2 s^-1
    return chi0 * f.exp(-E_rec / F.k_B / T)


def S(T):
    # solubility m^-3 Pa^-0.5
    return S0 * f.exp(-E_sol / F.k_B / T)


def Psi(T):
    return 1 / f.sqrt(2 * np.pi * M_D2 * F.k_B * T * e)


def k_bs(T, surf_conc, t):
    # n_IS / n_EFe is needed to obtain lambda_abs=n_surf/n_EFe in the final
    # expression for the bulk-to-surface flux of atoms as used in TESSIM
    return nu_bs * f.exp(-E_bs / F.k_B / T) * n_IS / n_EFe


def k_sb(T, surf_conc, t):
    # see eqs. (13-14) in K. Schmid and M. Zibrov 2021 Nucl. Fusion 61 086008
    K_bs = nu_bs * f.exp(-E_bs / F.k_B / T)
    return K_bs * S(T) * lambda_lat * f.sqrt(chi(T) / Psi(T) / Xi(T))


def normal_distr(X, sigma):
    return 2 / (1 + special.erf(X / np.sqrt(2) / sigma))


def temperature_after_load(t, t_load: float, mod):
    """Temperature evolution after the loading phase

    Args:
        t: sp.Symbol or float or np.NDArray, the absolute time in seconds
        t_load (float): the duration of the loading phase in seconds
        mod: sympy or numpy module

    Returns:
        appropriate object for the temperature evolution in K
    """
    if mod == sp:
        log = log_bis
    else:
        log = mod.log

    a1 = log(mod.cosh(0.005 * (-612700 + (143 * 3600 - t_load) + t)))
    a2 = log(mod.cosh(0.005 * (-607300 + (143 * 3600 - t_load) + t)))
    a3 = log(mod.cosh(0.005 * (-603200 + (143 * 3600 - t_load) + t)))
    a4 = log(mod.cosh(0.005 * (-603200 + (143 * 3600 - t_load) + t)))
    a5 = log(mod.cosh(0.005 * (-602200 + (143 * 3600 - t_load) + t)))

    value = (
        293.55
        + 50
        * (
            0
            - 0.05194 * a1
            + 0.05194
            * (
                -3035.806852819440
                - (3035.806852819440 + a1)
                + 2 * (3062.806852819440 + a2)
            )
        )
        + 50
        * (
            0
            - 0.06 * a2
            + 0.06
            * (
                -3020.806852819440
                - (3020.806852819440 + a2)
                + 2 * (3035.806852819440 + a3)
            )
        )
        + 50
        * (
            0
            - 0.04 * a3
            + 0.04
            * (
                -3015.306852819440
                - (3015.306852819440 + a3)
                + 2.00003 * (3020.806852819440 + a4)
            )
        )
        + 50
        * (
            0
            - 0.00339 * a4
            + 0.00339
            * (
                -3010.30685
                - (3010.306852819440 + a4)
                + 2.00009 * (3015.306852819440 + a5)
            )
        )
        + 76.45 * 0.5 * (1 - mod.tanh(0.002 * (t - 515800 + (143 * 3600 - t_load))))
    )
    return value


def run_simulation(t_load: float, is_dpa: bool, dpa_conc: float):

    def J_vs(T, surf_conc, t):

        tt = 0.002 * (t - t_load)
        cond = 0.5 - 0.5 * (f.exp(2 * tt) - 1) / (f.exp(2 * tt) + 1)

        J_diss = (
            2 * P_D2 * Xi(T) * (1 - surf_conc / n_surf) ** 2 * Psi(T)
        )  # dissociation flux

        J_rec = 2 * chi(T) * surf_conc**2  # recombination flux

        Omega_loss = 1.4e5
        J_loss = (
            (surf_conc / n_surf) * Omega_loss * indicent_flux * (1 - refl_coeff)
        )  # ad hoc flux for fit

        J_net = (J_diss - J_loss) * cond - J_rec
        return J_net

    def J_vs_r(T, surf_conc, t):

        tt = 0.002 * (t - t_load)
        cond = 0.5 - 0.5 * (f.exp(2 * tt) - 1) / (f.exp(2 * tt) + 1)

        J_diss = (
            2 * 1e-12 * Xi(T) * (1 - surf_conc / n_surf) ** 2 * Psi(T)
        )  # dissociation flux

        J_rec = 2 * chi(T) * surf_conc**2  # recombination flux

        J_net = J_diss * cond - J_rec
        return J_net

    EFe_model = F.Simulation(log_level=40)

    # Mesh
    vertices = np.concatenate(
        [
            np.linspace(0, 1e-8, num=100),
            np.linspace(1e-8, 4e-6, num=200),
            np.linspace(4e-6, L - 1e-8, num=250),
            np.linspace(L - 1e-8, L, num=100),
        ]
    )

    EFe_model.mesh = F.MeshFromVertices(vertices)

    EFe_model.materials = [F.Material(id=1, D_0=D0, E_D=E_diff)]

    surf_conc1 = F.SurfaceKinetics(
        k_sb=k_sb,
        k_bs=k_bs,
        lambda_IS=lambda_lat,
        n_surf=n_surf,
        n_IS=n_IS,
        J_vs=J_vs,
        surfaces=1,
        initial_condition=0,
        t=F.t,
    )

    surf_conc2 = F.SurfaceKinetics(
        k_sb=k_sb,
        k_bs=k_bs,
        lambda_IS=lambda_lat,
        n_surf=n_surf,
        n_IS=n_IS,
        J_vs=J_vs_r,
        surfaces=2,
        initial_condition=0,
        t=F.t,
    )

    EFe_model.boundary_conditions = [surf_conc1, surf_conc2]

    trap_intr = F.Trap(
        k_0=nu_tr / n_IS,
        E_k=E_tr,
        p_0=nu_dt,
        E_p=E_dt_intr,
        density=1e-5 * n_EFe,
        materials=EFe_model.materials[0],
    )
    trap_dpa = F.Trap(
        k_0=nu_tr / n_IS,
        E_k=E_tr,
        p_0=nu_dt,
        E_p=E_dt_dpa,
        density=(0.5 * dpa_conc * (1 - sp.tanh((F.x - 3.3e-6) / 0.01e-6))) * n_EFe,
        materials=EFe_model.materials[0],
    )

    EFe_model.traps = [trap_intr]
    if is_dpa:
        EFe_model.traps.append(trap_dpa)

    EFe_model.sources = [
        F.ImplantationFlux(
            flux=indicent_flux
            * (1 - refl_coeff)
            * normal_distr(R_impl, sigma)
            * 0.5
            * (1 - 1 * sp.tanh(0.002 * (F.t - t_load))),
            imp_depth=R_impl,
            width=sigma,
            volume=1,
        )
    ]

    EFe_model.T = F.Temperature(
        value=sp.Piecewise(
            (T_load, F.t <= t_load),
            (temperature_after_load(F.t, t_load, mod=sp), True),
        )
    )  # This temperature function is defined based on the TESSIM model

    def max_step_size(t):
        if t <= t_load:
            return 500
        elif t > t_load and t <= t_load + t_cool + t_storage:
            return 1000
        else:
            return 30

    EFe_model.dt = F.Stepsize(
        initial_value=1e-4,
        stepsize_change_ratio=1.1,
        max_stepsize=max_step_size,
        dt_min=1e-5,
        milestones=[
            t_load,
            t_load + t_cool,
            t_load + t_cool + t_storage,
        ],
    )

    EFe_model.settings = F.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-10,
        maximum_iterations=50,
        final_time=t_load + t_cool + t_storage + t_TDS,
    )

    derived_quantities = F.DerivedQuantities(
        [
            F.AdsorbedHydrogen(surface=1),
            F.AdsorbedHydrogen(surface=2),
            F.TotalSurface(field="T", surface=1),
        ],
        show_units=True,
    )

    EFe_model.exports = [derived_quantities]

    EFe_model.initialise()
    EFe_model.run()

    return derived_quantities


params = [
    {
        "t_load": 143 * 3600,
        "is_dpa": False,
        "dpa_conc": 0,
        "exp_data": "143hplasma",
    },
    {
        "t_load": 48 * 3600,
        "is_dpa": True,
        "dpa_conc": 2.5e-4,
        "exp_data": "DPA+48hplasma",
    },
    {
        "t_load": 143 * 3600,
        "is_dpa": True,
        "dpa_conc": 2.5e-4,
        "exp_data": "DPA+143hplasma",
    },
    {
        "t_load": 48 * 3600,
        "is_dpa": True,
        "dpa_conc": 5.0e-4,
        "exp_data": "DPA+D+48hplasma",
    },
]

results = {}
for i, prms in enumerate(params):
    results[i] = run_simulation(
        t_load=prms["t_load"], is_dpa=prms["is_dpa"], dpa_conc=prms["dpa_conc"]
    )
```

## Comparison with experiment

+++

FESTIM reproduces general trends of experimental TDS curves.

```{code-cell} ipython3
:tags: [hide-input]

import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from itertools import product


def total_flux(data):
    surf_conc1 = np.array(data[0].data)
    surf_conc2 = np.array(data[1].data)
    T = np.array(data[2].data)
    flux_left = 2 * surf_conc1**2 * chi0 * np.exp(-E_rec / F.k_B / T)
    flux_right = 2 * surf_conc2**2 * chi0 * np.exp(-E_rec / F.k_B / T)
    total_flux = flux_left + flux_right

    return T, total_flux

titles = [
    "Case 1: 143 h plasma",
    "Case 2: DPA &#8594; 48 h plasma",
    "Case 3: DPA &#8594; 143 h plasma",
    "Case 4: DPA+D &#8594; 48 h plasma",
]

fig = make_subplots(
    rows=2,
    cols=2,
    vertical_spacing=0.1,
    horizontal_spacing=0.05,
    shared_yaxes=True,
    shared_xaxes=True,
    x_title="Temperature, K",
    y_title="Desorption flux, 10<sup>17</sup> m<sup>-2</sup>s<sup>-1</sup>",
    subplot_titles=titles,
)

for i, (row, col) in enumerate(product(range(1, 3), range(1, 3))):
    prms = params[i]

    T, FESTIM_flux = total_flux(results[i])

    # Experimental data
    exp_data = json.load(open(f"./reference_data/{prms['exp_data']}.json"))

    fig.add_trace(
        go.Scatter(
            x=T,
            y=FESTIM_flux / 1e17,
            mode="lines",
            line=dict(width=3, color=px.colors.qualitative.Plotly[i]),
            name="FESTIM",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=exp_data["temptab"][::5],
            y=np.array(exp_data["experiment"][::5]) / 1e5,
            mode="markers",
            marker=dict(size=7, color=px.colors.qualitative.Plotly[i], opacity=0.5),
            name="Experiment",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    fig.update_yaxes(range=[0, 1.5], tick0=0, dtick=0.5, col=col, row=row)
    fig.update_xaxes(range=[300, 800], tick0=0, dtick=100, col=col, row=row)

fig.update_layout(template="simple_white", height=600)

# The writing-reading block below is needed to avoid the issue with compatibility
# of Plotly plots and dollarmath syntax extension in Jupyter Book
# For mode details, see https://github.com/jupyter-book/jupyter-book/issues/1528

fig.write_html("./schmid_comparison_exp.html")
from IPython.display import HTML, display

display(HTML("./schmid_comparison_exp.html"))
```

## Comparison with TESSIM-X

+++

 FESTIM correlates moderately with the TESSIM-X data. Minor discrepancy could be due to the differences in some input parameters and the kinetic surface model.

```{code-cell} ipython3
:tags: [hide-input]

fig = make_subplots(
    rows=2,
    cols=2,
    vertical_spacing=0.1,
    horizontal_spacing=0.05,
    shared_yaxes=True,
    shared_xaxes=True,
    x_title="Temperature, K",
    y_title="Desorption flux, 10<sup>17</sup> m<sup>-2</sup>s<sup>-1</sup>",
    subplot_titles=titles,
)

for i, (row, col) in enumerate(product(range(1, 3), range(1, 3))):
    prms = params[i]

    T, FESTIM_flux = total_flux(results[i])

    # Experimental data
    exp_data = json.load(open(f"./reference_data/{prms['exp_data']}.json"))

    fig.add_trace(
        go.Scatter(
            x=T,
            y=FESTIM_flux / 1e17,
            mode="lines",
            line=dict(width=3, color=px.colors.qualitative.Plotly[i]),
            name="FESTIM",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=exp_data["temptab"],
            y=np.array(exp_data["simflux"]) / 1e5,
            mode="lines",
            line=dict(width=3, color=px.colors.qualitative.Plotly[i], dash="dash"),
            name="TESSIM-X",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    fig.update_yaxes(range=[0, 1.5], tick0=0, dtick=0.5, col=col, row=row)
    fig.update_xaxes(range=[300, 800], tick0=0, dtick=100, col=col, row=row)

fig.update_layout(template="simple_white", height=600)

# The writing-reading block below is needed to avoid the issue with compatibility
# of Plotly plots and dollarmath syntax extension in Jupyter Book
# For mode details, see https://github.com/jupyter-book/jupyter-book/issues/1528

fig.write_html("./schmid_comparison_TESSIM.html")

display(HTML("./schmid_comparison_TESSIM.html"))
```
