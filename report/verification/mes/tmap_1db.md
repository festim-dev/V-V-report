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
mystnb:
  execution_mode: "off"
---

# Strong trapping regime

```{tags} 1D, MES, transient, trapping
```

This verification case consists of a slab of depth $l = 1 \ \mathrm{m}$ with one trap under the strong trapping regime.

A trapping parameter $\zeta$ is defined by

$$
    \zeta = \frac{\lambda ^ 2 \nu}{D_0 \rho} \exp \left(\frac{E_\mathrm{d} - E_p}{k_\mathrm{B} T}\right) + \frac{c_\mathrm{m}}{\rho}
$$

where

$\lambda \ \mathrm{(m)}$ is the lattice parameter, \
$\nu \ (\mathrm{s}^{-1})$, the Debye frequency \
$\rho$ is the trapping site fraction, \
$c_\mathrm{m}$ is the mobile atom fraction.

In the strong trapping regime, $\zeta \approx c_\mathrm{m} / \rho$, and no permeation occurs until essentially all the traps have been filled.

The breakthrough time is given by

$$
    \tau = \frac{l^2 \rho}{2 c_{\mathrm{m},0} D}
$$

where $c_{\mathrm{m},0}$ is the steady concentration of mobile atoms at $x=0$. This analytical solution was obtained from TMAP7's V&V report {cite}`ambrosek_verification_2008`, case Val-1db.

For this case, $\lambda=\sqrt{10^{-15}} \ \mathrm{m}$, $\nu=10^{13} \ \mathrm{s}^{-1}$, $D=1 \ \mathrm{m}^2 \mathrm{s}^{-1}$, and $E_p/k_\mathrm{B}=10000 \ \mathrm{K}$ are used to obtain $\zeta \approx 1.00454 c_\mathrm{m} / \rho$.

+++

## FESTIM Code

```{code-cell} ipython3
import festim as F
import numpy as np

# Define input parameters
n = 3.162e22
rho = 1e-1 * n
D_0 = 1
E_D = 0.0
k_0 = 1e15 / n
p_0 = 1e13
E_p = 10000 * F.k_B
T = 1000
sample_depth = 1

c_m = 1e-4 * n

# Create the FESTIM model
my_model = F.HydrogenTransportProblem()

vertices = np.linspace(0, sample_depth, num=100)
my_model.mesh = F.Mesh1D(vertices)

material = F.Material(D_0=D_0, E_D=E_D)
volume = F.VolumeSubdomain1D(id=1, material=material, borders=[0, sample_depth])
left_boundary = F.SurfaceSubdomain1D(id=1, x=0)
right_boundary = F.SurfaceSubdomain1D(id=2, x=sample_depth)

my_model.subdomains = [left_boundary, volume, right_boundary]

mobile_H = F.Species("H")
trapped_H = F.Species("trapped_H", mobile=False)
empty_trap = F.ImplicitSpecies(n=rho, others=[trapped_H])
my_model.species = [mobile_H, trapped_H]

trapping_reaction = F.Reaction(
    reactant=[mobile_H, empty_trap],
    product=[trapped_H],
    k_0=k_0,
    E_k=E_D,
    p_0=p_0,
    E_p=E_p,
    volume=volume,
)
my_model.reactions = [trapping_reaction]

my_model.temperature = T

my_model.boundary_conditions = [
    F.DirichletBC(subdomain=left_boundary, value=c_m, species=mobile_H),
    F.DirichletBC(subdomain=right_boundary, value=0, species=mobile_H),
]

my_model.settings = F.Settings(atol=2e15, rtol=5e-8, max_iterations=30, final_time=1000)

my_model.settings.stepsize = F.Stepsize(
    initial_value=1e-6,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=30,
    max_stepsize=2,
)

right_flux = F.SurfaceFlux(field=mobile_H, surface=right_boundary)

my_model.exports = [right_flux]

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
import plotly.express as px

# plot computed solution
t = np.array(right_flux.t)
computed_solution = np.array(right_flux.data)

time_exact = sample_depth**2 * rho / (2 * c_m * D_0)

fig = go.Figure()

fig.add_traces(
    go.Scatter(
        x=t,
        y=computed_solution,
        mode="lines",
        line=dict(width=3, color=px.colors.qualitative.Plotly[1]),
        name="FESTIM",
    )
)

fig.add_traces(
    go.Scatter(
        x=[time_exact, time_exact],
        y=[0, 4e18],
        mode="lines",
        line=dict(width=3, color=px.colors.qualitative.Plotly[0], dash="dash"),
        name="Analytical breakthrough time",
    )
)

fig.update_xaxes(title_text="Time, s", range=[0, 1000])
fig.update_yaxes(
    title_text="Downstream flux, H m<sup>-2</sup>s<sup>-1</sup>", range=[0, 3.25e18]
)
fig.update_layout(template="simple_white", height=600)

# The writing-reading block below is needed to avoid the issue with compatibility
# of Plotly plots and dollarmath syntax extension in Jupyter Book
# For mode details, see https://github.com/jupyter-book/jupyter-book/issues/1528
fig.write_html("./tmap_1db.html")
from IPython.display import HTML, display

display(HTML("./tmap_1db.html"))
```
