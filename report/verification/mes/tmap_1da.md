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

# Effective diffusivity regime

```{tags} 1D, MES, transient, trapping
```

This verification case consists of a slab of depth $l = 1 \ \mathrm{m}$ with one trap under the effective diffusivity regime.

A trapping parameter $\zeta$ is defined by

$$
    \zeta = \frac{\lambda ^ 2 \nu}{D_0 \rho} \exp \left(\frac{E_\mathrm{d} - E_p}{k_\mathrm{B} T}\right) + \frac{c_\mathrm{m}}{\rho}
$$

where

$\lambda \ \mathrm{(m)}$ is the lattice parameter, \
$\nu \ (\mathrm{s}^{-1})$, the Debye frequency \
$\rho$ is the trapping site fraction, \
$c_\mathrm{m}$ is the mobile atom fraction.

In the effective diffusivity regime, $\zeta \gg c_\mathrm{m} / \rho$ and the hydrogen transport can be described with an effective diffusivity $D_\mathrm{eff}$:

$$
    D_\text{eff} = \frac{D}{1 + \frac{1}{\zeta}}
$$

Then with a breakthrough time $\tau = \frac{l^2}{2\pi^2 D_\text{eff}}$, the exact solution for flux is

$$
    J = \frac{c_{\mathrm{m},0} D}{l} \left[ 1 + 2\sum_{m=1}^\infty (-1)^m \exp \left( -m^2 \frac{t}{\tau} \right) \right]
$$

where $c_{\mathrm{m},0}$ is the steady concentration of mobile atoms at $x=0$. This analytical solution was obtained from TMAP7's V&V report {cite}`ambrosek_verification_2008`, case Val-1da.

For this case, $\lambda=\sqrt{10^{-15}} \ \mathrm{m}$, $\nu=10^{13} \ \mathrm{s}^{-1}$, $D=1 \ \mathrm{m}^2 \mathrm{s}^{-1}$, and $E_p/k_\mathrm{B}=100 \ \mathrm{K}$ are used to obtain $\zeta \approx 91.48 c_\mathrm{m} / \rho$.

+++

## FESTIM Code

```{code-cell}
import festim as F
import numpy as np

# Define input parameters
n = 3.162e22
rho = 1e-1 * n
D_0 = 1
E_D = 0.0
k_0 = 1e15 / n
p_0 = 1e13
E_p = 100 * F.k_B
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

my_model.settings = F.Settings(atol=2e15, rtol=5e-8, max_iterations=30, final_time=10)

my_model.settings.stepsize = F.Stepsize(0.05)

right_flux = F.SurfaceFlux(field=mobile_H, surface=right_boundary)

my_model.exports = [right_flux]

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

```{code-cell}
:tags: [hide-input]

import plotly.graph_objects as go
import plotly.express as px

a = np.sqrt(1e-15)
zeta = (a**2 * 1e13 * n / D_0 * np.exp((E_D - E_p) / F.k_B / T) + c_m) / rho
D_eff = D_0 / (1 + 1 / zeta)

# plot computed solution
t = np.array(right_flux.t)
computed_solution = np.array(right_flux.data)

fig = go.Figure()

festim_plot = go.Scatter(
    x=t,
    y=computed_solution,
    mode="lines",
    line=dict(width=3, color=px.colors.qualitative.Plotly[1]),
    name="FESTIM",
)

# plot exact solution
m = np.arange(1, 10001)

# Calculate the exponential part for all m values at once
tau = sample_depth**2 / (2 * np.pi**2 * D_eff)
exp_part = np.exp(-(m**2) * t[:, None] / 2 / tau)

# Calculate the 'add' part for all m values and sum them up for each t
add = 2 * (-1) ** m * exp_part
series = 1 + add.sum(axis=1)  # Sum along the m dimension and add 1 for the initial ones

exact_solution = c_m * D_0 / sample_depth * series

exact_plot = go.Scatter(
    x=t,
    y=exact_solution,
    mode="markers",
    marker=dict(size=9, opacity=0.5, color=px.colors.qualitative.Plotly[0]),
    name="exact",
)

fig.add_traces([exact_plot, festim_plot])
fig.update_xaxes(title_text="Time, s", range=[0, 10])
fig.update_yaxes(title_text="Downstream flux, H m<sup>-2</sup>s<sup>-1</sup>")
fig.update_layout(template="simple_white", height=600)

# The writing-reading block below is needed to avoid the issue with compatibility
# of Plotly plots and dollarmath syntax extension in Jupyter Book
# For mode details, see https://github.com/jupyter-book/jupyter-book/issues/1528
fig.write_html("./tmap_1da.html")
from IPython.display import HTML, display

display(HTML("./tmap_1da.html"))
```
