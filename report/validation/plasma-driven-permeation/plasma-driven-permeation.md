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

# Plasma-driven permeation

This validation case is a plasma-driven permeation performed at INL in 1985 {cite}`anderl_tritium_1986`.
Deuterium ions at 3 keV was implanted in a 0.5 mm thick sample of 316 stainless steel variant called primary candidate alloy (PCA).

The ion beam was turned on and off repeatedly and it is estimated that the implanted flux is $4.9 \times 10^{19} \ \mathrm{D \ m^{-2} s^{-1}}$:

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

time = [0.0, 6420.0, 6420.1, 9420.0, 9420.1, 12480.0, 12480.1, 14940.0, 14940.1, 18180.0, 18180.1, 1.0e10]  # taken from the TMAP4 report
flux = [4.9e19, 4.9e19, 0.0, 0.0, 4.9e19, 4.9e19, 0.0, 0.0, 4.9e19, 4.9e19, 0.0, 0.0]

plt.plot(time[:-1], flux[:-1])
plt.xlabel('Time (s)')
plt.ylabel('Implantation flux (D/m$^2$/s)')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.fill_between(time[:-1], flux[:-1], alpha=0.3)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.grid(alpha=0.3)
plt.show()
```

SRIM calculations performed in the TMAP4 {cite}`longhurst_verification_1992` and TMAP7 {cite}`ambrosek_verification_2008` V&V reports showed that at this energy, deuterium ions were implanted at $11 \ \mathrm{nm} \pm 5.4 \ \mathrm{nm}$.

A normal distribution was therefore used with a mean at $12 \ \mathrm{nm}$ and a standard deviation of $2.3 \ \mathrm{nm}$.

```{note}
These normal distribution parameters were obtained by comparing the TMAP4 source distribution. See below.
```

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

deltax = [
    0.0,
    5 * [4.0e-9],
    1.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-5,
    10 * [4.88e-5],
]  # taken from the TMAP4 report
x = 0
x_points = []
for i in deltax:
    if isinstance(i, list):
        for j in i:
            x += j
            x_points.append(x)
    else:
        x += i
        x_points.append(x)

source_distrib_at_nodes = [
    3 * [0.0],
    0.25,
    1,
    0.25,
    15 * [0.0],
]  # taken from the TMAP4 report
source_distribution = []
for i in source_distrib_at_nodes:
    if isinstance(i, list):
        for j in i:
            source_distribution.append(j)
    else:
        source_distribution.append(i)

width = 2.4e-9
mean = 12e-9
x_festim = np.linspace(0, 40e-9, 200)
festim_source_distrib = (
    1 / (width * (2 * np.pi) ** 0.5) * np.exp(-0.5 * ((x_festim - mean) / width) ** 2)
)


area = np.trapz(source_distribution[1:], x_points)
source_distribution = np.array(source_distribution) / area

plt.plot(x_festim, festim_source_distrib, color="tab:orange")
plt.fill_between(x_festim, festim_source_distrib, alpha=0.3, color="tab:orange", label="Gaussian distrib.")
plt.plot(x_points, source_distribution[1:], label="TMAP7", marker="o")

plt.xlim(0, 40e-9)
plt.ylim(bottom=0)
plt.xlabel("Depth (m)")
plt.ylabel("Normalized source distribution")
plt.legend()
plt.gca().spines[["top", "right"]].set_visible(False)
plt.grid(alpha=0.3)
plt.show()
```

The diffusion coefficient was taken as $D = 3 \times 10^{-10} \ \mathrm{m^2 \ s^{-1}}$.

2nd order recombination fluxes were set on the boundaries.

On $x=0$ (the beam-facing surface), the recombination coefficient ($\mathrm{m^4 \ atom^{-1} \ s^{-1}}$) was $K_\mathrm{r} = 1.0 \times 10^{-27} \ \left( 1 - 0.9999 \ \exp{-1.2\times 10^{-4} \ t} \right) $.
The time dependent term was added to mimick the surface cleanup: as the sample is exposed the recombination coefficient increases and approches $1.0 \times 10^{-27} \ \mathrm{m^4 \ atom^{-1} \ s^{-1}}$.

On $x = 0.5 \ \mathrm{mm}$ (the non-exposed surface), the recombination coefficient was taken as $K_\mathrm{r} = 2.0 \times 10^{-31} \ \mathrm{m^4 \ atom^{-1} \ s^{-1}}$.


```{note}
This parametrisation was taken from the TMAP4 V&V report {cite}`longhurst_verification_1992`.
The authors originally added dissociation fluxes that would model the impact of low pressures in the upstream and downstream volumes but it turns out to be negligible.
The TMAP7 report has a slightly different parametrisation. We were able to reproduce their absolute values but there was an uncertainty on the units therefore we used the TMAP4 parameters.
```

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import festim as F
import sympy as sp

my_model = F.Simulation()

vertices = np.concatenate(
    [
        np.linspace(0, 20e-9, 50),
        np.linspace(20e-9, 3e-6, 500),
        np.linspace(3e-6, 0.5e-3, 500),
    ]
)

my_model.mesh = F.MeshFromVertices(vertices)

my_model.materials = F.Material(id=1, D_0=3e-10, E_D=0)


my_model.boundary_conditions = [
    F.RecombinationFlux(
        Kr_0=1.0e-27 * (1.0 - 0.9999 * sp.exp(-6.0e-5 * F.t)),
        E_Kr=0,
        surfaces=[1],
        order=2,
    ),
    F.RecombinationFlux(Kr_0=2.0e-31, E_Kr=0, surfaces=[2], order=2),
]

flux = sp.Piecewise(
    (4.9e19, F.t < 5820),
    (0, F.t < 9060),
    (4.9e19, F.t < 12160),
    (0, F.t < 14472),
    (4.9e19, F.t < 17678),
    (0, F.t < 1e10),
    (0, True),
)
my_model.sources = [
    F.ImplantationFlux(flux=flux, imp_depth=12e-9, width=2.4e-9, volume=1)
]

my_model.T = F.Temperature(500)  # ignored here

left_flux = F.SurfaceFlux(surface=1, field="solute")
right_flux = F.SurfaceFlux(surface=2, field="solute")
my_model.exports = [
    F.DerivedQuantities([left_flux, right_flux]),
    F.XDMFExport(field="solute", checkpoint=False),
]

my_model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-10,
    final_time=21000,
)

my_model.dt = F.Stepsize(initial_value=1, stepsize_change_ratio=1.01)

def max_stepsize(t):
    if t < 2500:
        return 1
    else:
        return 10
my_model.max_stepsize = max_stepsize

my_model.initialise()
my_model.run()
```

## Comparison with experimental data

The results produced by FESTIM are in good agreement with the experimental data.

```{code-cell} ipython3
:tags: [hide-input]

experimental_data = np.genfromtxt("plasma-driven-permeation-experiment-original.csv", delimiter=",")
experimental_t = experimental_data[:, 0]
experimental_flux = experimental_data[:, 1]

time = left_flux.t
flux_right_values = np.array(right_flux.data)

plt.plot(time, -flux_right_values, label="FESTIM")
plt.fill_between(time, -flux_right_values, alpha=0.3)
plt.scatter(experimental_t, experimental_flux, label="Experiment (original)", s=10)

plt.ylabel("Flux (D/m2/s)")
plt.xlabel("Time (s)")
plt.ylim(0, 3.5e17)
plt.legend(reverse=True)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.grid(alpha=0.3)
plt.show()
```

```{note}
The experimental data was taken from Figure 3 of the original experiment paper {cite}`anderl_tritium_1986` using [WebPlotDigitizer](https://automeris.io/)
```

```{warning}
There is a inconsistence between the TMAP4 and TMAP7 V&V reports. Both reference the same experimental paper, but TMAP4 gives it as $\mathrm{D \ m^{-2} \ s^{-1}}$ whereas TMAP7 gives it as $\mathrm{D \ s^{-1}}$.
This means the authors of the TMAP7 report must have multiplied the experimental curve by the surface. However, when computing the surface of a 20-mm diameter disk and multiplying the experimental data, we are not able to reproduce the curve of the TMAP7 report.
Moreoever, the TMAP4 report says in the text the implantation depth is $11 \ \mathrm{\mu m}$ when it should be $11 \ \mathrm{nm}$
```

```{code-cell} ipython3
:tags: [hide-input]

experimental_data_TMAP7 = np.genfromtxt("plasma-driven-permeation-experiment-tmap7.csv", delimiter=",")
experimental_t_TMAP7 = experimental_data_TMAP7[:, 0]
experimental_flux_TMAP7 = experimental_data_TMAP7[:, 1]

experimental_data_original = np.genfromtxt("plasma-driven-permeation-experiment-original.csv", delimiter=",")
experimental_t_original = experimental_data_original[:, 0]
experimental_flux_original = experimental_data_original[:, 1]


diameter = 20e-3 # m
surface = np.pi * (diameter/2)**2

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8))


axs[0].plot(experimental_t_original, experimental_flux_original, label="original")
axs[0].set_ylabel("Flux (D/m2/s)")
axs[0].legend()
axs[0].set_ylim(bottom=0)

axs[1].plot(experimental_t_TMAP7, experimental_flux_TMAP7, label="TMAP7")
axs[1].plot(experimental_t_original, experimental_flux_original * surface, label=r"original $\times$ surface")
axs[1].set_ylabel("Flux (D/s)")
axs[1].legend()
axs[1].set_yscale("log")
axs[1].set_xlabel("Time (s)")
axs[1].annotate(f"Surface: {surface:.2e} m2", xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center")

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
plt.show()
```
