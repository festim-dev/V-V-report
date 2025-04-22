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

# Co-permeation




```python
import festim as F
import numpy as np
import matplotlib.pyplot as plt
import h_transport_materials as htm

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


pd_thickness = 0.025e-3  # m
temperature = 870  # K

pd_diffusion_coeff = htm.diffusivities.filter(material=htm.PALLADIUM).mean()
```

```python
upstream_effective_H_pressure = 0.063  # Pa


def pressure_h2(p_H, p_D):
    return p_H**2 / (p_H + p_D)


def pressure_d2(p_H, p_D):
    return p_D**2 / (p_H + p_D)


def pressure_hd(p_H, p_D):
    p_h2 = pressure_h2(p_H, p_D)
    p_d2 = pressure_d2(p_H, p_D)
    return (4 * p_h2 * p_d2) ** 0.5
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

# import dolfinx.log
# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
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

    # ------ Post processsing ------ #

    # convert all data to mol
    for export in my_model.exports:
        avogadro = 6.022e23
        export.data = np.array(export.data) / avogadro

    all_d_desorption_fluxes.append(np.abs(D_flux_right.data)[-1])
    print(
        f"Desorption flux at {effective_d_pressure:.2e} Pa: {all_d_desorption_fluxes[-1]:.2e} molecules/m^2/s"
    )

    hh_desorption_fluxes.append(np.abs(HH_flux.data)[-1])
    hd_desorption_fluxes.append(np.abs(HD_flux.data)[-1])
    dd_desorption_fluxes.append(np.abs(DD_flux.data)[-1])
```

```python
import pandas as pd

# read experimental data
exp_data = pd.read_csv(
    "co_permeation_exp_data.csv",
    names=["H2_X", "H2_Y", "D2_X", "D2_Y", "HD_X", "HD_Y"],
    skiprows=2,
)

from pypalettes import load_cmap

cmap = load_cmap("Acadia")

plt.scatter(exp_data["H2_X"], exp_data["H2_Y"], marker="o", label="H2", color=cmap(0))
plt.scatter(exp_data["D2_X"], exp_data["D2_Y"], marker="^", label="D2", color=cmap(1))
plt.scatter(exp_data["HD_X"], exp_data["HD_Y"], marker="s", label="HD", color=cmap(2))

plt.plot(upstream_d_pressures, hh_desorption_fluxes, label="HH (FESTIM)", color=cmap(0))
plt.plot(upstream_d_pressures, dd_desorption_fluxes, label="DD (FESTIM)", color=cmap(1))
plt.plot(upstream_d_pressures, hd_desorption_fluxes, label="HD (FESTIM)", color=cmap(2))

plt.xlabel("Upstream D pressure (Pa)")
plt.ylabel("Desorption flux (mol/m^2/s)")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-8, 1e-3)
plt.legend()
plt.show()
```

```python
plt.figure()
plt.stackplot(
    H_flux_left.t,
    np.abs(H_flux_left.data),
    np.abs(D_flux_left.data),
    labels=["H_in", "D_in"],
)
plt.stackplot(
    H_flux_right.t,
    -np.abs(H_flux_right.data),
    -np.abs(D_flux_right.data),
    labels=["H_out", "D_out"],
)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Flux (molecule/m^2/s)")
plt.savefig("co_permeation_in_out.png")

plt.figure()
plt.stackplot(
    HD_flux.t,
    np.abs(HH_flux.data),
    np.abs(HD_flux.data),
    np.abs(DD_flux.data),
    labels=["HH", "HD", "DD"],
)
plt.legend(reverse=True)
plt.xlabel("Time (s)")
plt.ylabel("Flux (molecule/m^2/s)")

plt.show()
```
