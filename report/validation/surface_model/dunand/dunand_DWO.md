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

# Deuterium adsorption on oxidised tungsten

```{tags} 1D, kinetic surface model, TDS, transient
```

+++

This validation case reproduces TDS measurements on oxidised W performed by Dunand et al. {cite}`dunand_2022`.

In the experiments, single crystal W samples (2 mm thick) with different O coverages (clean, 0.5 ML of O, 0.75 ML of O) were exposed to the D<sub>2</sub> flux of $\approx 1.52\times10^{18}\,\mathrm{m}^{-2}\mathrm{s}^{-1}$. D<sub>2</sub> exposure lasted for 3000 s followed by the storage phase for 1 h. After that, TDS of samples was performed with 5 K/s ramp up to 800 K.

Following the approach of Hodille et al. {cite}`hodille_2024`, the evolution of the surface concentration, in the present FESTIM model, is assumed to be governed by D adsorption from the gas phase and desorption. The D diffusivity in W is defined by scaling the corresponding value for H (Fernandez et al. {cite}`fernandez_2015`) by a factor of $1/\sqrt{2}$. Traps are not considered, since the D absorption flux is negligible at room temperature. 

The FESTIM results are compared to the experimental data and the results of MHIMS simulation, both taken from {cite}`hodille_2024`.

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
T0 = 300  # K
ramp = 5  # K/s
t_imp = 3000  # exposure duration, s
t_storage = 3600  # storage time, s
t_TDS = 100  # s
L = 2e-3  # half thickness, m

P_D2 = 2e-5  # D2 pressure, Pa
m_D2 = 4.0282035557e-3 / 6.022e23  # D2 molecule mass, kg
molecular_flux = P_D2 / np.sqrt(
    2 * np.pi * m_D2 * T0 * 1.380649e-23
)  # flux of atoms, m^-2 s^-1

# W properties
rho_W = 6.3382e28  # W atomic concentration, m^-3
n_IS = 6 * rho_W  # concentration of interstitial sites, m^-3
lambda_IS = 1.117e-10  # distance between interstitial sites, m
n_surf_ref = 1.416e19  # concentration of adsorption sites, m^-2
nu0 = 1e13  # attempt frequency, s^-1

D_H = htm.diffusivities.filter(material=htm.Tungsten, author="fernandez")[0]
D0 = D_H.pre_exp.magnitude / np.sqrt(2)  # diffusivity pre-factor, m^2 s^-1
E_diff = D_H.act_energy.magnitude  # diffusion activation energy, eV

# Transitions
E_bs = E_diff  # energy barrier from bulk to surface, eV
E_diss = 0  # energy barrier for D2 dissociation, eV
Q_sol = 1  # heat of solution, eV


################### FUNCTIONS ###################
def S0(T):
    # the capturing coefficient
    return f.exp(-E_diss / F.k_B / T)


def K_bs(T, surf_conc, t):
    return nu0 * f.exp(-E_bs / F.k_B / T)


def run_sim(n_surf, E0, dE, theta_D0, dtheta_D, alpha, beta):
    lamda_des = 1 / np.sqrt(n_surf)  # average distance between adsorption sites, m

    def E_des(surf_conc):
        theta_D = surf_conc / n_surf
        E_FD = E0 + dE / (1 + f.exp((theta_D - theta_D0) / dtheta_D))
        E_des = E_FD * (1 - alpha * f.exp(-beta * (1 - theta_D)))
        return E_des

    def E_sb(surf_conc):
        # energy barrier from surface to bulk, eV
        return (E_des(surf_conc) - E_diss) / 2 + E_bs + Q_sol

    def K_sb(T, surf_conc, t):
        return nu0 * f.exp(-E_sb(surf_conc) / F.k_B / T)

    def J_vs(T, surf_conc, t):
        J_ads = (
            2
            * S0(T)
            * (1 - surf_conc / n_surf) ** 2
            * f.conditional(t <= t_imp, molecular_flux, 0)
        )

        J_des = (
            2
            * nu0
            * (lamda_des * surf_conc) ** 2
            * f.exp(-E_des(surf_conc) / F.k_B / T)
        )
        return J_ads - J_des

    W_model = F.Simulation(log_level=40)

    # Mesh
    vertices = np.linspace(0, L, num=500)
    W_model.mesh = F.MeshFromVertices(vertices)

    # Materials
    tungsten = F.Material(id=1, D_0=D0, E_D=E_diff)
    W_model.materials = tungsten

    W_model.T = F.Temperature(
        value=sp.Piecewise(
            (T0, F.t < t_imp + t_storage), (T0 + ramp * (F.t - t_imp - t_storage), True)
        )
    )

    my_BC = F.SurfaceKinetics(
        k_sb=K_sb,
        k_bs=K_bs,
        lambda_IS=lambda_IS,
        n_surf=n_surf,
        n_IS=n_IS,
        J_vs=J_vs,
        surfaces=1,
        initial_condition=0,
        t=F.t,
    )

    W_model.boundary_conditions = [my_BC, F.DirichletBC(field=0, value=0, surfaces=2)]

    W_model.dt = F.Stepsize(
        initial_value=1e-7,
        stepsize_change_ratio=1.25,
        max_stepsize=lambda t: 10 if t < t_imp + t_storage - 10 else 0.1,
        dt_min=1e-9,
    )

    W_model.settings = F.Settings(
        absolute_tolerance=1e5,
        relative_tolerance=1e-11,
        maximum_iterations=50,
        final_time=t_imp + t_storage + t_TDS,
    )

    # Exports
    derived_quantities = F.DerivedQuantities(
        [F.AdsorbedHydrogen(surface=1), F.TotalSurface(field="T", surface=1)],
        show_units=True,
    )

    W_model.exports = [derived_quantities]

    W_model.initialise()
    W_model.run()

    return derived_quantities


# Fitting parameters from the paper
cases = ["clean", "0.50ML of O", "0.75ML of O"]
inputs = {
    "n_surf": [n_surf_ref, n_surf_ref * (1 - 0.5), n_surf_ref * (1 - 0.75)],
    "E0": [1.142, 1.111, 1.066],
    "dE": [0.346, 0.289, 0.234],
    "theta_D0": [0.253, 0.113, 0.161],
    "dtheta_D": [0.180, 0.082, 0.057],
    "alpha": [0.303, 0.460, 0.437],
    "beta": [8.902, 7.240, 4.144],
}

results = {}
for i, case in enumerate(cases):
    results[case] = run_sim(*[inputs[key][i] for key in inputs.keys()])
```

