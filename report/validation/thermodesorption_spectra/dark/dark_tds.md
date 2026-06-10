---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: vv-festim-report-env-festim-2
  language: python
  name: python3
---

# Deuterium retention in self-damaged tungsten

```{tags} 1D, TDS, trapping, transient
```

+++

This validation case is a thermo-desorption spectrum measurement on damaged tungsten. The complete description is available in {cite}`dark_modelling_2024`.

Several 0.8 mm thick samples of tungsten were self-damaged and annealing before being used to perform a TDS measurement.

The diffusivity of tungsten in the FESTIM model is as measured by Holzner et al. {cite}`holzner_2020`.

An ion beam with an incident flux of $5.79 \times 10^{19} \ \mathrm{D \ m^{-2} \ s^{-1}}$ was turned on for $72 \ \mathrm{h}$ with an implantation temperature of $370 \ \mathrm{K}$. The sample then rested for $12 \ \mathrm{h}$ at $295 \ \mathrm{K}$ before beginning the TDS measurement at $300 \ \mathrm{K}$ with a temperature ramp of $0.05 \ \mathrm{K}/s$.

To reproduce this experiment, six traps are needed: 1 intrinsic trap and 5 neutron induced traps.
The trap densities for the neutron induced traps were fitted by {cite}`dark_modelling_2024` for each _dpa_ dose.

The damage distribution for the damage-induced traps is as follows:

$$
    f(x) = \frac{1}{1 + \exp{ \frac{\left( x - x_0 \right)}{\Delta x} }}
$$

The density distribution of the neutron-induced traps is $n_i \ f(x)$.

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import festim as F

import ufl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Two FESTIM 2.0 monkey-patches are needed to reproduce the FESTIM 1.0 results
# of this case (long high-flux implantation drives deep trap saturation):
#
#   1. ProblemBase.iterate hard-asserts on SNES divergence in 2.0; FESTIM 1.0
#      cuts dt back and retries. We restore that behaviour.
#
#   2. ImplicitSpecies.concentration computes empty = n - trapped with no
#      clamping; once a trap saturates and the iterate overshoots into c_t > n,
#      the trapping-reaction Jacobian becomes singular and Newton diverges no
#      matter how small dt is. We replace the expression by a smooth max(., 0).
# ---------------------------------------------------------------------------

_DT_MIN = 1e-6
_MAX_RETRIES = 30


def _iterate_with_cutback(self):
    self._timesteps.append(float(self.t))
    if self.show_progress_bar:
        self.progress_bar.update(
            min(self.dt.value, abs(self.settings.final_time - self.t.value))
        )
    t_backup = float(self.t.value)
    u_backup = self.u.x.array.copy()
    u_n_backup = self.u_n.x.array.copy()
    nb_its = None
    for attempt in range(_MAX_RETRIES + 1):
        self.t.value = t_backup + float(self.dt.value)
        self.update_time_dependent_values()
        try:
            _ = self.solver.solve()
            reason = self.solver.solver.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(f"SNES did not converge (reason {reason})")
            nb_its = self.solver.solver.getIterationNumber()
            break
        except BaseException:
            cur = float(self.dt.value)
            if attempt == _MAX_RETRIES or cur <= _DT_MIN:
                raise
            self.u.x.array[:] = u_backup
            self.u_n.x.array[:] = u_n_backup
            self.t.value = t_backup
            self.dt.value = max(cur * 0.5, _DT_MIN)
    self.post_processing()
    self.u_n.x.array[:] = self.u.x.array[:]
    if self.settings.stepsize.adaptive:
        self.dt.value = self.settings.stepsize.modify_value(
            value=self.dt.value, nb_iterations=nb_its, t=self.t.value
        )


F.ProblemBase.iterate = _iterate_with_cutback


_CLAMP_EPS = 1e6  # smoothing scale in atoms/m^3


def _clamped_concentration(self):
    raw = self.value_fenics - sum(o.solution for o in self.others)
    return 0.5 * (raw + ufl.sqrt(raw * raw + _CLAMP_EPS * _CLAMP_EPS))


F.ImplicitSpecies.concentration = property(_clamped_concentration)

# # ### Parameters ###
D_0 = 1.6e-7  # m^2 s^-1
E_D = 0.28  # eV
w_atom_density = 6.3222e28
sample_thickness = 0.8e-3  # m
sample_area = 12e-03 * 15e-03

detrapping_energies = [1.15, 1.35, 1.65, 1.85, 2.05]
dpa_n_i = {
    0: [],
    0.001: [1e24, 2.5e24, 1e24, 1e24, 2e23],
    0.005: [3.5e24, 5e24, 2.5e24, 1.9e24, 1.6e24],
    0.023: [2.2e25, 1.5e25, 6.5e24, 2.1e25, 6e24],
    0.1: [4.8e25, 3.8e25, 2.6e25, 3.6e25, 1.1e25],
    0.23: [5.4e25, 4.4e25, 3.6e25, 3.9e25, 1.4e25],
    0.5: [5.5e25, 4.6e25, 4e25, 4.5e25, 1.7e25],
    2.5: [5.8e25, 6.5e25, 4.5e25, 5.5e25, 2e25],  # re-fit
}

# Table 2 from Dark et al 10.1088/1741-4326/ad56a0
T_imp = 370  # K
T_rest = 295  # K
R_p = 0.7e-9  # m
sigma = 0.5e-9  # m
t_imp = 72 * 3600  # s
implantation_time = t_imp
t_rest = 12 * 3600  # s
resting_time = t_rest
Beta = 3 / 60  # K s^-1
fluence = 1.5e25
flux = fluence / t_imp
start_tds = t_imp + t_rest  # s
min_temp, max_temp = 300, 1000


center = 0.7e-9
width = 0.5e-9


def festim_sim(densities):

    # snes_error_if_not_converged=False lets the cutback patch react to
    # non-convergence by halving dt instead of aborting the run.
    model = F.HydrogenTransportProblem(
        petsc_options={
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
        }
    )
    vertices = np.concatenate(
        [
            # 200 verts per segment (was 100 in FESTIM 1.0): with the clamped
            # trap formulation, 100 under-resolves the steep damage_dist
            # sigmoid at x = 2.5e-6 m and the deeper traps over-fill by 2-5x,
            # producing spurious spikes in the TDS spectrum.
            np.linspace(0, 3e-9, num=200),
            np.linspace(3e-9, 8e-6, num=200),
            np.linspace(8e-6, 8e-5, num=200),
            np.linspace(8e-5, sample_thickness, num=200),
        ]
    )
    model.mesh = F.Mesh1D(vertices)
    # ### Material ###
    damaged_tungsten = F.Material(D_0, E_D)

    volume = F.VolumeSubdomain1D(
        id=1, borders=[0, sample_thickness], material=damaged_tungsten
    )
    left_boundary = F.SurfaceSubdomain1D(id=1, x=0)
    right_boundary = F.SurfaceSubdomain1D(id=2, x=sample_thickness)
    model.subdomains = [volume, left_boundary, right_boundary]

    H = F.Species("H")
    instrinsic_trapped_H = F.Species(f"Trapped 1", mobile=False)
    neutron_induced_trapped_species = [
        F.Species(f"Trapped D{i+1}", mobile=False) for i in range(len(densities))
    ]

    damage_dist = lambda x: 1 / (1 + ufl.exp((x[0] - 2.5e-06) / 5e-07))
    empty_neutron_induced_traps = []
    for i, (density, trapped_spe) in enumerate(
        zip(densities, neutron_induced_trapped_species)
    ):
        empty_neutron_induced_traps.append(
            F.ImplicitSpecies(
                n=lambda x, density=density: density * damage_dist(x),
                others=[trapped_spe],
                name=f"empty {i+2}",
            )
        )
        

    empty_intrinsic_traps = F.ImplicitSpecies(
        n=2.4e22, others=[instrinsic_trapped_H], name="empty 1"
    )

    assert len([empty_intrinsic_traps] + empty_neutron_induced_traps) == len(
        neutron_induced_trapped_species + [instrinsic_trapped_H]
    )

    model.species = [H] + [instrinsic_trapped_H] + neutron_induced_trapped_species

    # ### Source ###
    # Deuterium Beam Profile (S = flux * f(x))
    distribution = lambda x: (
        1 / (sigma * (2 * ufl.pi) ** 0.5) * ufl.exp(-0.5 * ((x[0] - R_p) / sigma) ** 2)
    )
    ion_flux = lambda x, t: ufl.conditional(t < t_imp, flux * distribution(x), 0)
    source_term = F.ParticleSource(value=ion_flux, volume=volume, species=H)
    model.sources = [source_term]
    # ### Boundary Conditions ###
    model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left_boundary, value=0, species=H),
        F.FixedConcentrationBC(subdomain=right_boundary, value=0, species=H),
    ]
    # ### Temperature ###

    def temp_fun(t):
        if t < t_imp:
            return T_imp
        elif t < start_tds:
            return T_rest
        else:
            return min_temp + Beta * (t - start_tds)

    model.temperature = temp_fun

    # ### Trap Settings ###
    k_0 = D_0 / (1.1e-10**2 * 6 * w_atom_density)

    trapping_reaction_1 = F.Reaction(
        reactant=[H, empty_intrinsic_traps],
        product=[instrinsic_trapped_H],
        k_0=k_0,
        E_k=damaged_tungsten.E_D,
        p_0=1e13,
        E_p=1.04,
        volume=volume,
    )

    reactions = [trapping_reaction_1]

    for i, _ in enumerate(densities):
        reactions.append(
            F.Reaction(
                reactant=[H, empty_neutron_induced_traps[i]],
                product=[neutron_induced_trapped_species[i]],
                k_0=k_0,
                E_k=damaged_tungsten.E_D,
                p_0=1e13,
                E_p=detrapping_energies[i],
                volume=volume,
            )
        )
    model.reactions = reactions

    model.settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        max_iterations=50,  # default 30 is too low under the clamp nonlinearity
        final_time=start_tds + (max_temp - min_temp) / Beta,  # time to reach max temp
    )
    # target_nb_iterations is set very high so dt grows geometrically to
    # max_stepsize regardless of the (always high) iteration count produced
    # by the clamped trapping nonlinearity; cutback (patched above) is what
    # actually protects against a step that doesn't converge.
    model.settings.stepsize = F.Stepsize(
        initial_value=2,
        growth_factor=1.1,
        cutback_factor=0.9,
        target_nb_iterations=1000,
        # cap dt during implantation: unbounded dt fights badly with the
        # clamped trapping nonlinearity and forces many Newton cutbacks.
        max_stepsize=lambda t: 50 if t > t_imp + t_rest * 0.5 else 200,
    )
    derived_quantities = [
        F.TotalVolume(field=spe, volume=volume) for spe in model.species[1:]
    ]

    flux_left = F.SurfaceFlux(field=H, surface=left_boundary)
    flux_right = F.SurfaceFlux(field=H, surface=right_boundary)
    derived_quantities.append(flux_left)
    derived_quantities.append(flux_right)

    vtx_exports = [
        # F.VTXSpeciesExport(filename=spe.name, field=spe) for spe in model.species
    ]

    model.exports = vtx_exports + derived_quantities

    # import dolfinx

    # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    model.initialise()
    # model.solver.convergence_criterion = "incremental"
    model.run()

    return derived_quantities
```

```{code-cell} ipython3
dpa_to_quantities = {}
for dpa, densities in dpa_n_i.items():
    dpa_to_quantities[dpa] = festim_sim(densities)
```

## Comparison with experimental data

The results produced by FESTIM are in good agreement with the experimental data. The grey areas represent the contribution of each trap to the global TDS spectrum.

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import cm, colors

norm = colors.LogNorm(
    vmin=min(list(dpa_n_i.keys())[1:]), vmax=max(dpa_n_i.keys())
)  # using [1:] indexing to ignore 0
colorbar = cm.viridis
sm = plt.cm.ScalarMappable(cmap=colorbar, norm=norm)


def plot_tds(derived_quantities: list, trap_contributions=False, **kwargs):
    t = np.array(derived_quantities[0].t)
    flux_left = np.array(derived_quantities[-2].data)
    flux_right = np.array(derived_quantities[-1].data)
    flux_total = flux_left + flux_right

    temp = min_temp + Beta * (t - start_tds)

    idx = np.where(t > start_tds)
    plt.plot(temp[idx], flux_total[idx], **kwargs)

    if trap_contributions:
        colors = [(0.9 * (i % 2), 0.2 * (i % 4), 0.4 * (i % 3)) for i in range(6)]
        trap_data = [derived_quantities[i].data for i in range(6)]
        contributions = [
            -np.diff(np.array(trap)[idx]) / np.diff(t[idx]) for trap in trap_data
        ]

        for i, cont in enumerate(contributions):
            label = derived_quantities[i].field.name

            plt.plot(temp[idx][1:], cont, linestyle="--", color=colors[i], label=label)
            plt.fill_between(temp[idx][1:], 0, cont, facecolor="grey", alpha=0.1)


for dpa, derived_quantities in dpa_to_quantities.items():
    filename = f"tds_data/{dpa}_dpa.csv"
    experimental_tds = np.genfromtxt(filename, delimiter=",")
    experimental_temp = experimental_tds[:, 0]
    experimental_flux = experimental_tds[:, 1] / sample_area

    if dpa == 0.1:
        plt.figure(1)
        plt.title(f"Damage = {dpa} dpa")
        plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
        plt.xlabel(r"Temperature (K)")
        plot_tds(
            derived_quantities, linewidth=3, label="FESTIM", trap_contributions=True
        )
        plt.scatter(
            experimental_temp,
            experimental_flux,
            color="black",
            label="experiment",
            s=16,
        )

    plt.figure(2)
    plot_tds(derived_quantities, linestyle="dashed", color="tab:grey", linewidth=2)
    plt.plot(
        experimental_temp,
        experimental_flux,
        color=colorbar(norm(dpa)) if dpa != 0 else "black",
        linewidth=3,
    )
    if dpa == 0:
        max_curve_y = np.max(experimental_flux)
        max_curve_x = experimental_temp[experimental_flux == max_curve_y][0]
        plt.annotate(
            "undamaged",
            xy=(max_curve_x, max_curve_y),  # Point to annotate
            xytext=(300, 0.4e17),  # Location of text
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3", facecolor="black"
            ),  # Arrow properties
        )

for i in [1, 2]:
    plt.figure(i)
    plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
    plt.xlabel(r"Temperature (K)")
    plt.ylim(bottom=0, top=1.2e17)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.figure(1)
plt.legend()

# Plotting color bar
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.figure(2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(sm, cax=cax, label="Damage (dpa)")

plt.show()
```

```{note}
The experimental data was taken from {cite}`dark_modelling_2024_code`.
```

## Trap Parameters

### Damage-induced trap parameters

This table displays the neutron-induced traps' detrapping energy $E_p$ and their density per dpa dose in $m^{-3}$.

```{code-cell} ipython3
:tags: [hide-input]

dpa_no_zero = dpa_n_i | {}
dpa_no_zero.pop(0, None)  # the 0-dpa key is optional
data = {"E_p (eV)": detrapping_energies} | dpa_no_zero
dpa_frame = pd.DataFrame(data)

dpa_frame.columns = dpa_frame.columns.map(
    lambda s: f"{s:.1e} dpa" if not isinstance(s, str) else s
)
dpa_frame.style.relabel_index([f"Trap D{i}" for i in range(1, 6)], axis=0).format(
    "{:.2e}".format
)
```
