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

# Kinetic surface model

```{tags} 1D, MMS, kinetic surface model, transient
```

This MMS case verifies the implementation of the `SurfaceKinetics` boundary condition in FESTIM. We will consider a time-dependent case of hydrogen diffusion on domain $\Omega: x\in[0,1] \cup t\in[0, 5]$ with a homogeneous diffusion coefficient $D$, and a Dirichlet boundary condition on the rear domain side.

+++

The problem is:

\begin{align}
    &\dfrac{\partial c_\mathrm{m}}{\partial t} = \nabla\cdot\left(D\nabla c_\mathrm{m} \right) + S \quad \textrm{ on } \Omega, \\
    &-D \nabla c_\mathrm{m} \cdot \mathbf{n} = \lambda_{\mathrm{IS}} \dfrac{\partial c_{\mathrm{m}}}{\partial t} + J_{\mathrm{bs}} - J_{\mathrm{sb}} \quad \textrm{ at } x=0, \\
    &c_\mathrm{m} = c_\mathrm{m, 0} \quad \textrm{ at } x=1, \\
    &c_\mathrm{m} = c_\mathrm{m, 0} \quad \textrm{ at } t=0, \\
    &\dfrac{d c_\mathrm{s}}{d t} = J_{\mathrm{bs}} - J_{\mathrm{sb}} + J_{\mathrm{vs}}  \quad \textrm{ at } x=0, \\
    &c_\mathrm{s}= c_\mathrm{s, 0}\quad \textrm{ at } t=0, \\
\end{align}

with $J_{\mathrm{bs}} = k_{\mathrm{bs}} c_{\mathrm{m}} \lambda_{\mathrm{abs}} \left(1 - \dfrac{c_\mathrm{s}}{n_{\mathrm{surf}}}\right)$, $J_{\mathrm{sb}} = k_{\mathrm{sb}} c_{\mathrm{s}} \left(1 - \dfrac{c_{\mathrm{m}}}{n_\mathrm{IS}}\right)$, $\lambda_{\mathrm{abs}}=n_\mathrm{surf}/n_\mathrm{IS}$.

The manufactured exact solution for mobile concentration is:
\begin{equation}
c_\mathrm{m, exact}=1+2x^2+x+2t.
\end{equation}

For this problem, we choose:
\begin{align*}
& k_{\mathrm{bs}}=1/\lambda_{\mathrm{abs}} \\
& k_{\mathrm{sb}}=2/\lambda_{\mathrm{abs}} \\
& n_{\mathrm{IS}} = 20 \\
& n_{\mathrm{surf}} = 5 \\
& D = 5 \\
& \lambda_\mathrm{IS} = 2
\end{align*}

Injecting these parameters and the exact solution for solute H, we obtain:

\begin{align}
& S = 2(1-2D) \\
& J_{\mathrm{vs}}=2n_\mathrm{surf}\dfrac{2n_\mathrm{IS}+2\lambda_\mathrm{IS}-D}{(2n_\mathrm{IS}-1-2t)^2}+2\lambda_\mathrm{IS}-D \\
& c_\mathrm{s, exact}=n_\mathrm{surf}\dfrac{1+2t+2\lambda_\mathrm{IS}-D}{2n_\mathrm{IS}-1-2t} \\
& c_\mathrm{s,0}=c_\mathrm{s, exact} \\
& c_\mathrm{m,0}=c_\mathrm{m, exact}
\end{align}

We can then run a FESTIM model with these values and compare the numerical solutions with $c_\mathrm{m, exact}$ and $c_\mathrm{s, exact}$.

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import festim as F
import matplotlib.pyplot as plt
import numpy as np

n_IS = 20
n_surf = 5
D = 5
lambda_IS = 2
k_bs = n_IS / n_surf
k_sb = 2 * n_IS / n_surf

solute_source = 2 * (1 - 2 * D)

exact_solution_cm = lambda x, t: 1 + 2 * x**2 + x + 2 * t
exact_solution_cs = (
    lambda t: n_surf * (1 + 2 * t + 2 * lambda_IS - D) / (2 * n_IS - 1 - 2 * t)
)

solute_source = 2 * (1 - 2 * D)


def run_sim(export_times=None):
    # Create the FESTIM model
    my_model = F.Simulation()

    my_model.mesh = F.MeshFromVertices(np.linspace(0, 1, 1000))

    my_model.sources = [F.Source(solute_source, volume=1, field="solute")]

    def J_vs(T, surf_conc, t):
        return (
            2 * n_surf * (2 * n_IS + 2 * lambda_IS - D) / (2 * n_IS - 1 - 2 * t) ** 2
            + 2 * lambda_IS
            - D
        )

    my_model.boundary_conditions = [
        F.DirichletBC(
            surfaces=[2], value=exact_solution_cm(x=F.x, t=F.t), field="solute"
        ),
        F.SurfaceKinetics(
            k_sb=k_sb,
            k_bs=k_bs,
            lambda_IS=lambda_IS,
            n_surf=n_surf,
            n_IS=n_IS,
            J_vs=J_vs,
            surfaces=1,
            initial_condition=exact_solution_cs(t=0),
            t=F.t,
        ),
    ]

    my_model.initial_conditions = [
        F.InitialCondition(field="solute", value=exact_solution_cm(x=F.x, t=F.t))
    ]

    my_model.materials = F.Material(id=1, D_0=D, E_D=0)

    my_model.T = 300  # this is ignored since no parameter is T-dependent

    my_model.settings = F.Settings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, transient=True, final_time=5
    )

    my_model.dt = F.Stepsize(initial_value=5e-3, milestones=export_times)

    derived_quantities = F.DerivedQuantities(
        [F.AdsorbedHydrogen(surface=1)], show_units=True
    )
    my_model.exports = [
        F.TXTExport("solute", filename="./mobile_conc.txt", times=export_times),
        derived_quantities,
    ]

    my_model.initialise()
    my_model.run()
    return derived_quantities
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input, hide-output]

def norm(x, c_comp, c_ex):
    return np.sqrt(np.trapz(y=(c_comp - c_ex) ** 2, x=x))


export_times = [1, 2, 3, 4, 5]

adsorbed_data = run_sim(export_times)
solute_data = np.genfromtxt("mobile_conc.txt", names=True, delimiter=",")
```

### Mobile H

```{code-cell} ipython3
:tags: [hide-input]

plot_times = [1, 3, 5]

for t in plot_times:
    x = solute_data["x"]
    y = solute_data[f"t{t:.2e}s".replace(".", "").replace("+", "")]
    # order y by x
    x, y = zip(*sorted(zip(x, y)))

    (l1,) = plt.plot(
        x,
        exact_solution_cm(np.array(x), t),
        label=f"exact t = {t}",
    )
    plt.scatter(
        x[::100],
        y[::100],
        label=f"t = {t}",
        color=l1.get_color(),
        alpha=0.6,
    )

    print(
        f"L2 error for mobile H at t = {t}: {norm(np.array(x), np.array(y), exact_solution_cm(x=np.array(x), t=t))}"
    )

plt.legend(reverse=True, ncols=3, loc=9, frameon=False)
plt.ylabel("$c_m$")
plt.xlabel("$x$")
plt.ylim(2, 16)
plt.show()
```

### Adsorbed H

```{code-cell} ipython3
:tags: [hide-input]

c_s_computed = adsorbed_data[0].data
t = adsorbed_data[0].t

print(
    f"L2 error for adsorbed H: {norm(t, c_s_computed, exact_solution_cs(t=np.array(t)))}"
)

plt.figure()

plt.scatter(t[::50], c_s_computed[::50], label="computed", alpha=0.6)
plt.plot(t, exact_solution_cs(np.array(t)), label="exact")
plt.ylabel("$c_s$")
plt.xlabel("$t$")
plt.legend()
plt.ylim(-0.05, 1.8)
plt.show()
```
