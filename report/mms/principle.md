# The Method of Manufactured Solutions

The Method of Manufactured Solutions (MMS) is a mathematical technique used to verify numerical simulation codes \cite{roy_exact_2010}.
In MMS, an exact analytical solution is intentionally _manufactured_.
The mathematical model is then operated on this manufactured solution to obtain an analytic source term.
Computer symbolic manipulation can be used to obtain the derivatives of the manufactured solution.
Boundary conditions obtained directly from the manufactured solution and the source term then serve as input for the numerical simulation code.

The error between the code output and the manufactured analytical solution is computed and used to assess the code's correctness and accuracy.
MMS is a valuable tool for code verification in scientific and engineering computing, ensuring the reliability of computational results.

This section covers the verification of interface discontinuities, the heat transfer module, and the trapping implementation in FESTIM.
For simplicity and clarity, only three distinct verification problems have been defined.
All MMS cases in this section are applied on a unit square (for easier visualisation), and the mesh has 100 cells in the $x$ direction and 100 cells in the $y$ direction.
Dirichlet boundary conditions enforcing the exact solution are applied on all surfaces.
Only steady-state problems for simplicity's sake, but MMS can be - and has been - applied to transient problems.
For each case, the L2 error $E$ is calculated in the domain $\Omega$:

\begin{equation}
    E = \sqrt{\int_\Omega(u_\mathrm{exact} - u_\mathrm{computed})^2 dx}
\end{equation}
where $u_\mathrm{exact}$ and $u_\mathrm{computed}$ are the exact and computed solutions, respectively.