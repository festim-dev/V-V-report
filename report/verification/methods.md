# Methods

In the realm of computational modeling and simulation, the Method of Exact Solutions (MES) and the Method of Manufactured Solutions (MMS) are two distinct methodologies employed for verification purposes. While both aim to assess the accuracy and correctness of numerical codes, they differ significantly in their approach and application.

## Method of Exact Solutions (MES)

The Method of Exact Solutions (MES) is a classical approach to code verification that involves solving mathematical equations analytically to obtain exact solutions. These analytical solutions serve as benchmarks against which the numerical solutions produced by the code under scrutiny are compared. MES is particularly well-suited for simple mathematical models with known analytical solutions, such as linear equations or idealized boundary value problems.

In MES, the governing equations of the problem are solved symbolically, typically by hand or using mathematical software. The resulting exact solutions provide a reference standard for assessing the accuracy and convergence of numerical algorithms implemented in the code. By comparing the numerical solutions to the exact solutions, developers can identify errors, inconsistencies, or numerical artifacts that may arise during the computational process.

## Method of Manufactured Solutions (MMS)

The Method of Manufactured Solutions (MMS) is a mathematical technique used to verify numerical simulation codes {cite}`roy_exact_2010`.
In MMS, an exact analytical solution is intentionally _manufactured_.
The mathematical model is then operated on this manufactured solution to obtain an analytic source term.
Computer symbolic manipulation can be used to obtain the derivatives of the manufactured solution.
Boundary conditions obtained directly from the manufactured solution and the source term then serve as input for the numerical simulation code.

The error between the code output and the manufactured analytical solution is computed and used to assess the code's correctness and accuracy.
MMS is a valuable tool for code verification in scientific and engineering computing, ensuring the reliability of computational results.

For each case, the L2 error $E$ is calculated in the domain $\Omega$:

\begin{equation}
    E = \sqrt{\int_\Omega(u_\mathrm{exact} - u_\mathrm{computed})^2 dx}
\end{equation}
where $u_\mathrm{exact}$ and $u_\mathrm{computed}$ are the exact and computed solutions, respectively.

A [detailed example](mms/simple.md) is available in this book.

## Differentiating MES and MMS

While both MES and MMS serve the overarching goal of code verification, they differ in their implementation and scope. MES relies on exact analytical solutions derived from simplified mathematical models, making it suitable for verifying basic numerical algorithms and validating code implementations in idealized scenarios. On the other hand, MMS offers a more flexible and comprehensive approach, allowing developers to assess the performance of numerical codes across a wide range of problem complexities and boundary conditions.

In summary, MES and MMS represent complementary methodologies for code verification, each offering unique advantages and insights into the accuracy and reliability of numerical simulations. By incorporating both MES and MMS into the verification process, developers can gain a comprehensive understanding of the strengths and limitations of their numerical codes, ultimately enhancing their confidence in the predictive capabilities of computational models.

