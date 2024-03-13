# Introduction

In the realm of computational modeling and simulation, the Method of Exact Solutions (MES) and the Method of Manufactured Solutions (MMS) are two distinct methodologies employed for verification purposes. While both aim to assess the accuracy and correctness of numerical codes, they differ significantly in their approach and application.

## Method of Exact Solutions (MES)

The Method of Exact Solutions (MES) is a classical approach to code verification that involves solving mathematical equations analytically to obtain exact solutions. These analytical solutions serve as benchmarks against which the numerical solutions produced by the code under scrutiny are compared. MES is particularly well-suited for simple mathematical models with known analytical solutions, such as linear equations or idealized boundary value problems.

In MES, the governing equations of the problem are solved symbolically, typically by hand or using mathematical software. The resulting exact solutions provide a reference standard for assessing the accuracy and convergence of numerical algorithms implemented in the code. By comparing the numerical solutions to the exact solutions, developers can identify errors, inconsistencies, or numerical artifacts that may arise during the computational process.

## Method of Manufactured Solutions (MMS)

In contrast to MES, the Method of Manufactured Solutions (MMS) is a more versatile and widely applicable approach to code verification. MMS involves the deliberate construction of artificial solutions, known as manufactured solutions, that satisfy the governing equations of the problem. These manufactured solutions are designed to possess specific mathematical properties, such as smoothness, nonlinearity, or complexity.

The key principle behind MMS is to create manufactured solutions that are exact solutions of the governing equations. These solutions are carefully crafted to challenge the numerical solver by incorporating mathematical properties that may lead to computational difficulties, such as steep gradients, discontinuities, or highly nonlinear behavior.

By comparing the numerical solutions obtained from the code against the manufactured solutions, developers can assess the code's ability to accurately capture the underlying physics of the problem and handle various boundary conditions and computational challenges in non-trivial geometries.

## Differentiating MES and MMS

While both MES and MMS serve the overarching goal of code verification, they differ in their implementation and scope. MES relies on exact analytical solutions derived from simplified mathematical models, making it suitable for verifying basic numerical algorithms and validating code implementations in idealized scenarios. On the other hand, MMS offers a more flexible and comprehensive approach, allowing developers to assess the performance of numerical codes across a wide range of problem complexities and boundary conditions.

In summary, MES and MMS represent complementary methodologies for code verification, each offering unique advantages and insights into the accuracy and reliability of numerical simulations. By incorporating both MES and MMS into the verification process, developers can gain a comprehensive understanding of the strengths and limitations of their numerical codes, ultimately enhancing their confidence in the predictive capabilities of computational models.

