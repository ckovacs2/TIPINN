# Topologically-Infused Physically-Informed Neural Networks (TIPINN)
Topologically-Infused Physically-Informed Neural Networks (TIPINN) is a new form of PINN that embeds topological loss as its penalty rather than the function itself.

# What are PINNs?
[PINNs](https://ieeexplore.ieee.org/document/712178) are neural networks that are typically trained to provide solutions to ODEs and PDEs using a physically informed penalization additional to the already prescribed loss in the model. Most Neural Network methods cannot extrapolate well from these curves, but PINNs offer a new way to predict further on using this loss.

# Topological Loss
The topological loss is based on the concept of sub-level persistence. The concept of sub-level graphs is given in set notation as $$P(y) = \{x \in A:\, f(x) \leq y\} $$ where $A$ is the set we are working in and $y$ is a restrictive parameter. This allows you to essentially construct a "graph" of local maximums and minima through a raised horizontal line that looks for intersections to the given function. This horizontal line is raised given some scale called $\varepsilon > 0$ and is increased until there is no more local extremum.

Since PINNs require you to find the solution to the differential equation for the informed loss, this will become a problem if we cannot find the solution or at least an approximation to the problem. Furthermore, it has been [shown](https://proceedings.neurips.cc/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf) that PINNs have trouble with specific differential equations with different parameters. Here, the topology of the curve can play an important role in characterizing the structure and aiding it in extrapolation.

In general, we are looking to generally minimize the following function $$\min_\theta L(u) + \lambda_{F}F(u),\quad L(u) = L_{u_0} + L_{u_b}$$ where the problem is the general differential equation $$F(u(x,t)) = 0,\quad x\in \Omega\subset\mathbb{R}^d,\quad t\in [0, T].$$
