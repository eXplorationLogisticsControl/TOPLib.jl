# TOPLib.jl
Trajectory Optimization via NLP transcriptions

## Multiple shooting

We consider the problem

$$
\begin{aligned}
    \min_{\boldsymbol{x}_k,\boldsymbol{u}_k} \quad& J(\boldsymbol{x}_k,\boldsymbol{u}_k)
    \\ \text{s.t.} \quad&
    x_{k+1} = \boldsymbol{x}_k + \int_{t_k}^{t_{k+1}} f(\boldsymbol{x}_k,\boldsymbol{u}_k) \mathrm{d}t \quad& k=1,\ldots,N-1
    \\&
    \boldsymbol{g}(\boldsymbol{x}_k,\boldsymbol{u}_k) = \boldsymbol{0}
    \\&
    \boldsymbol{h}(\boldsymbol{x}_k,\boldsymbol{u}_k) \leq \boldsymbol{0}
\end{aligned}
$$


### Minimum fuel

For minimum fuel, we opt for the 4-parameter representation, $\boldsymbol{u} = u \times [u_x,u_y,u_z]$, with ${u_x^2 + u_y^2 + u_z^2} \leq 1$ and $u \in [0,u_{\max}]$.
Then, the objective is $J(\boldsymbol{x}_k,\boldsymbol{u}_k) = J(\boldsymbol{u}_k) = sum_{k=1}^{N-1} u_k$.