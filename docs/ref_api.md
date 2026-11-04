# Differentiable simulation (API)

User-friendly object-oriented API for differentiable simulation `unsim.api`.
A `DifferentiableWorld` is obtained by `World.compile(backend="jax")`.

## DifferentiableWorld

```{eval-rst}
.. autoclass:: unsim.api.DifferentiableWorld
   :members:
```

## SimResult

```{eval-rst}
.. autoclass:: unsim.api.SimResult
   :members:
```

## Metrics

```{eval-rst}
.. autoclass:: unsim.api.Metrics
   :members:
```

## LinkView

```{eval-rst}
.. autoclass:: unsim.api.LinkView
   :members:
```

## NodeView

```{eval-rst}
.. autoclass:: unsim.api.NodeView
   :members:
```

## PathView

```{eval-rst}
.. autoclass:: unsim.api.PathView
   :members:
```

## ODView

```{eval-rst}
.. autoclass:: unsim.api.ODView
   :members:
```

## Objective

```{eval-rst}
.. autoclass:: unsim.api.Objective
   :members:
```

## Gradients

```{eval-rst}
.. autoclass:: unsim.api.Gradients
   :members:
```

## ParameterRef

```{eval-rst}
.. autoclass:: unsim.api.ParameterRef
   :members:
```

### Selecting multiple FD quantities on one link

Independent fundamental-diagram (FD) fields on the same link can be selected
together. For example, when the link was created with the default
`(u, kappa, tau)` parameterization:

```python
speed = model.parameter("road", "free_flow_speed")
density = model.parameter("road", "jam_density")
gradients = model.objective(loss_from_result).gradient(wrt=[speed, density])
```

The selected fields are applied together and the FD quantities are
recomputed once, so changing one selected value does not reset another. Their
order in `wrt` does not change the physical parameters. Unselected independent
quantities, including fixed `tau`, come from the compiled authoring snapshot,
as they do for a single FD parameter. The independent fields are:

| Link FD parameterization | Independently selectable fields |
| --- | --- |
| `(u, kappa, tau)` | `free_flow_speed`, `jam_density` |
| `(u, w, tau)` | `free_flow_speed`, `backward_wave_speed` |
| `(u, w, capacity)` | `free_flow_speed`, `backward_wave_speed`, `capacity` |

For a joint group on the same link, duplicate or derived fields, inconsistent
FD parameterizations, and an interleaved `CustomVariable` are rejected rather
than assigned an ambiguous ordering. A custom variable can run before or
after a joint group; its own injection remains responsible for any FD
consistency it changes. These checks do not turn a `ParameterRef` into an
owner-bound identity: use references from the intended compiled model.

### Fixed history-window limitation

The existing differentiable solver sizes its sliding history window when the
world is compiled, using the **initial** free-flow and backward-wave speeds.
Changing an FD parameter later does not resize `model.raw.config.window_size`.
If an admissible parameter value requires a longer lookback than that window
retains, the differentiable trajectory and its gradients can be incorrect. This
is an existing solver limitation, independent of joint FD parameter injection.

For every parameter value considered, including finite-difference probes,
check that the fixed window covers the largest required free-flow and
backward-wave interpolation lookback across all links. In the current solver,
a conservative check follows the compilation rule: with valid positive speeds,
the configured window should be at least
`min(ceil(max_i(L_i/(u_i*dt), L_i/(w_i*dt))) + 2, tsize)`, evaluated over the
entire allowed parameter range. Here `w_i` is the backward-wave speed used by
the solver after recomputing the FD, not necessarily an independently selected
parameter. Bounds that only cover the initial value are insufficient. When
the check fails, do not rely on the resulting windowed trajectories or
gradients as valid for those values; compare against a full-history forward run
and ensure the window and parameter range are safe. Satisfying the window
condition alone does not establish differentiability at min/max switches or
guarantee correctness across all parameter values.

For example, an independently reproduced one-link case with `L=1000 m`,
`dt=5 s`, `tmax=1200 s`, initial `u=20 m/s`, `kappa=0.2 veh/m`, and demand
`0.31 veh/s` over `[0,1200)` compiles a window of 42 steps. Changing only
`u` to `2 m/s` requires a 100-step free-flow lookback. At the end of this case,
the windowed differentiable and full-history forward runs give about 281.43 and
200.00 vehicles respectively for cumulative departure. This example is a
solver-boundary counterexample, not a claim about the joint-injection fix.

## PathRef

```{eval-rst}
.. autoclass:: unsim.api.PathRef
   :members:
```

## TollVariable

```{eval-rst}
.. autoclass:: unsim.api.TollVariable
   :members:
```

## CustomVariable

```{eval-rst}
.. autoclass:: unsim.api.CustomVariable
   :members:
```

## Problem

```{eval-rst}
.. autoclass:: unsim.api.Problem
   :members:
```

## Solution

```{eval-rst}
.. autoclass:: unsim.api.Solution
   :members:
```

## PiecewiseConstant

```{eval-rst}
.. autoclass:: unsim.api.PiecewiseConstant
   :members:
```
