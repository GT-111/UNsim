# UNsim gradient-derived calibration targets: CPU prototype

This prototype estimates bounded simulator parameters from **past** synthetic
observations using UNsim's differentiable physical simulator. It does not
train a learned world model, use future real traffic flow as an input, or claim
that a traffic predictor has improved.

The design adapts the live-row median relative clipping and gradient-derived
regression target ideas from [NablaMimic](https://github.com/OamIcs/NablaMimic)
commit `bfbb2a6773b4776c17402a34a93ec7589d9c7c18`, particularly
`src/nablamimic/nets.py:clip_rows_to_median` and the `action_target_update`
section of `train.py`. The upstream project is Apache-2.0 and its attribution
notice is preserved in this repository's `NOTICE`. NablaMimic clips action
cotangents during backpropagation through a robot simulator and forms action
targets using an action-scale RMS. This prototype instead clips the **final
gradient of physical calibration parameters**, then projects into declared
parameter bounds. It is an adaptation of the idea, not an implementation of
the complete NablaMimic training procedure or its learned policy.

For a fixed compiled topology, `evaluate_objectives` takes one UNsim objective
per sample and a parameter matrix `theta[B,P]`. Each objective can close over
that sample's already-observed historical measurements. It calls
`Objective.value_and_gradient` separately for each row. A UNsim
`CustomVariable` delegates injection back to the existing `VariableSet`; no
simulator equations or optimizer are reimplemented, and the compiled base
parameters are not changed. The helper currently checks the objective's
internal model identity to prevent accidental cross-model evaluation; this is
a dependency on the present UNsim API. It is host-side and intended for small
prototype batches, not a JIT-vectorized production training system.

The helper accepts only supported scalar, built-in `ParameterRef` descriptors:
link FD or plain physical fields, node fields and demand flow. It rejects toll
blocks, toll `ParameterRef`s and custom/unknown injection functions; advanced
uses should go through UNsim's original API and verify their own injection
semantics. Each descriptor is rebuilt on the supplied model and checked field
by field, including kind, field, index, name, unit, shape and FD parameterization.
The descriptor itself has no model-owner identity: another model with identical
descriptors is indistinguishable, and callers must decide whether its meaning
is appropriate for the target model. Rebuild references on the intended model.
Likewise, objective identity is checked but the helper has no forecast-time
anchor and does not verify whether the objective uses only historical data.

At most one fundamental-diagram field may be selected on any one link. The
existing `VariableSet.inject` recalculates the FD state from the original link
for each field, so selecting two fields of one link silently overwrites an
earlier injection. This helper rejects that combination before simulation;
different links and non-overlapping fields on the same link remain supported.

For finite box bounds `lower < upper`, define `z=(theta-lower)/(upper-lower)`
and the dimensionless-coordinate gradient
`g_z = g_theta * (upper-lower)`. Scale a gradient row down to the threshold
only if its norm exceeds `kappa` times the median of **nonzero** row norms.
The already-clipped batch then determines one RMS-normalized step multiplier;
the resulting `z - multiplier*g_z` is projected to `[0,1]`, mapped back to
physical units and detached with `jax.lax.stop_gradient` for regression.
Let `r` be the RMS of the clipped gradients over the live rows and all P
coordinates. Before projection, the ideal-arithmetic update RMS is
`delta * r / sqrt(r**2 + epsilon)`, at most the nominal `delta` in normalized
box coordinates; it can be much smaller when gradients are small relative to
`sqrt(epsilon)`. Projection cannot increase its distance from the starting
point in ideal arithmetic. `delta` is not a step size in veh/s or another
physical unit. Clipping scales a row only when its norm is strictly above the
threshold; epsilon is used in RMS normalization, not added to each row norm.
Rescaling before squaring reduces common FP32 underflow, but floating-point
limits, rounding and subnormal gradients still matter. The public wrapper
requires epsilon to be a finite normal FP32 value (at least
`numpy.finfo(numpy.float32).tiny` after conversion): some JAX backends flush
subnormal values to zero, which would silently remove the documented
regularization. The default `1e-30` satisfies this requirement. A row with zero
gradient remains unchanged; an all-zero batch reports no live rows, zero
median and step, and an infinite unused clipping threshold. Nonfinite inputs,
invalid bounds, out-of-box starting values and numerical overflow are rejected
rather than silently repaired.

The original simulator gradient is the gradient of the stated scalar loss.
The row-wise clipped direction is a modified training signal; it must not be
reported as that original exact gradient. Per-row clipping preserves each
finite nonzero row's direction, but its threshold and the RMS step depend on
the **other samples in the batch**. Tests cover row permutation, ignored zero
rows and an example where changing the rest of a batch changes the scale of a
fixed row. Direct raw objective evaluation does not mix rows. Projection,
clipping, a nonlinear simulator and a finite step do not guarantee that every
future loss will decrease.

## Real UNsim single-parameter check

The test uses a fixed 300 m link, a scalar demand flow, and a synthetic count
generated by UNsim at 150 s; this is before a 200 s forecast anchor. The
calibration objective is squared error in that historical cumulative departure
count. At flow 0.22 veh/s, the original simulator automatic gradient was about
`-5102.995`, and centered finite differences with `epsilon=0.001` gave about
`-5103.01`. With bounds `[0.05,0.65]`, `delta=0.05`, `kappa=3` and a batch
of one, the projected target is 0.25 and the actual simulator loss decreases
from about `357.21` to `220.52`. The check is local to this scenario; it is not
proof of finite-difference agreement at simulator kinks or for arbitrary
topologies.

The runnable example creates four synthetic scenarios on the same topology,
with different demand values and only their own historical 150 s observations.
For the fixed example seed, four loss values before the target step were
approximately `[590.49, 89.30, 357.21, 89.30]` and afterwards
`[339.85, 51.40, 205.59, 51.40]`. A direct JAX SGD update using the same
clipped gradient and projection agrees with the constructed normalized target
within `6e-8`. A 33-parameter JAX `stax` regressor, given only current demand
estimate and past count, fits those **same four examples**: after 250 Adam
steps its training target MSE falls from about `0.0547` to `0.0000436`, and the
simulator losses at its outputs are approximately
`[327.19, 47.56, 217.26, 64.91]`. This is an in-sample engineering check,
not evidence of generalization, forecasting accuracy, or superiority to direct
optimization. A held-out network/scenario comparison and complete simulation,
regression and inference costs would be required for such a claim.

`gradient_targets` validates inputs on the host with NumPy. Do not nest this
public wrapper in `jax.jit` or differentiate through it with `jax.grad`; either
operation on traced inputs raises a tracer-conversion error. Its detached labels
can be passed to JAX training after construction. The test of JIT and
stop-gradient on `_targets_math` concerns a private math core, not support for
tracing the public wrapper. Similarly, `evaluate_objectives(..., jit=True)`
requests JIT for its individual UNsim objective evaluations; it does not make
the host batch wrapper itself JIT-compatible.

JAX 0.12.0.1 and SciPy are available in the isolated CPU environment; Optax
and Flax are not installed. The example uses JAX's existing
`example_libraries.optimizers` (`sgd`, `adam`) and `stax` rather than adding a
dependency or implementing an optimizer. These are example APIs, not a claim
of a production optimizer integration. The experiments use an isolated CPU
environment and do not change the traffic-forecasting training environment.

From the UNsim repository root, after activating an isolated environment with
JAX and the repository dependencies installed, run on CPU only:

```sh
PYTHONPATH=. JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' \
  python examples/example_gradient_targets.py
```

Run `-m pytest -q tests/test_gradient_targets.py` with the same environment to
check UNsim AD versus finite differences, actual objective decrease, clipping,
box projection, target stop-gradient, batch coupling and optimizer equivalence.
