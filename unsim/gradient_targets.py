"""Bounded, gradient-derived calibration targets for UNsim experiments.

This is a small traffic-calibration adaptation of the live-row median clipping
and action-gradient regression target ideas in NablaMimic (Apache-2.0), commit
bfbb2a6773b4776c17402a34a93ec7589d9c7c18, ``src/nablamimic/nets.py`` and
``src/nablamimic/train.py``. It does not copy a learned world model: gradients
come from the original UNsim objective and its differentiable simulator.
See docs/gradient_targets.md for the differences, provenance and limitations.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .api import VariableSet


class GradientTargetBatch(NamedTuple):
    """Target values in physical and normalized units plus clipping diagnostics."""

    targets: jax.Array
    normalized_targets: jax.Array
    normalized_gradients: jax.Array
    clipped_gradients: jax.Array
    row_norms: jax.Array
    row_scales: jax.Array
    live: jax.Array
    median_norm: jax.Array
    threshold: jax.Array
    step_multiplier: jax.Array
    clipped_fraction: jax.Array


def evaluate_objectives(model,refs,objectives,parameters,*,jit=False):
    """Evaluate per-row UNsim ``Objective.value_and_gradient`` at selected values.

    ``model`` is one compiled fixed-topology DifferentiableWorld, ``refs`` is a
    nonempty sequence of its ParameterRef/Variable objects, ``objectives`` is
    one Objective per row (each may close over that row's observed past), and
    ``parameters`` has shape [batch, sum(ref.size)]. A public CustomVariable
    delegates the injection to UNsim's own VariableSet; the source model and
    its snapshot are never mutated. This is a host-side small-batch helper,
    not a vectorized JIT or a training loop.
    """
    refs = tuple(refs)
    if not refs or len(set(refs)) != len(refs):
        raise ValueError('refs must be a nonempty sequence of distinct variables')
    varset = VariableSet(model,refs)
    rows = np.asarray(parameters,dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != varset.size:
        raise ValueError(f'parameters must have shape [batch, {varset.size}]')
    if not np.isfinite(rows).all():
        raise ValueError('parameters must be finite')
    objectives = tuple(objectives)
    if len(objectives) != rows.shape[0]:
        raise ValueError('there must be one objective per parameter row')
    if any(getattr(objective,'_model',None) is not model for objective in objectives):
        raise ValueError('each UNsim objective must belong to the supplied model')
    values,gradients = [],[]
    for row,objective in zip(rows,objectives):
        variable = model.variable(shape=(varset.size,),initial=row,
                                  inject=lambda params,x:varset.inject(x,params),
                                  name='gradient_target_parameters')
        value,gradient = objective.value_and_gradient(wrt=[variable],jit=jit)
        values.append(value)
        gradients.append(gradient[variable])
    values,gradients = jnp.stack(values),jnp.stack(gradients)
    if not np.isfinite(np.asarray(values)).all() or not np.isfinite(np.asarray(gradients)).all():
        raise FloatingPointError('UNsim objective produced a nonfinite value or gradient')
    return values,gradients


def _targets_math(theta,grad,lower,upper,delta,kappa,epsilon):
    """JAX-only target math; public wrapper validates concrete batch inputs."""
    span = upper-lower
    normalized = (theta-lower)/span
    # dL/d(normalized parameter) = dL/d(physical parameter) * span.
    normalized_grad = grad*span
    norms = jnp.linalg.norm(normalized_grad,axis=-1)
    live = norms > 0
    median = jnp.where(jnp.any(live),jnp.nanmedian(jnp.where(live,norms,jnp.nan)),0.)
    threshold = jnp.where(jnp.any(live),kappa*median,jnp.inf)
    scales = jnp.minimum(1.,threshold/(norms+epsilon))
    clipped = normalized_grad*scales[:,None]
    count = jnp.sum(live)
    rms = jnp.sqrt(jnp.sum(jnp.square(clipped))/(jnp.maximum(count,1)*theta.shape[1])+epsilon)
    step_multiplier = jnp.where(count > 0,delta/rms,0.)
    target_norm = jnp.clip(normalized-step_multiplier*clipped,0.,1.)
    target = lower+span*target_norm
    fraction = jnp.sum(live & (scales < 1.))/jnp.maximum(count,1)
    # Treat regression labels as fixed targets, never differentiate through
    # the target-construction update when training a predictor.
    return GradientTargetBatch(jax.lax.stop_gradient(target),jax.lax.stop_gradient(target_norm),
                               normalized_grad,clipped,norms,scales,live,median,threshold,
                               step_multiplier,fraction)


def gradient_targets(parameters,physical_gradients,lower,upper,*,delta=0.05,kappa=3.0,epsilon=1e-30):
    """Clip per-row gradients and project one RMS-scaled step into a box.

    Arrays ``parameters`` and ``physical_gradients`` have shape [B,P]; finite
    ``lower``/``upper`` broadcast to [P]. ``delta`` is the target RMS update in
    unit-box coordinates *before* projection, not in veh/s or other physical
    units. Nonzero rows determine the batch median. The threshold and RMS are
    batch statistics, so changing other rows can change a given row's target.
    Invalid or out-of-bounds inputs raise rather than being silently repaired.
    The clipped gradient is not the unmodified derivative of the UNsim loss.
    """
    x = np.asarray(parameters,dtype=np.float32)
    g = np.asarray(physical_gradients,dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1 or g.shape != x.shape:
        raise ValueError('parameters and physical_gradients must have the same nonempty [B,P] shape')
    try:
        low = np.broadcast_to(np.asarray(lower,dtype=np.float32),(x.shape[1],))
        high = np.broadcast_to(np.asarray(upper,dtype=np.float32),(x.shape[1],))
    except ValueError as exc:
        raise ValueError('lower and upper must broadcast to [P]') from exc
    if not all(np.isfinite(a).all() for a in (x,g,low,high)):
        raise ValueError('parameters, gradients and bounds must be finite')
    if np.any(low >= high):
        raise ValueError('lower must be strictly less than upper')
    if np.any(x < low) or np.any(x > high):
        raise ValueError('initial parameters must lie in the declared box')
    if not (np.isfinite(delta) and delta >= 0 and np.isfinite(kappa) and kappa > 0
            and np.isfinite(epsilon) and np.float32(epsilon) > 0):
        raise ValueError('delta >= 0, kappa > 0 and epsilon > 0 must be finite')
    result = _targets_math(jnp.asarray(x),jnp.asarray(g),jnp.asarray(low),jnp.asarray(high),
                           float(delta),float(kappa),float(epsilon))
    finite_fields = (result.targets,result.normalized_gradients,result.clipped_gradients,
                     result.row_norms,result.row_scales,result.step_multiplier,result.clipped_fraction)
    if not all(np.isfinite(np.asarray(a)).all() for a in finite_fields):
        raise FloatingPointError('gradient clipping or projection overflowed')
    return result
