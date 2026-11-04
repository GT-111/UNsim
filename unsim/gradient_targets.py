"""Bounded, gradient-derived calibration targets for UNsim experiments.

This is a small traffic-calibration adaptation of the live-row median clipping
and action-gradient regression target ideas in NablaMimic (Apache-2.0), commit
bfbb2a6773b4776c17402a34a93ec7589d9c7c18, ``src/nablamimic/nets.py`` and
``src/nablamimic/train.py``. It does not copy a learned world model: gradients
come from the original UNsim objective and its differentiable simulator.
See docs/gradient_targets.md for the differences, provenance and limitations.
"""

from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .api import ParameterRef, VariableSet


# Keep the supported surface explicit: arbitrary injection functions and toll
# blocks can overlap other writes in ways this helper cannot verify.
_FD_FIELDS = frozenset(('free_flow_speed', 'jam_density', 'capacity', 'backward_wave_speed'))
_SUPPORTED_FIELDS = {
    'link': _FD_FIELDS | frozenset(('merge_priority', 'capacity_out', 'capacity_in')),
    'node': frozenset(('flow_capacity', 'absorption_ratio')),
    'demand': frozenset(('flow',)),
}
_REF_FIELDS = ('kind', 'field', 'index', 'name', 'unit', 'shape', 'fd_parameterization')


def _validate_refs(model, refs):
    """Rebuild supported descriptors and reject known noncomposable writes."""
    if not refs:
        raise ValueError('refs must be a nonempty sequence of distinct variables')
    fd_links = set()
    for ref in refs:
        if type(ref) is not ParameterRef:
            raise TypeError('refs must be builtin ParameterRef objects; use the native UNsim API for custom or toll variables')
        if ref.kind not in _SUPPORTED_FIELDS or ref.field not in _SUPPORTED_FIELDS[ref.kind]:
            raise ValueError('unsupported ParameterRef kind or field; use the native UNsim API for toll or custom injection')
        if isinstance(ref.index, bool) or not isinstance(ref.index, Integral) or ref.index < 0:
            raise ValueError('ParameterRef index must be a nonnegative integer in the supplied model')
        try:
            if ref.kind == 'link':
                entity = model._snapshot_link(ref.index)
            elif ref.kind == 'node':
                entity = model._world.NODES[ref.index]
            else:
                entity = model._snapshot_demand(ref.index)
            rebuilt = model.parameter(entity, ref.field)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError('ParameterRef cannot be reconstructed in the supplied model') from exc
        if any(getattr(ref, field) != getattr(rebuilt, field) for field in _REF_FIELDS):
            raise ValueError('ParameterRef does not match the supplied model descriptor')
        if ref.kind == 'link' and ref.field in _FD_FIELDS:
            if ref.index in fd_links:
                raise ValueError('multiple fundamental-diagram fields on the same link are not supported')
            fd_links.add(ref.index)
    if len(set(refs)) != len(refs):
        raise ValueError('refs must be a nonempty sequence of distinct variables')


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
    nonempty sequence of supported builtin ParameterRef descriptors, ``objectives`` is
    one Objective per row (each may close over that row's observed past), and
    ``parameters`` has shape [batch, sum(ref.size)]. A public CustomVariable
    delegates the injection to UNsim's own VariableSet; the source model and
    its snapshot are never mutated. This is a host-side small-batch helper,
    not a vectorized JIT or a training loop. Descriptors are compared field by
    field to references reconstructed from the supplied model. They carry no
    owner identity, so an identical descriptor from another model cannot be
    distinguished. Multiple FD fields for the same link and opaque or toll
    variables are rejected before simulation; see docs/gradient_targets.md.
    """
    refs = tuple(refs)
    _validate_refs(model, refs)
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
    # Rescale before squaring to avoid common FP32 underflow for small rows.
    # This is not a promise for arbitrary subnormal or extreme inputs.
    row_max = jnp.max(jnp.abs(normalized_grad),axis=-1)
    row_denom = jnp.where(row_max > 0,row_max,1.)
    norms = row_max*jnp.linalg.norm(normalized_grad/row_denom[:,None],axis=-1)
    live = norms > 0
    median = jnp.where(jnp.any(live),jnp.nanmedian(jnp.where(live,norms,jnp.nan)),0.)
    threshold = jnp.where(jnp.any(live),kappa*median,jnp.inf)
    safe_norms = jnp.where(live,norms,1.)
    scales = jnp.where(live & (norms > threshold),threshold/safe_norms,1.)
    clipped = normalized_grad*scales[:,None]
    count = jnp.sum(live)
    max_clipped = jnp.max(jnp.abs(clipped))
    safe_max = jnp.where(max_clipped > 0,max_clipped,1.)
    raw_rms = max_clipped*jnp.sqrt(jnp.sum(jnp.square(clipped/safe_max)) /
                                   (jnp.maximum(count,1)*theta.shape[1]))
    rms = jnp.hypot(raw_rms,jnp.sqrt(epsilon))
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
    ``lower``/``upper`` broadcast to [P]. ``delta`` is the nominal upper bound
    on RMS update over live rows in unit-box coordinates *before* projection,
    not in veh/s or other physical units. Epsilon suppresses small updates.
    Nonzero rows determine the batch median. The threshold and RMS are
    batch statistics, so changing other rows can change a given row's target.
    Invalid or out-of-bounds inputs raise rather than being silently repaired.
    The clipped gradient is not the unmodified derivative of the UNsim loss.
    This host-validating wrapper uses NumPy and cannot itself be traced by
    ``jax.jit`` or ``jax.grad``; only the private math core is JAX-traceable.
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
    # Fail closed for magnitudes whose original FP32 squared row norm would
    # overflow; use FP64 on the host solely for this range check.
    g64 = g.astype(np.float64)*(high.astype(np.float64)-low.astype(np.float64))
    if np.any(np.sum(g64*g64,axis=1) > np.finfo(np.float32).max):
        raise FloatingPointError('gradient clipping or projection overflowed')
    result = _targets_math(jnp.asarray(x),jnp.asarray(g),jnp.asarray(low),jnp.asarray(high),
                           float(delta),float(kappa),float(epsilon))
    finite_fields = (result.targets,result.normalized_gradients,result.clipped_gradients,
                     result.row_norms,result.row_scales,result.step_multiplier,result.clipped_fraction)
    if not all(np.isfinite(np.asarray(a)).all() for a in finite_fields):
        raise FloatingPointError('gradient clipping or projection overflowed')
    return result
