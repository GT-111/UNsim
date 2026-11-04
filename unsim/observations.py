"""Differentiable link observations derived from UNsim cumulative counts.

These are aggregated simulator quantities, not calibrated point-detector
observations. In particular, a downstream crossing rate divided by a whole-link
inventory density does not identify space-mean speed, so speed is unsupported.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp


class LinkObservations(NamedTuple):
    """Arrays have shape [complete_time_windows, directed_links] unless noted.

    ``flow_in_veh_s`` and ``flow_out_veh_s`` are changes in cumulative entry and
    exit counts divided by the interval. ``density_veh_m`` is the arithmetic
    mean of right-endpoint link inventories divided by length, not a detector
    occupancy or an exact Edie space-time density. ``speed_m_s`` is NaN with an
    all-false mask: boundary counts and link inventory do not determine speed.
    ``discarded_tail_s`` identifies a final incomplete observation window.
    """

    time_start_s: jnp.ndarray
    time_end_s: jnp.ndarray
    flow_in_veh_s: jnp.ndarray
    flow_out_veh_s: jnp.ndarray
    vehicle_count_start: jnp.ndarray
    vehicle_count_end: jnp.ndarray
    density_veh_m: jnp.ndarray
    speed_m_s: jnp.ndarray
    flow_in_valid: jnp.ndarray
    flow_out_valid: jnp.ndarray
    vehicle_count_start_valid: jnp.ndarray
    vehicle_count_end_valid: jnp.ndarray
    density_valid: jnp.ndarray
    speed_valid: jnp.ndarray
    discarded_tail_s: float


def from_state(state, config, sample_interval_s=300.0):
    """Aggregate a JAX ``SimState`` and its ``NetworkConfig`` without host copies.

    The requested interval must be an integer multiple of the simulation step.
    Only complete nonoverlapping windows are emitted; an incomplete tail is
    reported rather than padded. Counts are not clipped or repaired: negative
    or nonfinite increments invalidate flow; negative or nonfinite inventories
    invalidate density and the corresponding inventory snapshots.

    This function uses JAX operations on counts and is usable inside an
    objective or ``jax.jit`` when the config and interval are static. It adds no
    new dynamics and preserves UNsim's existing model and derivative semantics.
    """
    interval = float(sample_interval_s)
    dt = float(config.deltat)
    if not math.isfinite(interval) or not math.isfinite(dt) or interval <= 0 or dt <= 0:
        raise ValueError("sample interval and simulation timestep must be positive finite seconds")
    steps_float = interval / dt
    steps = round(steps_float)
    if steps < 1 or not math.isclose(steps_float, steps, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("sample interval must be an integer multiple of the simulation timestep")
    n_bins = int(config.tsize) // steps
    if n_bins < 1:
        raise ValueError("simulation duration must contain at least one complete observation window")
    used = n_bins * steps
    arrival = jnp.asarray(state.cum_arrival)
    departure = jnp.asarray(state.cum_departure)
    lengths = jnp.asarray(config.link_lengths)
    if (arrival.ndim != 2 or departure.shape != arrival.shape
            or arrival.shape[0] != int(config.n_links) or arrival.shape[1] < used + 1
            or lengths.shape != (int(config.n_links),)):
        raise ValueError("cumulative count or link-length shapes do not match the simulation config")

    arr0, arr1 = arrival[:, :used:steps], arrival[:, steps:used + 1:steps]
    dep0, dep1 = departure[:, :used:steps], departure[:, steps:used + 1:steps]
    inflow = (arr1 - arr0) / interval
    outflow = (dep1 - dep0) / interval
    arr_steps = jnp.diff(arrival[:, :used + 1], axis=1).reshape(config.n_links, n_bins, steps)
    dep_steps = jnp.diff(departure[:, :used + 1], axis=1).reshape(config.n_links, n_bins, steps)
    in_valid = jnp.isfinite(arr_steps).all(axis=-1) & (arr_steps >= 0).all(axis=-1)
    out_valid = jnp.isfinite(dep_steps).all(axis=-1) & (dep_steps >= 0).all(axis=-1)
    inventory = (arrival[:, 1:used + 1] - departure[:, 1:used + 1]).reshape(config.n_links, n_bins, steps)
    valid_length = jnp.isfinite(lengths) & (lengths > 0)
    safe_length = jnp.where(valid_length, lengths, 1.)
    density = inventory.mean(axis=-1) / safe_length[:, None]
    density_valid = (jnp.isfinite(inventory).all(axis=-1) & (inventory >= 0).all(axis=-1)
                     & valid_length[:, None])
    count_start, count_end = arr0 - dep0, arr1 - dep1
    count_start_valid = jnp.isfinite(count_start) & (count_start >= 0)
    count_end_valid = jnp.isfinite(count_end) & (count_end >= 0)
    times_start = jnp.arange(n_bins) * interval
    times_end = times_start + interval
    return LinkObservations(
        time_start_s=times_start, time_end_s=times_end,
        flow_in_veh_s=inflow.T, flow_out_veh_s=outflow.T,
        vehicle_count_start=count_start.T, vehicle_count_end=count_end.T,
        density_veh_m=density.T, speed_m_s=jnp.full_like(density.T, jnp.nan),
        flow_in_valid=in_valid.T, flow_out_valid=out_valid.T,
        vehicle_count_start_valid=count_start_valid.T, vehicle_count_end_valid=count_end_valid.T,
        density_valid=density_valid.T, speed_valid=jnp.zeros_like(density.T, dtype=bool),
        discarded_tail_s=(int(config.tsize) - used) * dt,
    )


def from_result(result, sample_interval_s=300.0):
    """Aggregate the public ``SimResult`` returned by ``World.compile().run()``."""
    return from_state(result.state, result.config, sample_interval_s=sample_interval_s)
