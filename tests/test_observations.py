"""Count-based observation export: units, masks and differentiability."""
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from unsim import World
from unsim.observations import from_result, from_state
from unsim.unsim_diff import world_to_jax, simulate


def test_known_counts_units_windows_and_unsupported_speed():
    state = SimpleNamespace(
        cum_arrival=jnp.array([[0., 2., 4., 5., 6., 7.], [0., 0., 3., 4., 4., 4.]]),
        cum_departure=jnp.array([[0., 0., 1., 2., 4., 5.], [0., 0., 0., 1., 2., 3.]]),
    )
    cfg = SimpleNamespace(deltat=5., tsize=5, n_links=2, link_lengths=jnp.array([100., 200.]))
    obs = from_state(state, cfg, sample_interval_s=10)
    np.testing.assert_allclose(obs.time_start_s, [0, 10])
    np.testing.assert_allclose(obs.time_end_s, [10, 20])
    np.testing.assert_allclose(obs.flow_in_veh_s[:, 0], [.4, .2])
    np.testing.assert_allclose(obs.flow_out_veh_s[:, 0], [.1, .3])
    np.testing.assert_allclose(obs.density_veh_m[:, 0], [.025, .025])
    np.testing.assert_allclose(obs.vehicle_count_start[:, 0], [0, 3])
    np.testing.assert_allclose(obs.vehicle_count_end[:, 0], [3, 2])
    assert obs.discarded_tail_s == 5
    assert np.all(np.asarray(obs.flow_out_valid))
    assert np.all(np.asarray(obs.vehicle_count_start_valid))
    assert np.all(np.asarray(obs.vehicle_count_end_valid))
    assert np.all(np.asarray(obs.density_valid))
    assert not np.any(np.asarray(obs.speed_valid))
    assert np.isnan(np.asarray(obs.speed_m_s)).all()


def test_invalid_data_are_masked_not_repaired():
    state = SimpleNamespace(cum_arrival=jnp.array([[0., 1., .5]]),
                            cum_departure=jnp.array([[0., 2., 2.]]))
    cfg = SimpleNamespace(deltat=5., tsize=2, n_links=1, link_lengths=jnp.array([100.]))
    obs = from_state(state, cfg, sample_interval_s=10)
    assert not bool(obs.flow_in_valid[0, 0])
    assert not bool(obs.density_valid[0, 0])
    assert not bool(obs.vehicle_count_end_valid[0, 0])
    assert float(obs.density_veh_m[0, 0]) < 0  # no silent clipping
    with pytest.raises(ValueError, match="integer multiple"):
        from_state(state, cfg, sample_interval_s=7)
    with pytest.raises(ValueError, match="at least one"):
        from_state(state, cfg, sample_interval_s=20)


def test_complete_observation_structure_is_jittable():
    cfg = SimpleNamespace(deltat=5., tsize=2, n_links=1, link_lengths=jnp.array([100.]))
    departure = jnp.array([[0., 0., 1.]])

    @jax.jit
    def export(arrival):
        return from_state(SimpleNamespace(cum_arrival=arrival, cum_departure=departure),
                          cfg, sample_interval_s=10)

    result = export(jnp.array([[0., 1., 2.]]))
    np.testing.assert_allclose(result.flow_out_veh_s, [[.1]])
    assert np.isnan(np.asarray(result.speed_m_s)).all()


def test_result_wrapper_and_jax_gradient_matches_finite_difference():
    world = World(name="", deltat=5, tmax=600, print_mode=0, show_mode=0, save_mode=0)
    world.addNode("origin", 0, 0)
    world.addNode("destination", 1, 0)
    world.addLink("road", "origin", "destination", length=1000, free_flow_speed=20, jam_density=.2)
    world.adddemand("origin", "destination", 0, 300, .31)
    params, config = world_to_jax(world)

    def objective(scale):
        state = simulate(params._replace(demand_rate=params.demand_rate * scale), config)
        # Sum interval exit flow; the exported state remains in JAX's computation graph.
        return jnp.sum(from_state(state, config, sample_interval_s=60).flow_out_veh_s)

    value = float(objective(1.))
    grad = float(jax.grad(objective)(1.))
    delta = .002
    finite = (float(objective(1. + delta)) - float(objective(1. - delta))) / (2 * delta)
    assert np.isfinite(value) and np.isfinite(grad)
    assert grad > 0
    assert grad == pytest.approx(finite, rel=.02, abs=.002)
    result = world.compile(backend="jax").run()
    obs = from_result(result, sample_interval_s=60)
    assert obs.flow_out_veh_s.shape == (10, 1)
    np.testing.assert_allclose(obs.vehicle_count_end[:, 0],
                               np.asarray(result.state.cum_arrival[0, 12::12] -
                                          result.state.cum_departure[0, 12::12]), atol=1e-5)
