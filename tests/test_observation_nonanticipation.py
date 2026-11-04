"""Small, single-link nonanticipation regression for the count adapter.

This is a scenario-specific check, not a proof for every UNsim topology or
route-choice mode.  The independently changed quantity is future demand;
no real-world data or held-out scenarios are used.
"""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from unsim import World
from unsim.observations import from_result, from_state
from unsim.unsim_diff import simulate


def _single_link_model(duration_s):
    world = World(name="", deltat=5, tmax=duration_s, print_mode=0,
                  show_mode=0, save_mode=0)
    world.addNode("origin", 0, 0)
    world.addNode("destination", 1, 0)
    world.addLink("road", "origin", "destination", length=100,
                  free_flow_speed=20, jam_density=0.2)
    # Keep the authored demand schedule the same in the full and short worlds.
    world.adddemand("origin", "destination", 0, 60, 0.3)
    return world.compile(backend="jax")


def test_future_demand_and_truncation_leave_past_observations_unchanged():
    model = _single_link_model(60)
    params, config = model.raw.params, model.raw.config
    baseline = simulate(params, config, differentiable=False)
    # Six 5 s steps end at the 30 s anchor.  Change only later demand slots,
    # consistently in both total and per-OD inputs.
    changed = params._replace(
        demand_rate=params.demand_rate.at[:, 6:].multiply(1.5),
        od_demand_rate=params.od_demand_rate.at[:, :, 6:].multiply(1.5),
    )
    alternative = simulate(changed, config, differentiable=False)
    observations = from_state(baseline, config, sample_interval_s=10)
    counterfactual = from_state(alternative, config, sample_interval_s=10)
    truncated = from_result(_single_link_model(30).run(differentiable=False),
                            sample_interval_s=10)

    np.testing.assert_array_equal(observations.time_end_s[:3], [10, 20, 30])
    np.testing.assert_array_equal(truncated.time_end_s, [10, 20, 30])
    names = (
        "flow_in_veh_s", "flow_out_veh_s", "vehicle_count_start",
        "vehicle_count_end", "density_veh_m", "flow_in_valid",
        "flow_out_valid", "vehicle_count_start_valid",
        "vehicle_count_end_valid", "density_valid",
    )
    for name in names:
        prefix = np.asarray(getattr(observations, name))[:3]
        np.testing.assert_array_equal(prefix, np.asarray(getattr(counterfactual, name))[:3])
        np.testing.assert_array_equal(prefix, np.asarray(getattr(truncated, name)))

    assert np.asarray(observations.flow_out_valid[:3]).all()
    # Ensure the intervention actually changes a future observable.
    assert np.max(np.abs(np.asarray(observations.flow_out_veh_s[3:]) -
                         np.asarray(counterfactual.flow_out_veh_s[3:]))) > 0.1


def test_density_is_window_average_not_endpoint_inventory_density():
    state = SimpleNamespace(cum_arrival=jnp.array([[0., 2., 4.]]),
                            cum_departure=jnp.array([[0., 0., 1.]]))
    config = SimpleNamespace(deltat=5., tsize=2, n_links=1,
                             link_lengths=jnp.array([100.]))
    obs = from_state(state, config, sample_interval_s=10)
    # At t=5 and t=10 the stocks are 2 and 3 vehicles, respectively.
    np.testing.assert_allclose(obs.flow_in_veh_s, [[0.4]], atol=1e-7)
    np.testing.assert_allclose(obs.flow_out_veh_s, [[0.1]], atol=1e-7)
    np.testing.assert_allclose(obs.vehicle_count_end, [[3.]], atol=0)
    np.testing.assert_allclose(obs.density_veh_m, [[0.025]], atol=1e-7)
    np.testing.assert_allclose(obs.vehicle_count_end / config.link_lengths,
                               [[0.03]], atol=1e-7)
    assert bool(obs.density_valid[0, 0])
