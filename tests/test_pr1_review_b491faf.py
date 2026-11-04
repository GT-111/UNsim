"""Regression probes for findings on draft PR #1 at b491faf; CPU-only."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from unsim import World
from unsim.gradient_targets import evaluate_objectives, gradient_targets
from unsim.observations import from_state


@pytest.fixture(scope='module')
def road_model():
    w = World(name='', deltat=5, tmax=300, print_mode=0)
    w.addNode('o', 0, 0)
    w.addNode('d', 1, 0)
    link = w.addLink('r', 'o', 'd', length=1000, free_flow_speed=23.4, jam_density=.2)
    w.adddemand('o', 'd', 0, 300, .2)
    model = w.compile(backend='jax')
    return model, link


def test_same_link_independent_fd_fields_must_not_be_silently_overwritten(road_model):
    model, link = road_model
    u = model.parameter(link, 'free_flow_speed')
    k = model.parameter(link, 'jam_density')
    objective = model.objective(lambda result: result.link('r').cum_departure()[12])
    with pytest.raises(ValueError, match='multiple fundamental-diagram fields'):
        evaluate_objectives(model, [u, k], [objective], [[27.1, .2]])


def test_foreign_ref_should_not_silently_set_same_index_on_another_world(road_model):
    source, source_link = road_model
    other_world = World(name='', deltat=5, tmax=100, print_mode=0)
    other_world.addNode('o', 0, 0)
    other_world.addNode('d', 1, 0)
    other_world.addLink('different_link', 'o', 'd', length=300, free_flow_speed=22, jam_density=.2)
    other_world.adddemand('o', 'd', 0, 100, .2)
    target = other_world.compile(backend='jax')
    foreign = source.parameter(source_link, 'free_flow_speed')
    objective = target.objective(lambda result: result.params.u[0])
    with pytest.raises(ValueError, match='model'):
        evaluate_objectives(target, [foreign], [objective], [[33.]])


def test_row_at_threshold_should_not_be_clipped():
    result = gradient_targets([[.5]], [[.001]], 0., 1., delta=.05, kappa=1., epsilon=.001)
    assert float(result.row_norms[0]) == pytest.approx(float(result.threshold))
    assert float(result.row_scales[0]) == pytest.approx(1.)


def test_documented_rms_step_under_small_nonzero_gradient():
    result = gradient_targets([[.5]], [[1e-16]], 0., 1., delta=.05)
    assert bool(result.live[0])
    expected = .05 * 1e-16 / np.hypot(1e-16, 1e-15)
    assert .5 - float(result.targets[0, 0]) == pytest.approx(expected, abs=5e-8)
    assert expected < .05


def test_public_gradient_targets_wrapper_is_host_only_not_jittable():
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.jit(lambda x: gradient_targets(x, jnp.ones_like(x), 0., 1.).targets)(jnp.array([[.5]]))
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.grad(lambda x: jnp.sum(gradient_targets(x, jnp.ones_like(x), 0., 1.).targets))(jnp.array([[.5]]))


def test_common_count_baseline_shift_invariant_but_independent_rebases_change_stock():
    cfg = SimpleNamespace(deltat=5., tsize=4, n_links=1, link_lengths=jnp.array([100.]))
    arrivals = jnp.array([[0., 2., 4., 5., 6.]])
    departures = jnp.array([[0., 0., 1., 2., 4.]])
    original = from_state(SimpleNamespace(cum_arrival=arrivals, cum_departure=departures), cfg, 10.)
    common = from_state(SimpleNamespace(cum_arrival=arrivals + 100., cum_departure=departures + 100.), cfg, 10.)
    independent = from_state(SimpleNamespace(cum_arrival=arrivals + 100., cum_departure=departures + 30.), cfg, 10.)
    for name in ('flow_in_veh_s', 'flow_out_veh_s', 'density_veh_m', 'vehicle_count_start', 'vehicle_count_end'):
        np.testing.assert_allclose(getattr(original, name), getattr(common, name), rtol=0, atol=0)
    np.testing.assert_allclose(original.flow_out_veh_s, independent.flow_out_veh_s)
    np.testing.assert_allclose(independent.density_veh_m - original.density_veh_m, .7, rtol=1e-6)
    assert np.asarray(independent.density_valid).all()
    # Native UNsim starts both cumulatives in a common zero gauge; the helper
    # cannot infer initial inventory if external counters are independently rebased.
