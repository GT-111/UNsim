"""Regression tests for simultaneous independent FD parameter injection."""
from dataclasses import replace
from itertools import combinations, permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from unsim import World
from unsim.api import CustomVariable, ParameterRef, VariableSet


def one_link_model(scheme='u_kappa_tau', demand=0.31, speed=20.0):
    world = World(deltat=5, tmax=600, print_mode=0, save_mode=0, show_mode=0)
    world.addNode('origin', 0, 0)
    world.addNode('destination', 1, 0)
    overrides = {'u_kappa_tau': {},
                 'u_w_tau': {'backward_wave_speed': 5.0},
                 'u_w_capacity': {'backward_wave_speed': 5.0, 'capacity': 0.8}}[scheme]
    world.addLink('road', 'origin', 'destination', length=1000,
                  free_flow_speed=speed, jam_density=0.2, **overrides)
    world.adddemand('origin', 'destination', 0, 600, demand)
    model = world.compile(backend='jax')
    assert model._snapshot_link(0).fd_parameterization == scheme
    return model


def test_two_independent_fields_compose_for_one_link():
    model = one_link_model()
    u = model.parameter('road', 'free_flow_speed')
    kappa = model.parameter('road', 'jam_density')
    variables = VariableSet(model, [u, kappa])
    result = variables.inject(jnp.array([18.0, 0.25]), model.raw.params)
    tau = model._snapshot_link(0).tau
    w = 1.0 / (tau * 0.25)
    expected_capacity = 18.0 * w * 0.25 / (18.0 + w)
    np.testing.assert_allclose([result.u[0], result.kappa[0], result.q_star[0]],
                               [18.0, 0.25, expected_capacity], rtol=1e-6)


def test_negative_and_positive_aliases_identify_the_same_link():
    model = one_link_model()
    speed = model.parameter(-1, 'free_flow_speed')
    density = model.parameter('road', 'jam_density')
    variables = VariableSet(model, [speed, density])
    params = variables.inject(jnp.array([18.0, 0.25]), model.raw.params)
    np.testing.assert_allclose(extract(params), manual_triplet('u_kappa_tau', CASES['u_kappa_tau'],
                                                             model._snapshot_link(0).tau), rtol=1e-6)


CASES = {
    'u_kappa_tau': {'free_flow_speed':18.0, 'jam_density':0.25},
    'u_w_tau': {'free_flow_speed':18.0, 'backward_wave_speed':4.5},
    'u_w_capacity': {'free_flow_speed':18.0, 'backward_wave_speed':4.5, 'capacity':0.7},
}


def manual_triplet(scheme, values, tau):
    u = values['free_flow_speed']
    if scheme == 'u_kappa_tau':
        kappa = values['jam_density']
        w = 1.0 / (tau * kappa)
        capacity = u * w * kappa / (u + w)
    elif scheme == 'u_w_tau':
        w = values['backward_wave_speed']
        kappa = 1.0 / (tau * w)
        capacity = u * w * kappa / (u + w)
    else:
        w = values['backward_wave_speed']
        capacity = values['capacity']
        kappa = capacity * (u + w) / (u * w)
    return np.asarray([u, kappa, capacity])


def extract(params, i=0):
    return jnp.stack([params.u[i], params.kappa[i], params.q_star[i]])


@pytest.mark.parametrize('scheme', list(CASES))
def test_each_parameterization_is_order_independent_and_matches_manual_fd(scheme):
    model = one_link_model(scheme)
    values = CASES[scheme]
    expected = manual_triplet(scheme, values, model._snapshot_link(0).tau)
    for order in permutations(values):
        refs = [model.parameter('road', field) for field in order]
        variables = VariableSet(model, refs)
        theta = jnp.array([values[field] for field in order])
        eager = variables.inject(theta, model.raw.params)
        compiled = jax.jit(lambda x: extract(variables.inject(x, model.raw.params)))(theta)
        np.testing.assert_allclose(extract(eager), expected, rtol=1e-6)
        np.testing.assert_allclose(compiled, expected, rtol=1e-6)
        initial = variables.inject(variables.initial_theta(), model.raw.params)
        np.testing.assert_allclose(extract(initial), extract(model.raw.params), rtol=1e-6)


@pytest.mark.parametrize('fields', list(combinations(CASES['u_w_capacity'], 2)))
def test_unselected_independent_field_stays_at_snapshot_value(fields):
    model = one_link_model('u_w_capacity')
    snapshot = model._snapshot_link(0)
    values = {'free_flow_speed':snapshot.u, 'backward_wave_speed':snapshot.w, 'capacity':snapshot.q_star}
    values.update({field:CASES['u_w_capacity'][field] for field in fields})
    variables = VariableSet(model, [model.parameter('road', field) for field in fields])
    params = variables.inject(jnp.array([values[field] for field in fields]), model.raw.params)
    np.testing.assert_allclose(extract(params), manual_triplet('u_w_capacity', values, snapshot.tau), rtol=1e-6)


@pytest.mark.parametrize('scheme', list(CASES))
def test_each_joint_field_has_matching_nonzero_autodiff_and_finite_difference(scheme):
    model = one_link_model(scheme)
    values = CASES[scheme]
    fields = list(values)
    variables = VariableSet(model, [model.parameter('road', field) for field in fields])
    theta = jnp.array([values[field] for field in fields])
    fn = jax.jit(lambda x: extract(variables.inject(x, model.raw.params)))
    jacobian = np.asarray(jax.jacfwd(fn)(theta))
    columns = []
    for i, value in enumerate(np.asarray(theta)):
        h = 0.001 * max(1.0, abs(float(value)))
        step = np.zeros(len(fields), dtype=np.float32)
        step[i] = h
        columns.append((np.asarray(fn(theta + step)) - np.asarray(fn(theta - step))) / (2 * h))
    numerical = np.stack(columns, axis=1)
    assert np.all(np.linalg.norm(jacobian, axis=0) > 1e-4)
    np.testing.assert_allclose(jacobian, numerical, rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize('scheme,fields,steps', [
    ('u_kappa_tau', ('free_flow_speed', 'jam_density'), (0.01, 0.0001)),
    ('u_w_tau', ('free_flow_speed', 'backward_wave_speed'), (0.01, 0.005)),
])
def test_joint_fields_have_nonzero_simulation_gradient(scheme, fields, steps):
    # Congested, away from the checked local min/max switches; both selected
    # FD fields influence actual simulated cumulative exits at this point.
    model = one_link_model(scheme, demand=0.95, speed=23.4)
    refs = [model.parameter('road', field) for field in fields]
    variables = VariableSet(model, refs)
    objective = model.objective(lambda result: jnp.sum(result.link('road').cum_departure()))
    loss = jax.jit(objective._make_loss(variables))
    theta = variables.initial_theta()
    _, grad = objective.value_and_gradient(wrt=refs, jit=True)
    ad = np.asarray([grad[ref] for ref in refs])
    finite = []
    for i, h in enumerate(steps):
        step = np.zeros(2, dtype=np.float32)
        step[i] = h
        finite.append((float(loss(theta + step)) - float(loss(theta - step))) / (2*h))
    assert np.all(np.abs(ad) > 1.0)
    np.testing.assert_allclose(ad, finite, rtol=5e-3)


def test_u_w_capacity_all_fields_affect_a_spillback_simulation():
    # The downstream bottleneck activates spillback, making the effect of w
    # observable in actual cumulative counts at this local working point.
    world = World(deltat=5, tmax=900, print_mode=0, save_mode=0, show_mode=0)
    for i in range(3):
        world.addNode(str(i), i, 0)
    world.addLink('A', '0', '1', length=1000, free_flow_speed=23.4,
                  backward_wave_speed=4.7, capacity=0.8)
    world.addLink('B', '1', '2', length=1000, free_flow_speed=20,
                  backward_wave_speed=5, capacity=0.31)
    world.adddemand('0', '2', 0, 900, 0.93)
    model = world.compile(backend='jax')
    fields = ('free_flow_speed', 'backward_wave_speed', 'capacity')
    refs = [model.parameter('A', field) for field in fields]
    variables = VariableSet(model, refs)
    objective = model.objective(lambda result: jnp.sum(result.link('A').cum_arrival()) +
                                             jnp.sum(result.link('B').cum_departure()))
    loss = jax.jit(objective._make_loss(variables))
    theta = variables.initial_theta()
    _, grad = objective.value_and_gradient(wrt=refs, jit=True)
    ad = np.asarray([grad[ref] for ref in refs])
    finite = []
    for i, h in enumerate((0.02, 0.01, 0.0005)):
        step = np.zeros(3, dtype=np.float32)
        step[i] = h
        finite.append((float(loss(theta + step)) - float(loss(theta - step))) / (2*h))
    assert np.all(np.abs(ad) > 1.0)
    np.testing.assert_allclose(ad, finite, rtol=5e-3)


def test_distinct_links_compose_with_interleaved_independent_fields():
    world = World(deltat=5, tmax=600, print_mode=0, save_mode=0, show_mode=0)
    for i in range(3):
        world.addNode(str(i), i, 0)
    world.addLink('A', '0', '1', length=1000, free_flow_speed=20,
                  backward_wave_speed=5, capacity=0.8)
    world.addLink('B', '1', '2', length=1000, free_flow_speed=20, jam_density=0.2)
    world.adddemand('0', '2', 0, 600, 0.3)
    model = world.compile(backend='jax')
    selected = [('A', 'free_flow_speed', 18.0), ('B', 'free_flow_speed', 14.0),
                ('A', 'backward_wave_speed', 4.5), ('B', 'jam_density', 0.25),
                ('A', 'capacity', 0.7), ('B', 'capacity_out', 0.4)]
    variables = VariableSet(model, [model.parameter(link, field) for link, field, _ in selected])
    params = variables.inject(jnp.array([value for _, _, value in selected]), model.raw.params)
    expected_a = manual_triplet('u_w_capacity', CASES['u_w_capacity'], model._snapshot_link(0).tau)
    expected_b = manual_triplet('u_kappa_tau', {'free_flow_speed':14.0, 'jam_density':0.25},
                               model._snapshot_link(1).tau)
    np.testing.assert_allclose(extract(params, 0), expected_a, rtol=1e-6)
    np.testing.assert_allclose(extract(params, 1), expected_b, rtol=1e-6)
    assert float(params.capacity_out[1]) == pytest.approx(0.4)


def test_joint_fd_commutes_with_independent_non_fd_ref():
    model = one_link_model()
    speed = model.parameter('road', 'free_flow_speed')
    density = model.parameter('road', 'jam_density')
    priority = model.parameter('road', 'merge_priority')
    variables = VariableSet(model, [speed, priority, density])
    params = variables.inject(jnp.array([18.0, 2.5, 0.25]), model.raw.params)
    np.testing.assert_allclose(extract(params), manual_triplet('u_kappa_tau', CASES['u_kappa_tau'],
                                                             model._snapshot_link(0).tau), rtol=1e-6)
    assert float(params.merge_priority[0]) == pytest.approx(2.5)


def test_joint_fd_refuses_ambiguous_groups():
    model = one_link_model()
    speed = model.parameter('road', 'free_flow_speed')
    density = model.parameter('road', 'jam_density')
    with pytest.raises(ValueError, match='Duplicate FD field'):
        VariableSet(model, [speed, speed])
    with pytest.raises(ValueError, match='Conflicting FD parameterizations'):
        VariableSet(model, [speed, replace(density, fd_parameterization='u_w_capacity')])
    derived = ParameterRef('link', 'capacity', 0, fd_parameterization='u_kappa_tau')
    with pytest.raises(ValueError, match='independent scalars'):
        VariableSet(model, [speed, derived])
    wrong_scheme_speed = replace(speed, fd_parameterization='u_w_tau')
    wrong_scheme_wave = ParameterRef('link', 'backward_wave_speed', 0, fd_parameterization='u_w_tau')
    with pytest.raises(ValueError, match='does not match link'):
        VariableSet(model, [wrong_scheme_speed, wrong_scheme_wave])
    custom = CustomVariable((), 2.0,
                            lambda params, x: params._replace(merge_priority=params.merge_priority.at[0].set(x)))
    with pytest.raises(ValueError, match='cannot be interleaved'):
        VariableSet(model, [speed, custom, density])


def test_custom_variable_before_or_after_joint_group_keeps_sequence():
    model = one_link_model()
    speed = model.parameter('road', 'free_flow_speed')
    density = model.parameter('road', 'jam_density')
    custom = CustomVariable((), 22.0, lambda params, x: params._replace(u=params.u.at[0].set(x)))
    before = VariableSet(model, [custom, speed, density])
    after = VariableSet(model, [speed, density, custom])
    params_before = before.inject(jnp.array([22.0, 18.0, 0.25]), model.raw.params)
    params_after = after.inject(jnp.array([18.0, 0.25, 22.0]), model.raw.params)
    expected = manual_triplet('u_kappa_tau', CASES['u_kappa_tau'], model._snapshot_link(0).tau)
    np.testing.assert_allclose(extract(params_before), expected, rtol=1e-6)
    np.testing.assert_allclose(extract(params_after), [22.0, expected[1], expected[2]], rtol=1e-6)
