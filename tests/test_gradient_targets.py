import jax
import jax.numpy as jnp
import numpy as np
import pytest

from unsim import World
from unsim.api import VariableSet
from unsim.gradient_targets import _targets_math, evaluate_objectives, gradient_targets


def test_live_row_median_clip_scaling_and_normalized_rms():
    parameters = np.full((4,2),.5,dtype=np.float32)
    gradients = np.array([[1,0],[2,0],[100,0],[0,0]],dtype=np.float32)
    batch = gradient_targets(parameters,gradients,0.,1.,delta=.1,kappa=1.5)
    np.testing.assert_allclose(batch.row_norms,[1,2,100,0])
    np.testing.assert_allclose(batch.median_norm,2)
    np.testing.assert_allclose(batch.threshold,3)
    np.testing.assert_allclose(batch.row_scales,[1,1,.03,1],rtol=1e-6)
    np.testing.assert_allclose(batch.clipped_gradients,[[1,0],[2,0],[3,0],[0,0]],rtol=1e-6)
    np.testing.assert_allclose(batch.clipped_fraction,1/3,rtol=1e-6)
    raw_update = np.asarray(batch.normalized_targets)-parameters
    np.testing.assert_allclose(np.sqrt(np.square(raw_update[:3]).mean()),.1,rtol=1e-6)
    np.testing.assert_array_equal(batch.live,[True,True,True,False])


def test_projection_physical_units_zero_rows_and_invalid_inputs():
    parameters = np.array([[.2,15.],[.9,18.]],dtype=np.float32)
    gradients = np.array([[0.,-100.],[100.,0.]],dtype=np.float32)
    result = gradient_targets(parameters,gradients,[0.,10.],[1.,20.],delta=10.,kappa=100.)
    np.testing.assert_allclose(result.normalized_gradients,[[0,-1000],[100,0]])
    np.testing.assert_allclose(result.targets,[[.2,20.],[0.,18.]],rtol=1e-6)
    zero = gradient_targets(parameters,np.zeros_like(parameters),[0.,10.],[1.,20.])
    np.testing.assert_array_equal(zero.targets,parameters)
    assert float(zero.median_norm) == 0 and float(zero.step_multiplier) == 0
    assert np.isinf(float(zero.threshold))
    with pytest.raises(ValueError,match='finite'):
        gradient_targets(parameters,np.full_like(parameters,np.nan),[0.,10.],[1.,20.])
    with pytest.raises(ValueError,match='declared box'):
        gradient_targets(parameters,gradients,[.3,10.],[1.,20.])
    with pytest.raises(ValueError,match='strictly less'):
        gradient_targets(parameters,gradients,[0.,20.],[1.,10.])
    with pytest.raises(FloatingPointError,match='overflowed'):
        gradient_targets([[.5]],[[3e38]],[0.],[1.])


def test_clip_is_per_row_permutation_equivariant_but_batch_dependent():
    gradients = np.array([[1.,0.],[2.,0.],[20.,0.],[0.,0.]],dtype=np.float32)
    theta = np.full_like(gradients,.5)
    original = gradient_targets(theta,gradients,0.,1.,delta=.01,kappa=1.)
    perm = np.array([3,1,0,2])
    moved = gradient_targets(theta[perm],gradients[perm],0.,1.,delta=.01,kappa=1.)
    np.testing.assert_allclose(moved.targets,original.targets[perm],rtol=1e-6)
    np.testing.assert_allclose(moved.row_scales,original.row_scales[perm],rtol=1e-6)
    np.testing.assert_allclose(original.clipped_gradients,
                               gradients*np.asarray(original.row_scales)[:,None])
    with_zeros = gradient_targets(np.concatenate([theta,np.full((20,2),.5)]),
                                  np.concatenate([gradients,np.zeros((20,2))]),0.,1.,delta=.01,kappa=1.)
    np.testing.assert_allclose(with_zeros.threshold,original.threshold)
    np.testing.assert_allclose(with_zeros.targets[:4],original.targets)
    pair = gradient_targets(np.full((2,1),.5),[[1.],[2.]],0.,1.,kappa=1.)
    batch = gradient_targets(np.full((4,1),.5),[[1.],[2.],[100.],[100.]],0.,1.,kappa=1.)
    assert float(pair.row_scales[1]) < 1 and float(batch.row_scales[1]) == 1


def test_target_construction_stops_gradient_and_jax_core_jits():
    x = jnp.array([[.3,.7],[.4,.6]])
    g = jnp.array([[1.,2.],[3.,4.]])
    low,high = jnp.zeros(2),jnp.ones(2)
    calculate = lambda v:_targets_math(v,g,low,high,.05,3.,1e-30).targets
    np.testing.assert_allclose(jax.grad(lambda z:jnp.sum(calculate(z)))(x),0.,atol=0)
    np.testing.assert_allclose(jax.jit(calculate)(x),calculate(x),atol=1e-7)


def test_direct_jax_sgd_matches_projected_gradient_target():
    from jax.example_libraries import optimizers
    theta = np.array([[.18],[.25],[.42],[.31]],dtype=np.float32)
    gradient = np.array([[-6561.],[-2551.],[5103.],[-2551.]],dtype=np.float32)
    batch = gradient_targets(theta,gradient,[.05],[.65],delta=.05,kappa=3.)
    initial = (jnp.asarray(theta)-.05)/.6
    init,update,get = optimizers.sgd(float(batch.step_multiplier))
    state = update(0,batch.clipped_gradients,init(initial))
    np.testing.assert_allclose(jnp.clip(get(state),0.,1.),batch.normalized_targets,atol=1e-7)


@pytest.fixture(scope='module')
def toy():
    world = World(name='',deltat=5,tmax=300,print_mode=0)
    world.addNode('o',0,0)
    world.addNode('d',1,0)
    world.addLink('road','o','d',length=300,free_flow_speed=20,jam_density=.2)
    demand = world.adddemand('o','d',0,200,.22)
    model = world.compile(backend='jax')
    ref = model.parameter(demand,'flow')
    vset = VariableSet(model,[ref])
    truth_params = vset.inject(jnp.array([.36]),model.raw.params)
    truth_state = model.raw.simulate(truth_params,model.raw.config,differentiable=False)
    # This synthetic past observation at 150 s is available before the
    # arbitrary forecast anchor at 200 s; no future real observations exist.
    observed = jax.lax.stop_gradient(truth_state.cum_departure[0,30])
    objective = model.objective(lambda result:(result.link('road').cum_departure()[30]-observed)**2)
    return model,ref,objective


def test_real_unsim_ad_matches_finite_difference_and_bounded_step_lowers_loss(toy):
    model,ref,objective = toy
    before = np.array([[.22],[.221],[.219]],dtype=np.float32)
    values,grads = evaluate_objectives(model,[ref],[objective]*3,before)
    base_value,base_gradient = objective.value_and_gradient(wrt=[ref])
    np.testing.assert_allclose(values[0],base_value,rtol=1e-6)
    np.testing.assert_allclose(grads[0,0],base_gradient[ref],rtol=1e-6)
    finite_difference = (float(values[1])-float(values[2]))/.002
    np.testing.assert_allclose(float(grads[0,0]),finite_difference,rtol=2e-3,atol=.1)
    target = gradient_targets(before[:1],grads[:1],[.05],[.65],delta=.05,kappa=3.)
    after,after_grad = evaluate_objectives(model,[ref],[objective],target.targets)
    assert .05 <= float(target.targets[0,0]) <= .65
    assert float(after[0]) < float(values[0])
    assert np.isfinite(np.asarray(after_grad)).all()
    # Objective/parameter snapshots remain immutable after all candidate calls.
    np.testing.assert_allclose(VariableSet(model,[ref]).initial_theta(),[.22],rtol=1e-6)


def test_objective_batch_rejects_cross_model_and_bad_shapes(toy):
    model,ref,objective = toy
    with pytest.raises(ValueError,match='one objective'):
        evaluate_objectives(model,[ref],[objective],[[.22],[.23]])
    with pytest.raises(ValueError,match='distinct'):
        evaluate_objectives(model,[ref,ref],[objective],[[.22,.23]])
    with pytest.raises(ValueError,match='supplied model'):
        evaluate_objectives(model,[ref],[object()],[[.22]])
