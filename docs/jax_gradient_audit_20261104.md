# CPU gradient and observation audit — 2026-11-04

This is a small development audit of source revision
`0251a4fd2eeb8192e391ff3d9c45d973606db418`, not a validation of arbitrary
networks or real-road calibration. It used a separate Python 3.12 CPU environment
with JAX/JAXLIB 0.12.0.1, `JAX_PLATFORMS=cpu`, one-CPU affinity and one numerical
thread. The TimeBase training environment was not upgraded. No GPU was used.

Five representative upstream tests covering demand, free-speed and merge
gradients, one finite-difference regression and multiple API parameter references
passed. The full upstream `TestMergeAD` and `TestMergeFD` classes also passed
their 12 tests. Pytest emitted two class-scoped fixture deprecation warnings. A
passing upstream regression does **not** establish AD=FD: `_check_reg` permits
20% relative deviation and the AD and FD cases are checked independently
against different stored numbers rather than compared directly.

The added count-based observation tests exercise known-window units, masks,
tail handling, JIT, the public result wrapper and an actual simulator gradient
against finite differences. Speed is deliberately unsupported; see
[observations.md](observations.md). The independent one-link example, at demand
0.31 vehicles/s and free speed 23.4 m/s, gave:

| Derivative of 60-second inventory objective | JAX AD | Central FD, delta 0.01 |
| --- | ---: | ---: |
| Demand | 12,820.508 | 12,821.130 |
| Free speed | -169.845 | -169.739 |

The objective was 3,974.345 vehicle-seconds. The corresponding relative
differences are about 0.0049% and 0.062%. When free speed is changed, the probe
also updates the capacity implied by the link's `(u, kappa, tau)` fundamental
diagram; keeping derived capacity fixed would answer a different question. This
agreement concerns one selected operating point only.

The second probe reproduces the upstream merge example and distinguishes two
potential nonsmooth cases. Perturbing **only originally positive demand times**
at equal merge priorities gave AD 332,500.59 versus central FD 392,335.94 at
delta 0.002. Changing the second priority from 1 to 2 gave AD 392,500.78 versus
central FD 392,363.28. When all demand time slots are perturbed at priority 2,
forward/backward differences were approximately 504,492/392,203: negative
perturbations create negative values in otherwise zero-demand slots, while the
origin flow uses nonnegative clamps. This supports a boundary/subgradient
interpretation; it is not by itself proof of an implementation defect. Both
experiments are useful counterexamples to interpreting a passing regression as
global smoothness.

Reproduction, from this repository, using an already isolated JAX CPU environment:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' \
  taskset -c 0 python -m pytest tests/test_observations.py \
  tests/test_diff.py::TestNumericalAgreement::test_1link_freeflow -q

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' \
  taskset -c 0 python examples/example_observation_gradient.py

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' \
  taskset -c 0 python examples/example_merge_nonsmooth_probe.py
```

`taskset -c 0` is suitable only if CPU 0 is allowed by the caller's affinity
mask. The examples accept `--output` to save machine-readable results. The
observed numerical values can depend on JAX version, precision, and whether a
sample lies at a discrete traffic-model boundary.
