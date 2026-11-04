# Link observations from cumulative counts

`unsim.observations` is an optional adapter for the JAX `SimResult` or raw
`SimState` and `NetworkConfig`. It uses the existing UNsim simulator outputs;
it does not change the simulation or infer a detector model.

```python
from unsim.observations import from_result

result = world.compile(backend="jax").run()
obs = from_result(result, sample_interval_s=300)
print(obs.flow_out_veh_s.shape)   # [complete windows, directed links]
print(obs.density_veh_m.shape)
print(obs.speed_valid)            # all false: speed is unsupported
```

For a static sample interval equal to an integer number of simulation steps,
each nonoverlapping observation window reports:

- `flow_in_veh_s`, `flow_out_veh_s`: changes of cumulative link entry and exit
  counts divided by window duration. These are **boundary crossing rates**,
  averaged over time, in vehicles/s.
- `vehicle_count_start`, `vehicle_count_end`: cumulative arrivals minus
  departures at the window boundaries, in vehicles.
- `density_veh_m`: the arithmetic mean of the right-endpoint inventory
  snapshots within the window, divided by link length, in vehicles/m. This is a
  time-average of link inventory density, not a point detector occupancy and not
  necessarily an exact Edie space-time density.
- `speed_m_s`: NaN with an all-false `speed_valid` mask. A downstream boundary
  flow and whole-link inventory do not identify space-mean or detector speed;
  the adapter deliberately does not call their ratio a measured speed.

Inventory is calculated as `cumulative_arrivals - cumulative_departures`.
For that difference to represent the true number of vehicles, both curves
must use compatible baselines: at the initial time their difference must equal
the initial link inventory. Native empty-start UNsim counts use a shared zero
baseline. Adding the same constant to both curves preserves inventory; resetting
the two counters independently does not. In general,
`inventory(t) = inventory(t0) + (arrivals(t)-arrivals(t0)) -
(departures(t)-departures(t0))`; unknown initial stock or independently
rebased external counts must be aligned before calling this adapter. Its
validity masks cannot detect an unknown but numerically plausible baseline.

Do not multiply the NaN speed placeholder by a zero mask in a loss: IEEE
arithmetic still returns NaN. Omit this unsupported channel or use a
mask-aware `where` before arithmetic.

`time_start_s` and `time_end_s` define each window. A final incomplete interval
is omitted and its length is returned as `discarded_tail_s`; a simulation too
short for even one complete window raises `ValueError`. There are separate
validity masks for inflow, outflow, start/end inventory and density. A genuinely
zero count or flow is valid. Nonfinite values, negative inventory, negative
within-window increments or nonpositive link length produce false masks; raw
values are not silently clipped to conceal a simulation or data issue.

The calculation uses JAX operations and preserves the gradient paths of the
existing simulator for continuous counts, provided the config and sample
interval are static. For example, an objective may access
`from_state(simulate(params, config), config, 60).density_veh_m`. This does not
make the whole traffic model globally smooth: the simulator has min/max and
capacity switches, discrete path selection, interpolation segment selection,
and some deliberate gradient stops. Gradients and finite differences can differ
at such boundaries. The examples `example_observation_gradient.py` and
`example_merge_nonsmooth_probe.py` give small CPU diagnostics, not general
accuracy or real-world calibration claims.

These gradients are within the JAX simulator and adapter. Passing arrays to a
separate PyTorch model does not automatically preserve derivatives across the
framework boundary; a correct cross-framework VJP or a single-framework objective
is required before claiming end-to-end differentiation through both.
