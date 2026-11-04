"""Audit fixed-merge AD and finite differences at candidate nonsmooth points.

The upstream regression tests separately check AD and FD numbers; this example
compares them under both full-time-axis and positive-demand-only perturbations.
It changes no simulator internals and does not claim all discrepancies are bugs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import jaxlib
from unsim import World
from unsim.unsim_diff import world_to_jax, simulate, total_travel_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError("The probe is restricted to a CPU JAX backend")
    world = World(name="", deltat=5, tmax=2000, print_mode=0, save_mode=0, show_mode=0)
    for name, x, y in (("orig1", 0, 0), ("orig2", 0, 2), ("merge", 1, 1), ("dest", 2, 1)):
        world.addNode(name, x, y)
    for name, start, end in (("link1", "orig1", "merge"), ("link2", "orig2", "merge"),
                             ("link3", "merge", "dest")):
        world.addLink(name, start, end, length=1000, free_flow_speed=20, jam_density=.2, merge_priority=1)
    world.adddemand("orig1", "dest", 0, 1000, .45)
    world.adddemand("orig2", "dest", 400, 1000, .6)
    params, config = world_to_jax(world)
    full = jnp.ones_like(params.demand_rate[0])
    active = (params.demand_rate[0] > 0).astype(full.dtype)

    def loss(delta, mask, priority2):
        rate = params.demand_rate.at[0, :].add(delta * mask)
        priorities = params.merge_priority.at[1].set(priority2)
        p = params._replace(demand_rate=rate, merge_priority=priorities)
        return total_travel_time(simulate(p, config), config)

    forward = jax.jit(loss)
    value_grad = jax.jit(jax.value_and_grad(loss, argnums=0))
    results = {}
    for label, mask in (("all_time_steps", full), ("positive_demand_steps_only", active)):
        for priority in (1., 2.):
            value, derivative = value_grad(0., mask, priority)
            baseline = float(value)
            row = {"base_vehicle_seconds": baseline, "autodiff": float(derivative), "differences": {}}
            for delta in (.01, .005, .002, .001):
                plus, minus = float(forward(delta, mask, priority)), float(forward(-delta, mask, priority))
                row["differences"][str(delta)] = {"centered": (plus - minus) / (2 * delta),
                                                   "forward": (plus - baseline) / delta,
                                                   "backward": (baseline - minus) / delta}
            results[f"{label}_priority2_{priority}"] = row
    root = Path(__file__).resolve().parents[1]
    report = {"backend": jax.default_backend(), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "perturbation": "Add delta to origin-1 demand either across all times or only originally positive times",
              "warning": "Full-time negative perturbation makes initially zero demand negative; the simulator clips flows. "
                         "Priority ties and min/max switches can select different subgradients. "
                         "Differences do not by themselves prove a simulator bug.",
              "cases": results}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
