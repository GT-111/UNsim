"""CPU UNsim observation-gradient probe against centered finite differences.

This is a small one-link simulator check, not a real-road calibration or proof of
global smoothness. Run with JAX installed and, for a one-CPU experiment, pin the
process affinity externally. The new export does not infer speed from q/k.
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
from unsim.observations import from_state
from unsim.unsim_diff import world_to_jax, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError("The probe is restricted to a CPU JAX backend")
    world = World(name="", deltat=5, tmax=600, print_mode=0, save_mode=0, show_mode=0)
    world.addNode("origin", 0, 0)
    world.addNode("destination", 1, 0)
    world.addLink("road", "origin", "destination", length=1000, free_flow_speed=23.4, jam_density=.2)
    world.adddemand("origin", "destination", 0, 300, .31)
    params, config = world_to_jax(world)

    def loss(demand, free_speed):
        # Preserve the link's (u,kappa,tau) parameterization when u changes.
        demand_rate = params.demand_rate * (demand / .31)
        w, jam = 5., .2  # 1 / (reaction_time * jam_density), both held fixed
        capacity = free_speed * w * jam / (free_speed + w)
        p = params._replace(demand_rate=demand_rate, u=params.u.at[0].set(free_speed),
                            q_star=params.q_star.at[0].set(capacity))
        obs = from_state(simulate(p, config), config, sample_interval_s=60)
        # Right-endpoint inventory approximation, units vehicle-seconds.
        return jnp.sum(obs.density_veh_m) * 1000 * 60

    objective = jax.jit(loss)
    derivative = jax.jit(jax.grad(loss, argnums=(0, 1)))
    q, u = .31, 23.4
    value = float(objective(q, u))
    grad_q, grad_u = map(float, derivative(q, u))
    finite = {}
    for delta in (.01, .005, .002):
        finite[str(delta)] = {
            "demand": (float(objective(q + delta, u)) - float(objective(q - delta, u))) / (2 * delta),
            "free_speed": (float(objective(q, u + delta)) - float(objective(q, u - delta))) / (2 * delta),
        }
    capacity = float(params.q_star[0])
    eps = .002
    at_capacity = float(objective(capacity, u))
    near_capacity = {
        "demand_veh_s": capacity,
        "autodiff_demand": float(derivative(capacity, u)[0]),
        "forward_difference": (float(objective(capacity + eps, u)) - at_capacity) / eps,
        "backward_difference": (at_capacity - float(objective(capacity - eps, u))) / eps,
        "interpretation": "One-sided numeric diagnostic near capacity; unequal values alone do not establish a kink.",
    }
    root = Path(__file__).resolve().parents[1]
    result = {"backend": jax.default_backend(), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "observation_source_sha256": hashlib.sha256((root / "unsim/observations.py").read_bytes()).hexdigest(),
              "objective": "sum of 60s right-endpoint link inventory, vehicle-seconds",
              "demand_veh_s": q, "free_speed_m_s": u, "objective_value": value,
              "autodiff": {"demand": grad_q, "free_speed": grad_u},
              "centered_finite_difference": finite, "near_capacity": near_capacity,
              "limitations": "One graph and operating point; route switches, min/max and discrete indexing are not globally smooth."}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
