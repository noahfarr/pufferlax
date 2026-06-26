import functools
import os
import time

import jax
import tqdx.rich

import pufferlax

pufferlax.register("craftax", "pufferlib")
env, params = pufferlax.make("craftax", batch_shape=(8192,), num_threads=os.cpu_count())


@functools.partial(jax.jit, static_argnums=(1,))
def rollout(key, steps):
    reset_keys = jax.random.split(key, env.num_envs)
    obs, state = jax.vmap(env.reset)(reset_keys)

    def step(carry, _):
        obs, state, key = carry
        key, sub = jax.random.split(key)
        actions = jax.random.randint(sub, (env.num_envs,), 0, env.num_actions)
        step_keys = jax.random.split(sub, env.num_envs)
        obs, state, reward, done, _ = jax.vmap(env.step)(step_keys, state, actions)
        return (obs, state, key), (reward, done)

    _, (rewards, dones) = tqdx.rich.scan(step, (obs, state, key), None, length=steps)
    return rewards, dones


steps = 2048
key = jax.random.PRNGKey(0)
jax.block_until_ready(rollout(key, steps))

start = time.perf_counter()
rewards, dones = jax.block_until_ready(rollout(key, steps))
elapsed = time.perf_counter() - start

total_steps = env.num_envs * steps
print(f"SPS: {total_steps / elapsed:,.0f} ({total_steps:,} steps in {elapsed:.3f}s)")
