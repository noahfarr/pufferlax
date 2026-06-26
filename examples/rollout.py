import functools

import jax
import jax.numpy as jnp

import pufferlax

pufferlax.register("breakout", "pufferlib")
env, params = pufferlax.make(
    "breakout",
    batch_shape=(64,),
    num_threads=8,
    frameskip=4, width=576, height=330,
    paddle_width=62, paddle_height=8, ball_width=32, ball_height=32,
    brick_width=32, brick_height=12, brick_rows=6, brick_cols=18,
    initial_ball_speed=256, max_ball_speed=448, paddle_speed=620, continuous=0,
)


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

    _, (rewards, dones) = jax.lax.scan(step, (obs, state, key), None, length=steps)
    return rewards, dones


rewards, dones = rollout(jax.random.PRNGKey(0), 512)
print("summed reward:", float(rewards.sum()), "episodes:", int(dones.sum()))
