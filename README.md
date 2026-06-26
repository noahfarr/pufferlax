# pufferlax

A thin **JAX** wrapper for
[PufferLib](https://github.com/PufferAI/PufferLib)-Ocean (C) vectorized
environments.

PufferLib's Ocean envs are fast C simulators exposed through a small `_C`
vec-env ABI (`create_vec`, `cpu_step`, raw obs/reward/terminal buffers). pufferlax
drives that ABI from inside `jax.pure_callback`, so a C env pool looks like an
ordinary gymnax env: `reset` / `step` that compose with `vmap`, `jit`, and
`lax.scan`.

## Install

`pufferlax` builds against [PufferLib](https://github.com/PufferAI/PufferLib)
4.x, vendored as a git submodule and compiled per-env:

```bash
git clone --recurse-submodules git@github.com:noahfarr/pufferlax.git
cd pufferlax
uv sync --group build

cd vendor/pufferlib && uv run --project .. bash build.sh breakout --cpu
```

PufferLib 4.x links a single `pufferlib._C` extension to one env at build time,
so re-run `build.sh <env> --cpu` to switch envs. The `--cpu` (float32) backend is
required: `pufferlax` reads the raw buffers as `float32`, while the default build
is bf16.

## Use

pufferlax doesn't ship any environment — you register the module that exposes a
compiled PufferLib-Ocean `_C` extension, then `make` it:

```python
import jax
import pufferlax

pufferlax.register("breakout", "pufferlib")

env, params = pufferlax.make(
    "breakout",
    batch_shape=(8,),
    num_threads=4,
    frameskip=4, width=576, height=330,
    paddle_width=62, paddle_height=8, ball_width=32, ball_height=32,
    brick_width=32, brick_height=12, brick_rows=6, brick_cols=18,
    initial_ball_speed=256, max_ball_speed=448, paddle_speed=620, continuous=0,
)

key = jax.random.PRNGKey(0)
obs, state = jax.vmap(env.reset)(jax.random.split(key, env.num_envs))
actions = jax.random.randint(key, (env.num_envs,), 0, env.num_actions)
obs, state, reward, done, info = jax.vmap(env.step)(
    jax.random.split(key, env.num_envs), state, actions
)
```

Extra keyword args to `make` are forwarded to the C env as its config; the keys
an Ocean env needs are listed in pufferlib's `config/<env>.ini`.

See [`examples/`](examples/) for more:

- [`rollout.py`](examples/rollout.py) — a `jit`+`lax.scan` random rollout.
- [`ppo_breakout.py`](examples/ppo_breakout.py) — end-to-end PPO (adapted from
  [purejaxrl](https://github.com/luchris429/purejaxrl)) training on a pufferlax
  env pool, reporting episodic return and steps-per-second.

The PPO example needs a few extra packages (`distrax`, `flax`, `optax`); install
them with the `examples` extra, then run it:

```bash
uv sync --extra examples
uv run python examples/ppo_breakout.py
```

## How it works

- The C backend holds `num_envs = prod(batch_shape)` environments behind a single
  vec handle and steps them in parallel via `cpu_step`.
- The gymnax API is per-env; `vmap` over a leading axis of `batch_shape` drives
  the whole pool in one `pure_callback` (the env uses `vmap_method="broadcast_all"`).
- The C backend **auto-resets** terminated envs, so pufferlax overrides gymnax's
  public `reset`/`step` to skip the usual reset-on-done blend.

## Limitations

- **Single discrete action dimension.** `action_space` is a flat `Discrete`; the
  first entry of the env's `act_sizes` is used.
- **Shared pool / RNG.** A multi-axis `batch_shape` is one C pool sharing one
  RNG, so sub-batches are not independently seeded.
- **float32 observations.** Observations are read as a flat `float32` vector.
- **ABI.** Targets the PufferLib-Ocean vec-env ABI (`create_vec`, `obs_ptr`,
  `rewards_ptr`, `terminals_ptr`, `act_sizes`, `obs_size`, `reset`, `cpu_step`).
  If your pufferlib build differs, adjust the registered module accordingly.
