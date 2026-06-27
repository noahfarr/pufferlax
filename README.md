# pufferlax

A thin **JAX** wrapper for
[PufferLib](https://github.com/PufferAI/PufferLib)-Ocean (C) vectorized
environments.

PufferLib's Ocean envs are fast C simulators. pufferlax steps them **in-graph**
via an XLA FFI custom call on JAX's CUDA stream — no `pure_callback`, no host
round-trip — so a C env pool looks like an ordinary gymnax env: `reset` / `step`
that compose with `vmap`, `jit`, and `lax.scan`.

## Install

pufferlax vendors [PufferLib](https://github.com/PufferAI/PufferLib) as a git
submodule and compiles a small FFI shared library per env:

```bash
git clone --recurse-submodules git@github.com:noahfarr/pufferlax.git
cd pufferlax
uv sync
```

The per-env FFI library is built on first use, or explicitly with
`python -m pufferlax.build_ffi craftax`. The build needs only `g++` and the CUDA
runtime that ships with `jax[cuda12]` — no CUDA toolkit or `nvcc` — so it also
runs on a login node with no GPU. A CUDA GPU is required at run time: the env
step is an FFI custom call registered on the CUDA platform.

## Use

```python
import jax
import pufferlax

env, params = pufferlax.make("craftax", batch_shape=(8,), num_threads=4)

key = jax.random.PRNGKey(0)
obs, state = jax.vmap(env.reset)(jax.random.split(key, env.num_envs))
actions = jax.random.randint(key, (env.num_envs,), 0, env.num_actions)
obs, state, reward, done, info = jax.vmap(env.step)(
    jax.random.split(key, env.num_envs), state, actions
)
```

`make` accepts `seed_offset` and `reset_pool_size` (forwarded to the C env) and
`num_threads` (the C pool's worker count). See
[`examples/rollout.py`](examples/rollout.py) for a `jit`+`lax.scan` random
rollout.

## How it works

- The C backend holds `num_envs = prod(batch_shape)` environments behind a single
  vec handle and steps them in parallel on CPU worker threads.
- `reset` / `step` issue an XLA FFI custom call: actions are copied to the env's
  host buffers on JAX's stream, the C env steps, and obs/reward/terminal are
  copied back to the device output buffers — all in-graph, no host materialization.
- The gymnax API is per-env; `vmap` over a leading axis of `batch_shape` drives
  the whole pool in one FFI call (`vmap_method="broadcast_all"`).
- The C backend **auto-resets** terminated envs, so pufferlax overrides gymnax's
  public `reset`/`step` to skip the usual reset-on-done blend.

## Limitations

- **CUDA only.** The FFI handler is registered on the CUDA platform.
- **Single discrete action dimension.** `action_space` is a flat `Discrete`; the
  first entry of the env's `act_sizes` is used.
- **Shared pool / RNG.** A multi-axis `batch_shape` is one C pool sharing one
  RNG, so sub-batches are not independently seeded.
- **float32 observations.** Observations are read as a flat `float32` vector.
