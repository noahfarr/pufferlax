import ctypes
import importlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces

REGISTRY: dict[str, str] = {}


def register(env_name: str, module_path: str) -> None:
    REGISTRY[env_name] = module_path


def load_C(env_name: str):
    if env_name not in REGISTRY:
        raise ValueError(
            f"{env_name!r} is not registered; known: {sorted(REGISTRY)}. "
            "Call pufferlax.register(env_name, module_path) first."
        )
    importlib.import_module(f"{REGISTRY[env_name]}._C")
    module = importlib.import_module(REGISTRY[env_name])
    assert hasattr(module, "_C")
    return module._C


@struct.dataclass
class PufferlaxState(environment.EnvState):
    time: int = 0


@struct.dataclass
class PufferlaxParams(environment.EnvParams):
    pass


class PufferlaxEnv(environment.Environment):
    def __init__(self, C, batch_shape=(1,), num_threads: int = 16, **env_kwargs):
        super().__init__()
        self._C = C
        self.env_name = C.env_name
        self.batch_shape = tuple(batch_shape)
        self.num_envs = int(np.prod(self.batch_shape))

        if len(self.batch_shape) > 1:
            warnings.warn(
                f"PufferlaxEnv batch_shape={self.batch_shape} treats leading axes as "
                "one env pool sharing a single C vec env and its RNG, so "
                "sub-batches are not independently seeded.",
                stacklevel=2,
            )

        self.vec = C.create_vec(
            {
                "vec": {
                    "total_agents": self.num_envs,
                    "num_buffers": 1,
                    "num_threads": int(num_threads),
                },
                "env": dict(env_kwargs),
            },
            0,
        )
        self.obs_size = int(self.vec.obs_size)
        self._num_actions, *_ = self.vec.act_sizes
        self.obs_shape = (self.obs_size,)
        self.obs_dtype = jnp.float32

        self._obs = self._view(self.vec.obs_ptr, self.num_envs * self.obs_size).reshape(
            self.num_envs, self.obs_size
        )
        self._rewards = self._view(self.vec.rewards_ptr, self.num_envs)
        self._terminals = self._view(self.vec.terminals_ptr, self.num_envs)
        self._actions = np.zeros(self.num_envs, dtype=np.float32)

        self.vec.reset()

    @staticmethod
    def _view(ptr: int, count: int) -> np.ndarray:
        return np.ctypeslib.as_array((ctypes.c_float * count).from_address(ptr))

    @property
    def name(self) -> str:
        return self.env_name

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def default_params(self) -> PufferlaxParams:
        return PufferlaxParams()

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=self.obs_shape, dtype=self.obs_dtype
        )

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(self._num_actions)

    def reset(self, key, params=None):
        return self.reset_env(key, params)

    def step(self, key, state, action, params=None):
        return self.step_env(key, state, action, params)

    def reset_env(self, key, params=None):
        def _reset(key):
            self.vec.reset()
            return jnp.asarray(
                self._obs.reshape(self.batch_shape + self.obs_shape),
                dtype=self.obs_dtype,
            )

        obs = jax.pure_callback(
            _reset,
            jax.ShapeDtypeStruct(self.obs_shape, self.obs_dtype),
            key,
            vmap_method="broadcast_all",
        )
        return obs, PufferlaxState(time=0)

    def step_env(self, key, state, action, params=None):
        def _step(action):
            self._actions[:] = np.asarray(action, dtype=np.float32).reshape(-1)
            self.vec.cpu_step(self._actions.ctypes.data)
            return (
                jnp.asarray(
                    self._obs.reshape(self.batch_shape + self.obs_shape),
                    dtype=self.obs_dtype,
                ),
                jnp.asarray(self._rewards.reshape(self.batch_shape), dtype=jnp.float32),
                jnp.asarray(
                    self._terminals.reshape(self.batch_shape) > 0.5, dtype=jnp.bool_
                ),
            )

        result_specs = (
            jax.ShapeDtypeStruct(self.obs_shape, self.obs_dtype),
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
        obs, reward, done = jax.pure_callback(
            _step, result_specs, action, vmap_method="broadcast_all"
        )
        return obs, PufferlaxState(time=state.time + 1), reward, done, {}


def make(
    env_name: str = "craftax", batch_shape=(1,), num_threads: int = 16, **env_kwargs
):
    C = load_C(env_name)
    if C.env_name != env_name:
        raise RuntimeError(
            f"{REGISTRY[env_name]}._C is compiled for {C.env_name!r}, "
            f"expected {env_name!r}."
        )
    env = PufferlaxEnv(C, batch_shape=batch_shape, num_threads=num_threads, **env_kwargs)
    return env, env.default_params
