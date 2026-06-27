import ctypes
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces

_PACKAGE = Path(__file__).resolve().parent
_BUILD_SCRIPT = _PACKAGE.parent / "build_ffi.py"
_REGISTERED: set[str] = set()


def _load(env_name: str):
    library_path = _PACKAGE / f"ffi_{env_name}.so"
    if not library_path.exists():
        subprocess.run([sys.executable, str(_BUILD_SCRIPT), env_name], check=True)
    library = ctypes.CDLL(str(library_path))
    library.puffer_create.restype = ctypes.c_void_p
    library.puffer_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double
    ]
    library.puffer_close.argtypes = [ctypes.c_void_p]
    library.puffer_observation_size.restype = ctypes.c_int
    library.puffer_observation_size.argtypes = [ctypes.c_void_p]
    library.puffer_num_actions.restype = ctypes.c_int
    library.puffer_num_actions.argtypes = [ctypes.c_void_p]
    if env_name not in _REGISTERED:
        jax.ffi.register_ffi_target(
            f"puffer_step_{env_name}",
            jax.ffi.pycapsule(library.PufferStep),
            platform="CUDA",
            api_version=1,
        )
        jax.ffi.register_ffi_target(
            f"puffer_reset_{env_name}",
            jax.ffi.pycapsule(library.PufferReset),
            platform="CUDA",
            api_version=1,
        )
        _REGISTERED.add(env_name)
    return library


@struct.dataclass
class PufferLibState(environment.EnvState):
    time: int = 0


class PufferLibEnv(environment.Environment):
    def __init__(
        self,
        env_name: str = "craftax",
        batch_shape=(1,),
        num_threads: int = 16,
        seed_offset: int = 0,
        reset_pool_size: int = 0,
    ):
        super().__init__()
        self.env_name = env_name
        self.batch_shape = tuple(batch_shape)
        self.num_envs = int(np.prod(self.batch_shape))
        self._library = _load(env_name)
        self._handle = self._library.puffer_create(
            self.num_envs, int(num_threads), float(seed_offset), float(reset_pool_size)
        )
        self.obs_size = int(self._library.puffer_observation_size(self._handle))
        self._num_actions = int(self._library.puffer_num_actions(self._handle))
        self.obs_shape = (self.obs_size,)
        self.obs_dtype = jnp.float32

    @property
    def name(self) -> str:
        return self.env_name

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def default_params(self) -> None:
        return None

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
        seed = jax.random.bits(key).astype(jnp.float32)
        obs = jax.ffi.ffi_call(
            f"puffer_reset_{self.env_name}",
            jax.ShapeDtypeStruct(self.obs_shape, self.obs_dtype),
            has_side_effect=True,
            vmap_method="broadcast_all",
        )(seed, env_handle=np.int64(self._handle))
        return obs, PufferLibState(time=0)

    def step_env(self, key, state, action, params=None):
        result = (
            jax.ShapeDtypeStruct(self.obs_shape, self.obs_dtype),
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.float32),
        )
        obs, reward, terminal = jax.ffi.ffi_call(
            f"puffer_step_{self.env_name}",
            result,
            has_side_effect=True,
            vmap_method="broadcast_all",
        )(action.astype(jnp.float32), env_handle=np.int64(self._handle))
        return obs, PufferLibState(time=state.time + 1), reward, terminal > 0.5, {}


def make(
    env_name: str = "craftax",
    batch_shape=(1,),
    num_threads: int = 16,
    seed_offset: int = 0,
    reset_pool_size: int = 0,
):
    env = PufferLibEnv(
        env_name,
        batch_shape=batch_shape,
        num_threads=num_threads,
        seed_offset=seed_offset,
        reset_pool_size=reset_pool_size,
    )
    return env, env.default_params
