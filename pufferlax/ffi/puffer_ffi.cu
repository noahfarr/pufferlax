// pufferlax GPU FFI: step any pufferlib (Ocean) vec env in-graph from JAX.
// The env compute stays on CPU (that is what pufferlib's "gpu" path does too);
// this handler bridges XLA device buffers <-> the env's host buffers on XLA's
// stream, with no Python round-trip and no jnp.asarray host materialization.
//
// Generic: the only env-specific thing is which static lib it links against.
// Built per-env (link libstatic_<env>.a) from a single generic source.

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// ---- pufferlib ABI (replicated from vendor/pufferlib/src/vecenv.h, NOT #included
//      because that header's step implementations need env-specific OBS_SIZE macros) ----
extern "C" {

typedef struct { const char* key; double value; void* ptr; } DictItem;
typedef struct { DictItem* items; int size; int capacity; } Dict;

struct StaticThreading;

typedef struct StaticVec {
    void* envs;
    int size;
    int total_agents;
    int buffers;
    int agents_per_buffer;
    int* buffer_env_starts;
    int* buffer_env_counts;
    void* observations;            // HOST obs
    float* actions;                // HOST actions
    float* rewards;                // HOST rewards
    float* terminals;              // HOST terminals
    unsigned char* action_mask;
    void* gpu_observations;
    float* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    unsigned char* gpu_action_mask;
    cudaStream_t* streams;
    StaticThreading* threading;
    int obs_size;
    int num_atns;
    int action_mask_size;
    int gpu;
    int* agent_perm;
} StaticVec;

// env entry points, implemented in libstatic_<env>.a
StaticVec* create_static_vec(int total_agents, int num_buffers, int gpu,
                             Dict* vec_kwargs, Dict* env_kwargs);
void static_vec_reset(StaticVec* vec);
void static_vec_close(StaticVec* vec);
void cpu_vec_step(StaticVec* vec);
int get_obs_size(void);
int get_num_atns(void);
size_t get_obs_elem_size(void);

}  // extern "C"

static inline Dict* puf_dict(int cap) {
    Dict* d = (Dict*)calloc(1, sizeof(Dict));
    d->capacity = cap;
    d->items = (DictItem*)calloc(cap, sizeof(DictItem));
    return d;
}
static inline void puf_set(Dict* d, const char* k, double v) {
    d->items[d->size].key = k;
    d->items[d->size].value = v;
    d->size++;
}

// ================= ctypes surface (env lifecycle, called from Python host) =================
extern "C" {

void* puffer_create(int total_agents, double seed_offset, double reset_pool_size) {
    cudaSetDevice(0);
    Dict* vk = puf_dict(4);
    puf_set(vk, "total_agents", (double)total_agents);
    puf_set(vk, "num_buffers", 1.0);
    Dict* ek = puf_dict(4);
    puf_set(ek, "seed_offset", seed_offset);
    puf_set(ek, "reset_pool_size", reset_pool_size);
    StaticVec* vec = create_static_vec(total_agents, 1, /*gpu=*/0, vk, ek);
    static_vec_reset(vec);
    return (void*)vec;
}
void puffer_close(void* h) { static_vec_close((StaticVec*)h); }
int puffer_obs_size(void* h) { return ((StaticVec*)h)->obs_size; }
int puffer_num_atns(void* h) { return ((StaticVec*)h)->num_atns; }
int puffer_total_agents(void* h) { return ((StaticVec*)h)->total_agents; }
long long puffer_obs_elem_size() { return (long long)get_obs_elem_size(); }

}  // extern "C"

// ================= XLA FFI handlers (run on XLA's CUDA stream, in-graph) =================

static ffi::Error StepImpl(
    ffi::Buffer<ffi::DataType::F32> actions,                 // Arg0: f32 actions (cast in JAX)
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> obs_out,    // Ret0
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> reward_out, // Ret1
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> done_out,   // Ret2
    int64_t env_handle,                                      // Attr
    cudaStream_t stream) {                                   // Ctx
    StaticVec* vec = reinterpret_cast<StaticVec*>(static_cast<uintptr_t>(env_handle));
    const int N = vec->total_agents;
    const size_t obs_bytes = (size_t)N * vec->obs_size * sizeof(float);
    const size_t act_bytes = (size_t)N * vec->num_atns * sizeof(float);

    // D2H actions onto the env's host buffer, then sync (the CPU step reads host actions).
    cudaMemcpyAsync(vec->actions, actions.typed_data(), act_bytes,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cpu_vec_step(vec);  // CPU env step

    // H2D obs/reward/done back onto XLA's device output buffers, on XLA's stream.
    cudaMemcpyAsync(obs_out->typed_data(), vec->observations, obs_bytes,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(reward_out->typed_data(), vec->rewards, (size_t)N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(done_out->typed_data(), vec->terminals, (size_t)N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    return ffi::Error::Success();
}

static ffi::Error ResetImpl(
    ffi::Buffer<ffi::DataType::F32> /*seed: ignored, present so vmap can size the batch*/,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> obs_out,
    int64_t env_handle,
    cudaStream_t stream) {
    StaticVec* vec = reinterpret_cast<StaticVec*>(static_cast<uintptr_t>(env_handle));
    static_vec_reset(vec);
    const size_t obs_bytes = (size_t)vec->total_agents * vec->obs_size * sizeof(float);
    cudaMemcpyAsync(obs_out->typed_data(), vec->observations, obs_bytes,
                    cudaMemcpyHostToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PufferStep, StepImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Attr<int64_t>("env_handle")
        .Ctx<ffi::PlatformStream<cudaStream_t>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PufferReset, ResetImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Attr<int64_t>("env_handle")
        .Ctx<ffi::PlatformStream<cudaStream_t>>());
