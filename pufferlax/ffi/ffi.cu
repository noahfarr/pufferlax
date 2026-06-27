#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" {

typedef struct {
    const char* key;
    double value;
    void* pointer;
} Item;

typedef struct {
    Item* items;
    int size;
    int capacity;
} Dict;

struct StaticThreading;

typedef struct StaticVec {
    void* environments;
    int size;
    int total_agents;
    int buffers;
    int agents_per_buffer;
    int* buffer_env_starts;
    int* buffer_env_counts;
    void* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    void* gpu_observations;
    float* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    unsigned char* gpu_action_mask;
    cudaStream_t* streams;
    StaticThreading* threading;
    int observation_size;
    int num_actions;
    int action_mask_size;
    int gpu;
    int* agent_permutation;
} StaticVec;

StaticVec* create_static_vec(int total_agents, int num_buffers, int gpu,
                             Dict* vec_kwargs, Dict* env_kwargs);
void static_vec_reset(StaticVec* vec);
void static_vec_close(StaticVec* vec);
void cpu_vec_step(StaticVec* vec);
size_t get_obs_elem_size(void);

}

static inline Dict* make_dict(int capacity) {
    Dict* dict = (Dict*)calloc(1, sizeof(Dict));
    dict->capacity = capacity;
    dict->items = (Item*)calloc(capacity, sizeof(Item));
    return dict;
}

static inline void dict_set(Dict* dict, const char* key, double value) {
    dict->items[dict->size].key = key;
    dict->items[dict->size].value = value;
    dict->size++;
}

extern "C" {

void* puffer_create(int total_agents, double seed_offset, double reset_pool_size) {
    cudaSetDevice(0);
    Dict* vec_kwargs = make_dict(4);
    dict_set(vec_kwargs, "total_agents", (double)total_agents);
    dict_set(vec_kwargs, "num_buffers", 1.0);
    Dict* env_kwargs = make_dict(4);
    dict_set(env_kwargs, "seed_offset", seed_offset);
    dict_set(env_kwargs, "reset_pool_size", reset_pool_size);
    StaticVec* vec = create_static_vec(total_agents, 1, 0, vec_kwargs, env_kwargs);
    static_vec_reset(vec);
    return (void*)vec;
}

void puffer_close(void* handle) {
    static_vec_close((StaticVec*)handle);
}

int puffer_observation_size(void* handle) {
    return ((StaticVec*)handle)->observation_size;
}

int puffer_num_actions(void* handle) {
    return ((StaticVec*)handle)->num_actions;
}

int puffer_total_agents(void* handle) {
    return ((StaticVec*)handle)->total_agents;
}

long long puffer_observation_element_size() {
    return (long long)get_obs_elem_size();
}

}

static ffi::Error step(
    ffi::Buffer<ffi::DataType::F32> actions,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> observations,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> rewards,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> terminals,
    int64_t env_handle,
    cudaStream_t stream) {
    StaticVec* vec = reinterpret_cast<StaticVec*>(static_cast<uintptr_t>(env_handle));
    const int num_agents = vec->total_agents;
    const size_t observation_bytes = (size_t)num_agents * vec->observation_size * sizeof(float);
    const size_t action_bytes = (size_t)num_agents * vec->num_actions * sizeof(float);

    cudaMemcpyAsync(vec->actions, actions.typed_data(), action_bytes,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cpu_vec_step(vec);

    cudaMemcpyAsync(observations->typed_data(), vec->observations, observation_bytes,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(rewards->typed_data(), vec->rewards, (size_t)num_agents * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(terminals->typed_data(), vec->terminals, (size_t)num_agents * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    return ffi::Error::Success();
}

static ffi::Error reset(
    ffi::Buffer<ffi::DataType::F32>,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> observations,
    int64_t env_handle,
    cudaStream_t stream) {
    StaticVec* vec = reinterpret_cast<StaticVec*>(static_cast<uintptr_t>(env_handle));
    static_vec_reset(vec);
    const size_t observation_bytes = (size_t)vec->total_agents * vec->observation_size * sizeof(float);
    cudaMemcpyAsync(observations->typed_data(), vec->observations, observation_bytes,
                    cudaMemcpyHostToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PufferStep, step,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Attr<int64_t>("env_handle")
        .Ctx<ffi::PlatformStream<cudaStream_t>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PufferReset, reset,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Attr<int64_t>("env_handle")
        .Ctx<ffi::PlatformStream<cudaStream_t>>());
