// Inference-only float32 geometry for the pinned Merizo IPA configuration.
#include <cuda_runtime.h>
#include <cstddef>

__global__ void point_logits(const float* q, const float* k, const float* weight,
                             float* out, int n, int heads) {
    const int key_head = blockIdx.x * blockDim.x + threadIdx.x;
    if (key_head >= n * heads) return;
    const int h = key_head % heads;
    const size_t query = static_cast<size_t>(blockIdx.z) * n + blockIdx.y;
    const size_t key = static_cast<size_t>(blockIdx.z) * n * heads + key_head;
    const float* qi = q + (query * heads + h) * 12;
    const float* kj = k + key * 12;
    const float w = weight[h];
    float term[4];
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float dx = qi[p*3] - kj[p*3];
        const float dy = qi[p*3+1] - kj[p*3+1];
        const float dz = qi[p*3+2] - kj[p*3+2];
        const float squared = (dx * dx + dy * dy) + dz * dz;
        term[p] = squared * w;
    }
    // PyTorch sums its four contiguous points in adjacent pairs.
    out[query * n * heads + key_head] = ((term[0] + term[1]) + (term[2] + term[3])) * -0.5f;
}

extern "C" int merizo_point_logits(const float* q, const float* k, const float* weight,
                                   float* out, int batch, int n, int heads, void* stream) {
    dim3 grid((n * heads + 127) / 128, n, batch);
    point_logits<<<grid, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(q, k, weight, out, n, heads);
    return static_cast<int>(cudaGetLastError());
}

// PyTorch 2.0.1's CUDA sum for this strided dimension uses four accumulators.
// At length 256 it additionally divides the input across 16 reduction lanes.
template<int lanes>
__global__ void weighted_points(const float* attention, const float* values, float* out,
                                int n, int outputs) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = tid / lanes;
    if (index >= outputs) return;
    const int lane = threadIdx.x % lanes;
    const int component = index % 24;  // point * 3 + coordinate
    const int query = index / 24 % n;
    const int batch_head = index / 24 / n;
    const float* a = attention + (static_cast<size_t>(batch_head) * n + query) * n;
    const float* v = lanes == 1
        ? values + static_cast<size_t>(batch_head) * n * 24 + component
        : values + (static_cast<size_t>(batch_head) * 24 + component) * n;
    const int stride = lanes == 1 ? 24 : 1;
    float partial[4] = {};
    for (int start = lane; start < n; start += 4 * lanes) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int key = start + item * lanes;
            if (key < n) partial[item] += a[key] * v[key * stride];
        }
    }
    float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
    #pragma unroll
    for (int offset = lanes / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(__activemask(), sum, offset, lanes);
    }
    if (lane == 0) {
        const int coord = component % 3, point = component / 3;
        out[((static_cast<size_t>(batch_head) * 3 + coord) * n + query) * 8 + point] = sum;
    }
}

extern "C" int merizo_weighted_points(const float* a, const float* v, float* out,
                                      int batch, int n, int heads, void* stream) {
    const int outputs = batch * heads * n * 24;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (n < 256) {
        weighted_points<1><<<(outputs + 127) / 128, 128, 0, cuda_stream>>>(a, v, out, n, outputs);
    } else {
        weighted_points<16><<<(outputs * 16 + 127) / 128, 128, 0, cuda_stream>>>(a, v, out, n, outputs);
    }
    return static_cast<int>(cudaGetLastError());
}
