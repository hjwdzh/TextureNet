#include <iostream>

__device__ int compute_layer(float tx, float ty, float radius) {
    int x = int(std::abs(tx) / radius + 0.5);
    int y = int(std::abs(ty) / radius + 0.5);
    if (x == 1 && y == 1)
        return 2;
    int c = 0;
    if (x + y < 2)
        c = x + y;
    else
        c = x + y + 1;
    if (c > 5)
        c = 5;
    return c;
}

__global__ void five_kernel(int batch_size, int num_points, int num_featdim, int num_neighbors, int num_groups, int num_feat_per_threads,
    const float* points, const float* tex, const int* idx, float* out, float radius)
{
    int b = blockIdx.y;
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= num_groups)
        return;
    int feat_start = num_feat_per_threads * blockIdx.z;
    int feat_end = feat_start + num_feat_per_threads;
    if (feat_end >= num_featdim)
        feat_end = num_featdim;
    const int* group_array = idx + (b * num_groups + group_idx) * num_neighbors;
    const float* tex_array = tex + 2 * num_points * b;
    const float* points_array = points + (b * num_points * num_featdim);
    float* out_array = out + (b * num_groups + group_idx) * (6 * num_featdim);
    int layers_counts[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < num_neighbors; ++i) {
        int index = group_array[i];
        if (tex_array[index * 2] > 1e20)
            continue;
        int layer = compute_layer(tex_array[index * 2], tex_array[index * 2 + 1], radius);
        layers_counts[layer] += 1;
    }

    for (int i = 0; i < num_neighbors; ++i) {
        int index = group_array[i];
        if (tex_array[index * 2] > 1e20)
            continue;
        int layer = compute_layer(tex_array[index * 2], tex_array[index * 2 + 1], radius);
        float* out_temp = out_array + layer * num_featdim;

        const float* point_temp = points_array + index * num_featdim;
        for (int j = feat_start; j < feat_end; ++j) {
            out_temp[j] += point_temp[j] / layers_counts[layer];
        }
    }
    for (int i = 0; i < 6; ++i) {
        if (layers_counts[i] == 0) {
            int front = i;
            int rear = i;
            float weight_front = 0.0f;
            float weight_rear = 0.0f;
            while (front >= 0 && layers_counts[front] == 0)
                front -= 1;
            while (rear < 6 && layers_counts[rear] == 0)
                rear += 1;
            if (front >= 0 && rear < 6) {
                weight_rear = (i - front) / (rear - front + 0.0f);
                weight_front = 1.0f - weight_rear;
            }
            else if (front >= 0) {
                weight_front = 1.0f;
                weight_rear = 0.0f;
                rear = 5;
            }
            else {
                weight_front = 0.0f;
                weight_rear = 1.0f;
                front = 0;
            }
            float* out_temp = out_array + i * num_featdim;
            float* out_front = out_array + front * num_featdim;
            float* out_rear = out_array + rear * num_featdim;
            for (int j = feat_start; j < feat_end; ++j) {
                out_temp[j] = out_front[j] * weight_front + out_rear[j] * weight_rear;
            }
        }
    }
}

__global__ void fivegrad_kernel(int batch_size, int num_points, int num_featdim, int num_neighbors, int num_groups, int num_feat_per_threads,
    const float* points, const float* tex, const int* idx, const float* grad_out, float* grad_points, float radius)
{
    int b = blockIdx.y;
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= num_groups)
        return;
    int feat_start = num_feat_per_threads * blockIdx.z;
    int feat_end = feat_start + num_feat_per_threads;
    if (feat_end >= num_featdim)
        feat_end = num_featdim;
    const int* group_array = idx + (b * num_groups + group_idx) * num_neighbors;
    const float* tex_array = tex + 2 * num_points * b;
    float* points_array = grad_points + (b * num_points * num_featdim);
    const float* out_array = grad_out + (b * num_groups + group_idx) * (6 * num_featdim);
    int layers_counts[6] = {0, 0, 0, 0, 0, 0};
    float weights_front[6] = {0, 0, 0, 0, 0, 0};
    float weights_rear[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < num_neighbors; ++i) {
        int index = group_array[i];
        if (tex_array[index * 2] > 1e20)
            continue;
        int layer = compute_layer(tex_array[index * 2], tex_array[index * 2 + 1], radius);
        layers_counts[layer] += 1;
    }

    for (int i = 0; i < 6; ++i) {
        if (layers_counts[i] == 0) {
            int front = i;
            int rear = i;
            float weight_front = 0.0f;
            float weight_rear = 0.0f;
            while (front >= 0 && layers_counts[front] == 0)
                front -= 1;
            while (rear < 6 && layers_counts[rear] == 0)
                rear += 1;
            if (front >= 0 && rear < 6) {
                weight_rear = (i - front) / (rear - front + 0.0f);
                weight_front = 1.0f - weight_rear;
            }
            else if (front >= 0) {
                weight_front = 1.0f;
                weight_rear = 0.0f;
                rear = 5;
            }
            else {
                weight_front = 0.0f;
                weight_rear = 1.0f;
                front = 0;
            }
            weights_front[i] = weight_front;
            weights_rear[i] = weight_rear;
        }
    }

    for (int i = 0; i < num_neighbors; ++i) {
        int index = group_array[i];
        if (tex_array[index * 2] > 1e20)
            continue;
        int layer = compute_layer(tex_array[index * 2], tex_array[index * 2 + 1], radius);
        const float* out_temp = out_array + layer * num_featdim;
        float* point_temp = points_array + index * num_featdim;
        for (int j = feat_start; j < feat_end; ++j) {
            float signal = out_temp[j];
            int l = layer - 1;
            const float* out_temp_step = out_temp - num_featdim;
            while (l >= 0 && layers_counts[l] == 0) {
                signal += out_temp_step[j] * weights_rear[l];
                out_temp_step -= num_featdim;
                l -= 1;
            }
            l = layer + 1;
            out_temp_step = out_temp + num_featdim;
            while (l < 6 && layers_counts[l] == 0) {
                signal += out_temp_step[j] * weights_front[l];
                out_temp_step += num_featdim;
                l += 1;
            }
            atomicAdd(&point_temp[j], signal / layers_counts[layer]);
        }
    }
}

void fivekernel_gpu(int batch_size, int num_points, int num_featdim, int num_neighbors, int num_groups, const float* points, const float* tex, const int* idx, float* out, float radius) {
    int num_threads_for_feat = (num_groups + 255) / num_groups;
    int num_feat_per_threads = (num_featdim + num_threads_for_feat - 1) / num_threads_for_feat;
    five_kernel<<<dim3((num_groups + 255) / 256, batch_size, num_threads_for_feat), dim3(256, 1, 1)>>>(batch_size, num_points, num_featdim, num_neighbors, num_groups, num_feat_per_threads, points, tex, idx, out, radius);
}

void fivekernelgrad_gpu(int batch_size, int num_points, int num_featdim, int num_neighbors, int num_groups, const float* points, const float* tex, const int* idx, const float* grad_out, float* grad_points, float radius) {
    int num_threads_for_feat = (num_groups + 255) / num_groups;
    int num_feat_per_threads = (num_featdim + num_threads_for_feat - 1) / num_threads_for_feat;
    fivegrad_kernel<<<dim3((num_groups + 255) / 256, batch_size, num_threads_for_feat), dim3(256, 1, 1)>>>(
        batch_size, num_points, num_featdim, num_neighbors, num_groups, num_feat_per_threads, points, tex, idx, grad_out, grad_points, radius);
    
}