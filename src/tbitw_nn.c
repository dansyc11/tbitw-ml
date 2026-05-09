#include "tbitw_nn.h"

NN nn_alloc(Arena *a, u32 *arch, u32 arch_count) {
    NN nn;
    nn.count = arch_count - 1;
    
    nn.ws = arena_alloc(a, sizeof(Mat) * nn.count);
    nn.bs = arena_alloc(a, sizeof(Mat) * nn.count);
    nn.as = arena_alloc(a, sizeof(Mat) * (nn.count + 1));
    
    nn.as[0] = mat_alloc(a, 1, arch[0]);
    
    for (u32 i = 1; i < arch_count; ++i) {
        nn.ws[i-1] = mat_alloc(a, nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(a, 1, arch[i]);
        nn.as[i]   = mat_alloc(a, 1, arch[i]);
    }
    
    return nn;
}

void nn_rand(NN nn, f32 low, f32 high) {
    for (u32 i = 0; i < nn.count; ++i) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn) {
    for (u32 i = 0; i < nn.count; ++i) {
        mat_dot(nn.as[i], nn.ws[i], nn.as[i+1]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
}

f32 nn_cost(NN nn, Mat ti, Mat to) {
    f32 cost = 0.0f;
    u32 n = ti.rows;
    
    for (u32 i = 0; i < n; ++i) {
        Mat x = {ti.data + i * ti.cols, 1, ti.cols};
        Mat y = {to.data + i * to.cols, 1, to.cols};
        
        memcpy(NN_INPUT(nn).data, x.data, sizeof(f32) * x.cols);
        nn_forward(nn);
        
        for (u32 j = 0; j < to.cols; ++j) {
            f32 d = NN_OUTPUT(nn).data[j] - y.data[j];
            cost += d * d;
        }
    }
    
    return cost / n;
}

void nn_backprop(NN nn, NN grad, Mat ti, Mat to) {
    u32 n = ti.rows;
    
    for (u32 i = 0; i < nn.count; ++i) {
        mat_fill(grad.ws[i], 0);
        mat_fill(grad.bs[i], 0);
        mat_fill(grad.as[i], 0);
    }
    
    for (u32 i = 0; i < n; ++i) {
        Mat x = {ti.data + i * ti.cols, 1, ti.cols};
        Mat y = {to.data + i * to.cols, 1, to.cols};
        
        memcpy(NN_INPUT(nn).data, x.data, sizeof(f32) * x.cols);
        nn_forward(nn);
        
        for (u32 j = 0; j < to.cols; ++j) {
            grad.as[nn.count].data[j] = 2.0f * (NN_OUTPUT(nn).data[j] - y.data[j]);
        }
        
        for (i32 l = nn.count - 1; l >= 0; --l) {
            for (u32 j = 0; j < nn.as[l+1].cols; ++j) {
                f32 a = nn.as[l+1].data[j];
                f32 da = grad.as[l+1].data[j];
                f32 qa = a * (1 - a);
                
                grad.bs[l].data[j] += da * qa;
                
                for (u32 k = 0; k < nn.as[l].cols; ++k) {
                    f32 pa = nn.as[l].data[k];
                    f32 w = nn.ws[l].data[k * nn.ws[l].cols + j];
                    
                    grad.ws[l].data[k * grad.ws[l].cols + j] += pa * da * qa;
                    grad.as[l].data[k] += w * da * qa;
                }
            }
        }
    }
    
    for (u32 i = 0; i < grad.count; ++i) {
        for (u32 j = 0; j < grad.ws[i].rows * grad.ws[i].cols; ++j) {
            grad.ws[i].data[j] /= n;
        }
        for (u32 j = 0; j < grad.bs[i].cols; ++j) {
            grad.bs[i].data[j] /= n;
        }
    }
}

void nn_learn(NN nn, NN grad, f32 rate) {
    for (u32 i = 0; i < nn.count; ++i) {
        for (u32 j = 0; j < nn.ws[i].rows * nn.ws[i].cols; ++j) {
            nn.ws[i].data[j] -= rate * grad.ws[i].data[j];
        }
        for (u32 j = 0; j < nn.bs[i].cols; ++j) {
            nn.bs[i].data[j] -= rate * grad.bs[i].data[j];
        }
    }
}

void nn_print(NN nn, const char *name) {
    printf("%s = [\n", name);
    for (u32 i = 0; i < nn.count; ++i) {
        printf("  Layer %u:\n", i);
        mat_print(nn.ws[i], "    W");
        mat_print(nn.bs[i], "    b");
    }
    printf("]\n");
}
