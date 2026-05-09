#include "tbitw_math.h"

Mat mat_alloc(Arena *a, u32 rows, u32 cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = arena_alloc(a, sizeof(f32) * rows * cols);
    return m;
}

void mat_fill(Mat m, f32 value) {
    for (u32 i = 0; i < m.rows * m.cols; ++i) {
        m.data[i] = value;
    }
}

void mat_rand(Mat m, f32 low, f32 high) {
    for (u32 i = 0; i < m.rows * m.cols; ++i) {
        m.data[i] = low + (f32)rand() / RAND_MAX * (high - low);
    }
}

void mat_dot(Mat a, Mat b, Mat dst) {
    for (u32 i = 0; i < a.rows; ++i) {
        for (u32 j = 0; j < b.cols; ++j) {
            f32 sum = 0.0f;
            for (u32 k = 0; k < a.cols; ++k) {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            dst.data[i * dst.cols + j] = sum;
        }
    }
}

void mat_sum(Mat a, Mat b) {
    for (u32 i = 0; i < a.rows * a.cols; ++i) {
        a.data[i] += b.data[i];
    }
}

void mat_print(Mat m, const char *name) {
    printf("%s = [\n", name);
    for (u32 i = 0; i < m.rows; ++i) {
        printf("  ");
        for (u32 j = 0; j < m.cols; ++j) {
            printf("%8.4f ", m.data[i * m.cols + j]);
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_rand_xavier(Mat m) {
    f32 limit = sqrtf(6.0f / (f32)(m.rows + m.cols));
    for (u32 i = 0; i < m.rows * m.cols; ++i) {
        m.data[i] = -limit + 2.0f * limit * (f32)rand() / RAND_MAX;
    }
}

f32 sigmoidf(f32 x) {
    return 1.0f / (1.0f + expf(-x));
}

f32 reluf(f32 x) {
    return x > 0 ? x : 0;
}

void mat_sig(Mat m) {
    for (u32 i = 0; i < m.rows * m.cols; ++i) {
        m.data[i] = sigmoidf(m.data[i]);
    }
}

void mat_relu(Mat m) {
    for (u32 i = 0; i < m.rows * m.cols; ++i) {
        m.data[i] = reluf(m.data[i]);
    }
}
