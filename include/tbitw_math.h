#ifndef TBITW_MATH_H
#define TBITW_MATH_H

#include "tbitw_base.h"
#include "tbitw_arena.h"

typedef struct {
    f32 *data;
    u32 rows;
    u32 cols;
} Mat;

Mat mat_alloc(Arena *a, u32 rows, u32 cols);
void mat_fill(Mat m, f32 value);
void mat_rand(Mat m, f32 low, f32 high);
void mat_dot(Mat a, Mat b, Mat dst);
void mat_sum(Mat a, Mat b);
void mat_print(Mat m, const char *name);
void mat_rand_xavier(Mat m);
f32  sigmoidf(f32 x);
f32  reluf(f32 x);
void mat_sig(Mat m);
void mat_relu(Mat m);

#endif // TBITW_MATH_H
