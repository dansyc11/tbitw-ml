#ifndef TBITW_NN_H
#define TBITW_NN_H

#include "tbitw_math.h"

typedef struct {
    Mat *ws;
    Mat *bs;
    Mat *as;
    u32 count;
} NN;

// Adam optimizer state
typedef struct {
    Mat *m_w;  // momentum for weights
    Mat *v_w;  // velocity for weights
    Mat *m_b;  // momentum for biases
    Mat *v_b;  // velocity for biases
    u32 t;     // timestep
} Adam;

NN nn_alloc(Arena *a, u32 *arch, u32 arch_count);
void nn_rand(NN nn, f32 low, f32 high);
void nn_forward(NN nn);
f32  nn_cost(NN nn, Mat ti, Mat to);
void nn_backprop(NN nn, NN grad, Mat ti, Mat to);
void nn_learn(NN nn, NN grad, f32 rate);

// Adam optimizer functions
Adam adam_alloc(Arena *a, NN nn);
void adam_update(NN nn, NN grad, Adam *adam, f32 lr);

#define NN_INPUT(nn)  (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

#endif // TBITW_NN_H
