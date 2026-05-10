#include "../include/tbitw_arena.h"
#include "../include/tbitw_nn.h"
#include <time.h>
#include <unistd.h>

int main(void) {
    srand(time(0) ^ getpid());
    
    Arena arena = arena_create(1024 * 1024);
    
    f32 ti_data[] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    };
    f32 to_data[] = {
        0,
        1,
        1,
        0,
    };
    
    Mat ti = {ti_data, 4, 2};
    Mat to = {to_data, 4, 1};
    
    u32 arch[] = {2, 16, 1};
    NN nn = nn_alloc(&arena, arch, ARRAY_LEN(arch));
    NN grad = nn_alloc(&arena, arch, ARRAY_LEN(arch));
    Adam adam = adam_alloc(&arena, nn);
    
    // Xavier initialization
    for (u32 i = 0; i < nn.count; ++i) {
        f32 limit = sqrtf(6.0f / (f32)(nn.ws[i].rows + nn.ws[i].cols));
        mat_rand(nn.ws[i], -limit, limit);
        mat_fill(nn.bs[i], 0.0f);
    }
    
    f32 rate = 0.005f;  // Even Lower learning rate for Adam
    printf("Initial cost: %f\n", nn_cost(nn, ti, to));
    
    for (u32 i = 0; i < 50000; ++i) {
        nn_backprop(nn, grad, ti, to);
        adam_update(nn, grad, &adam, rate);
        
        if (i % 1000 == 0) {
            printf("Epoch %u: cost = %f\n", i, nn_cost(nn, ti, to));
        }
    }
    
    printf("\nFinal cost: %f\n", nn_cost(nn, ti, to));
    
    printf("\nPredictions:\n");
    for (u32 i = 0; i < 4; ++i) {
        NN_INPUT(nn).data[0] = ti.data[i * 2 + 0];
        NN_INPUT(nn).data[1] = ti.data[i * 2 + 1];
        nn_forward(nn);
        printf("%.0f XOR %.0f = %.4f (expected %.0f)\n",
               ti.data[i * 2 + 0], ti.data[i * 2 + 1],
               NN_OUTPUT(nn).data[0], to.data[i]);
    }
    
    arena_free(&arena);
    return 0;
}
