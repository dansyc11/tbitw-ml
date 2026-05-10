#include "../include/tbitw_arena.h"
#include "../include/tbitw_nn.h"
#include "../include/tbitw_mnist.h"

int main(void) {
    Arena arena = arena_create(512* 1024 * 1024);
    
    u32 arch[] = {784, 128, 10};
    NN nn = nn_load(&arena, "mnist_model.tbitw", arch, ARRAY_LEN(arch));
    
    MNIST_Dataset test = mnist_load(&arena, "data/t10k-images-idx3-ubyte", 
                                             "data/t10k-labels-idx1-ubyte");
    
    u32 correct = 0;
    for (u32 i = 0; i < test.count; ++i) {
        Mat img = {test.images.data + i * 784, 1, 784};
        memcpy(NN_INPUT(nn).data, img.data, sizeof(f32) * 784);
        nn_forward(nn);
        
        u32 pred = 0;
        f32 max_val = NN_OUTPUT(nn).data[0];
        for (u32 j = 1; j < 10; ++j) {
            if (NN_OUTPUT(nn).data[j] > max_val) {
                max_val = NN_OUTPUT(nn).data[j];
                pred = j;
            }
        }
        
        u32 true_label = 0;
        for (u32 j = 0; j < 10; ++j) {
            if (test.labels.data[i * 10 + j] > 0.5f) {
                true_label = j;
                break;
            }
        }
        
        if (pred == true_label) correct++;
    }
    
    printf("Loaded model accuracy: %.2f%% (%u/%u)\n", 
           100.0f * correct / test.count, correct, test.count);
    
    arena_free(&arena);
    return 0;
}
