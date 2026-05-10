#include "../include/tbitw_arena.h"
#include "../include/tbitw_nn.h"
#include "../include/tbitw_mnist.h"

void print_digit(Mat image) {
    const char *chars = " .:oO@";
    printf("\n");
    for (u32 i = 0; i < 28; ++i) {
        for (u32 j = 0; j < 28; ++j) {
            f32 pixel = image.data[i * 28 + j];
            int idx = (int)(pixel * 5);
            printf("%c", chars[idx]);
        }
        printf("\n");
    }
}

int main(void) {
    Arena arena = arena_create(512 * 1024 * 1024);
    
    u32 arch[] = {784, 128, 10};
    NN nn = nn_load(&arena, "mnist_model.tbitw", arch, ARRAY_LEN(arch));
    
    MNIST_Dataset test = mnist_load(&arena, "data/t10k-images-idx3-ubyte", 
                                             "data/t10k-labels-idx1-ubyte");
    
    while (1) {
        printf("\n=== MNIST Digit Classifier ===\n");
        printf("Enter image index (0-%u) or -1 to quit: ", test.count - 1);
        
        int idx;
        scanf("%d", &idx);
        
        if (idx < 0) break;
        if (idx >= (int)test.count) {
            printf("Invalid index!\n");
            continue;
        }
        
        Mat img = {test.images.data + idx * 784, 1, 784};
        
        u32 true_label = 0;
        for (u32 j = 0; j < 10; ++j) {
            if (test.labels.data[idx * 10 + j] > 0.5f) {
                true_label = j;
                break;
            }
        }
        
        memcpy(NN_INPUT(nn).data, img.data, sizeof(f32) * 784);
        nn_forward(nn);
        
        u32 pred = 0;
        f32 max_confidence = NN_OUTPUT(nn).data[0];
        for (u32 j = 1; j < 10; ++j) {
            if (NN_OUTPUT(nn).data[j] > max_confidence) {
                max_confidence = NN_OUTPUT(nn).data[j];
                pred = j;
            }
        }
        
        print_digit(img);
        printf("\nPrediction: %u (%.1f%% confidence)\n", pred, max_confidence * 100);
        printf("True label: %u\n", true_label);
        printf("%s\n", (pred == true_label) ? "CORRECT!" : "WRONG!");
        
        printf("\nAll predictions:\n");
        for (u32 j = 0; j < 10; ++j) {
            printf("  %u: ", j);
            int bar_length = (int)(NN_OUTPUT(nn).data[j] * 50);
            for (int k = 0; k < bar_length; ++k) printf("=");
            printf(" %.1f%%\n", NN_OUTPUT(nn).data[j] * 100);
        }
    }
    
    arena_free(&arena);
    return 0;
}
