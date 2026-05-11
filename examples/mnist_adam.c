#include "../include/tbitw_arena.h"
#include "../include/tbitw_nn.h"
#include "../include/tbitw_mnist.h"
#include <time.h>
#include <unistd.h>
// MNIST with Adam Optimizer, should be better

int main(void) {
    srand(time(0) ^ getpid());
    
    Arena arena = arena_create(1024 * 1024 * 1024);  // 1GB
    
    // Load MNIST
    MNIST_Dataset train = mnist_load(&arena, "data/train-images-idx3-ubyte", 
                                              "data/train-labels-idx1-ubyte");
    MNIST_Dataset test = mnist_load(&arena, "data/t10k-images-idx3-ubyte", 
                                             "data/t10k-labels-idx1-ubyte");
    
    // Create network: 784 -> 128 -> 10
    u32 arch[] = {784, 128, 10};
    NN nn = nn_alloc(&arena, arch, ARRAY_LEN(arch));
    NN grad = nn_alloc(&arena, arch, ARRAY_LEN(arch));
    Adam adam = adam_alloc(&arena, nn);
    
    // Xavier initialization
    for (u32 i = 0; i < nn.count; ++i) {
        f32 limit = sqrtf(6.0f / (f32)(nn.ws[i].rows + nn.ws[i].cols));
        mat_rand(nn.ws[i], -limit, limit);
        mat_fill(nn.bs[i], 0.0f);
    }
    
    printf("\nTraining neural network...\n");
    printf("Architecture: 784 -> 128 -> 10\n\n");
    
    f32 rate = 0.001f;
    u32 epochs = 20;
    u32 batch_size = 100;
    
    for (u32 epoch = 0; epoch < epochs; ++epoch) {
        // Train on batches
        for (u32 batch = 0; batch < train.count / batch_size; ++batch) {
            u32 start = batch * batch_size;
            
            Mat batch_imgs = {train.images.data + start * 784, batch_size, 784};
            Mat batch_lbls = {train.labels.data + start * 10, batch_size, 10};
            
            nn_backprop(nn, grad, batch_imgs, batch_lbls);
            adam_update(nn, grad, &adam, rate);
        }
        
        // Test accuracy
        u32 correct = 0;
        for (u32 i = 0; i < test.count; ++i) {
            Mat img = {test.images.data + i * 784, 1, 784};
            memcpy(NN_INPUT(nn).data, img.data, sizeof(f32) * 784);
            nn_forward(nn);
            
            // Find predicted class
            u32 pred = 0;
            f32 max_val = NN_OUTPUT(nn).data[0];
            for (u32 j = 1; j < 10; ++j) {
                if (NN_OUTPUT(nn).data[j] > max_val) {
                    max_val = NN_OUTPUT(nn).data[j];
                    pred = j;
                }
            }
            
            // Find true class
            u32 true_label = 0;
            for (u32 j = 0; j < 10; ++j) {
                if (test.labels.data[i * 10 + j] > 0.5f) {
                    true_label = j;
                    break;
                }
            }
            
            if (pred == true_label) correct++;
        }
        
        f32 accuracy = 100.0f * correct / test.count;
        printf("Epoch %u: Accuracy = %.2f%% (%u/%u)\n", 
               epoch + 1, accuracy, correct, test.count);
    }

    printf("\nSave trained model? (y/n): ");
    char response;
    scanf(" %c", &response);

    if (response == 'y' || response == 'Y') {
	char filename[256];
	printf("Enter filename (e.g., my_model.tbitw): ");
	scanf("%255s", filename);
	nn_save(nn, filename);
	printf("✓ Saved as %s\n", filename);
    } else {
    printf("Model not saved\n");
    }
} 
