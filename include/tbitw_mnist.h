#ifndef TBITW_MNIST_H
#define TBITW_MNIST_H

#include "tbitw_math.h"

typedef struct {
    Mat images;
    Mat labels;
    u32 count;
} MNIST_Dataset;

MNIST_Dataset mnist_load(Arena *a, const char *images_path, const char *labels_path);
void mnist_print_digit(Mat image);

#endif // TBITW_MNIST_H
