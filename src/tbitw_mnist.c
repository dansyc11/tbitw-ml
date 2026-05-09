#include "tbitw_mnist.h"

static u32 read_u32_be(FILE *f) {
    u8 buf[4];
    fread(buf, 1, 4, f);
    return ((u32)buf[0] << 24) | ((u32)buf[1] << 16) | 
           ((u32)buf[2] << 8) | (u32)buf[3];
}

MNIST_Dataset mnist_load(Arena *a, const char *images_path, const char *labels_path) {
    MNIST_Dataset dataset = {0};
    
    // Load images
    FILE *img_file = fopen(images_path, "rb");
    if (!img_file) {
        fprintf(stderr, "Failed to open %s\n", images_path);
        exit(1);
    }
    
    u32 magic = read_u32_be(img_file);
    u32 num_images = read_u32_be(img_file);
    u32 rows = read_u32_be(img_file);
    u32 cols = read_u32_be(img_file);
    
    printf("Loading %u images (%ux%u)...\n", num_images, rows, cols);
    
    dataset.count = num_images;
    dataset.images = mat_alloc(a, num_images, rows * cols);
    
    for (u32 i = 0; i < num_images; ++i) {
        for (u32 j = 0; j < rows * cols; ++j) {
            u8 pixel;
            fread(&pixel, 1, 1, img_file);
            dataset.images.data[i * (rows * cols) + j] = pixel / 255.0f;
        }
    }
    fclose(img_file);
    
    // Load labels
    FILE *lbl_file = fopen(labels_path, "rb");
    if (!lbl_file) {
        fprintf(stderr, "Failed to open %s\n", labels_path);
        exit(1);
    }
    
    magic = read_u32_be(lbl_file);
    u32 num_labels = read_u32_be(lbl_file);
    
    dataset.labels = mat_alloc(a, num_labels, 10);
    mat_fill(dataset.labels, 0.0f);
    
    for (u32 i = 0; i < num_labels; ++i) {
        u8 label;
        fread(&label, 1, 1, lbl_file);
        dataset.labels.data[i * 10 + label] = 1.0f;  // One-hot encoding
    }
    fclose(lbl_file);
    
    printf("Loaded %u images and labels\n", dataset.count);
    return dataset;
}

void mnist_print_digit(Mat image) {
    const char *chars = " .:-=+*#%@";
    for (u32 i = 0; i < 28; ++i) {
        for (u32 j = 0; j < 28; ++j) {
            f32 pixel = image.data[i * 28 + j];
            int idx = (int)(pixel * 9);
            printf("%c", chars[idx]);
        }
        printf("\n");
    }
}
