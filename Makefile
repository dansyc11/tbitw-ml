CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -I./include
LIBS = -lm

SRC = src/tbitw_arena.c src/tbitw_math.c src/tbitw_nn.c

all: xor mnist mnist_adam mnist_inference mnist_predict

xor: $(SRC) examples/xor.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

mnist: $(SRC) src/tbitw_mnist.c examples/mnist.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

mnist_adam: $(SRC) src/tbitw_mnist.c examples/mnist_adam.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@


mnist_inference: $(SRC) src/tbitw_mnist.c examples/mnist_inference.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

mnist_predict: $(SRC) src/tbitw_mnist.c examples/mnist_predict.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

clean:
	rm -f xor mnist mnist_adam mnist_inference mnist_predict
.PHONY: all clean
