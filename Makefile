CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -I./include
LIBS = -lm

SRC = src/tbitw_arena.c src/tbitw_math.c src/tbitw_nn.c

all: xor mnist

xor: $(SRC) examples/xor.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

mnist: $(SRC) src/tbitw_mnist.c examples/mnist.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

clean:
	rm -f xor mnist

.PHONY: all clean
