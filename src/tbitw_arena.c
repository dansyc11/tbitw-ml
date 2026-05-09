#include "tbitw_arena.h"

Arena arena_create(usize size) {
    Arena a = {0};
    a.memory = malloc(size);
    a.size = size;
    a.used = 0;
    return a;
}

void *arena_alloc(Arena *a, usize size) {
    if (a->used + size > a->size) {
        fprintf(stderr, "Arena out of memory\n");
        exit(1);
    }
    
    void *ptr = a->memory + a->used;
    a->used += size;
    memset(ptr, 0, size);
    
    return ptr;
}

void arena_reset(Arena *a) {
    a->used = 0;
}

void arena_free(Arena *a) {
    free(a->memory);
    a->memory = NULL;
    a->size = 0;
    a->used = 0;
}
