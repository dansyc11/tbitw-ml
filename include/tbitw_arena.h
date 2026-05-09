#ifndef TBITW_ARENA_H
#define TBITW_ARENA_H

#include "tbitw_base.h"

typedef struct {
    u8 *memory;
    usize size;
    usize used;
} Arena;

Arena arena_create(usize size);
void *arena_alloc(Arena *a, usize size);
void  arena_reset(Arena *a);
void  arena_free(Arena *a);

#endif // TBITW_ARENA_H
