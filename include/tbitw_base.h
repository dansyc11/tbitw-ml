#ifndef TBITW_BASE_H
#define TBITW_BASE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t  i32;
typedef float    f32;
typedef size_t   usize;

#define ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))

#endif // TBITW_BASE_H
