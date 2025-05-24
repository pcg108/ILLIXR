#ifndef RITNET_H
#define RITNET_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void gemmini_inference(int8_t image_input[1][160][240][1], float eye_x, float eye_y); // put actual signature here

#ifdef __cplusplus
}
#endif

#endif

