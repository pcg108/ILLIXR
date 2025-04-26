#ifndef RITNET_H
#define RITNET_H

#include "gemmini.h"
#include "ritnet_params.h"
#include "ritnet_weights.h"
#include "ritnet_helpers.h"

void gemmini_inference(elem_t * images, float eye_x, float eye_y);


#endif