/*
 * language_network_constants.h
 *
 *  Created on: 2013-12-22
 *      Author: Chen Pu
 */

/*
 * Global constants to simplify kernel call API
 */
#ifndef LANGUAGE_NETWORK_CONSTANTS_H_
#define LANGUAGE_NETWORK_CONSTANTS_H_
#include "language_network_typedefs.h"
#include <cuda.h>

//static const unsigned int FEATURESIZE = 128;
static const window_t MAXWINDOW = 7;
static const unsigned int VOCABSIZE = 8485;
static const unsigned int TREESIZE = 8484;
static const indx_t GRIDDIAG = 300;
static const size_t VALIDATIONDIAG = 300;
static const dim3 GRIDDIM(GRIDDIAG,GRIDDIAG,1);
static const dim3 VALIDDIM(VALIDATIONDIAG,VALIDATIONDIAG,1);
static const real LEARNING_RATE = 0.01;
static const size_t BUFFERSIZE = 80000000;
#endif /* LANGUAGE_NETWORK_CONSTANTS_H_ */
