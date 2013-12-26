/*
 * language_network_defaults.h
 *
 *  Created on: 2013-12-23
 *      Author: Chen Pu
 */

#ifndef LANGUAGE_NETWORK_DEFAULTS_H_
#define LANGUAGE_NETWORK_DEFAULTS_H_

static const std::string training_default = "training.nn";
static const std::string huff_code_default = "huff_code.nn";
static const std::string huff_point_default = "huff_point.nn";
static const std::string point_feature_default ="point_feature.nn";
static const std::string word_feature_default = "word_feature.nn";
static const std::string binary_training_default = "neuralnet.tmp.training.bin";
static const std::string binary_window_default = "neuralnet.tmp.window.bin";
static const size_t grid_diag_default = 300;
static const size_t iteration_default = 30;
static const size_t validation_ratio_default = 10;
static const dim3 grid_dim_default(grid_diag_default,grid_diag_default,1);
static const size_t batch_size_default = grid_diag_default*grid_diag_default;
static const float learning_rate_default = 0.01;
static const unsigned int vocab_size_default = 8485;
static const unsigned int tree_size_default = 8484;
static const unsigned int feature_size_default = 128;

#endif /* LANGUAGE_NETWORK_DEFAULTS_H_ */
