/*
 * language_network_utility.h
 *
 *  Created on: 2013-12-22
 *      Author: Chen Pu
 */

#ifndef LANGUAGE_NETWORK_UTILITY_H_
#define LANGUAGE_NETWORK_UTILITY_H_
#include "language_network_constants.h"
#include "language_network_typedefs.h"
#include "language_network_kernel.h"


struct options_mutable
{
	std::string training;
	std::string huff_code;
	std::string huff_point;
	std::string point_feature;
	std::string word_feature;
	size_t batch_size;
	size_t iteration;
	size_t validation_ratio;
	std::string binary_training;
	std::string binary_window;
	size_t grid_diagnal;
	dim3 grid_dim;
	float learning_rate;
	size_t feature_size;
	size_t tree_size;
	size_t vocab_size;
	options_mutable();

};

struct options
{
	const std::string training;
	const std::string huff_code;
	const std::string huff_point;
	const std::string point_feature;
	const std::string word_feature;
	const size_t batch_size;
	const size_t iteration;
	const size_t validation_ratio;
	const std::string binary_training;
	const std::string binary_window;
	const size_t grid_diagnal;
	const dim3 grid_dim;
	float learning_rate;
	const size_t feature_size;
	const size_t tree_size;
	const size_t vocab_size;
	options(options_mutable & opt);
};


void random_window_cut(environment & env,const options & opts);
void random_window_cut(const std::vector<size_t> & delimiter,const char* dir);
void random_window_cut(std::vector<size_t> & delimiter, std::vector<window_t> & window);
void save_point_feature(const real* feature,options & opts);
void validation(environment & env,size_t size=0,const dim3 dim=GRIDDIM);
void display_likelihood(const real* likelihood, const size_t size);
void window_recut(environment & env,const size_t size);
void load_huffman_code(std::vector<huffman_code_t> & huffman_code,
					   std::vector<huffman_indx_t> &huffman_indx,const char* resource);
void load_huffman_point(std::vector<huffman_point_t> & huffman_point,
						std::vector<huffman_indx_t> & huffman_indx,const char* resource);
void allocate_model_parameters(cuda_auto_ptr<real> & point_ptr,
		cuda_auto_ptr<real> & word_ptr, options & opts);
void write_results(environment &env,const options & opts);
inline real neural_net_parameter_initialization(const size_t feature_size);
bool next_batch(environment & env, options & opt);
void binary_process(const char* in, const char* out,std::vector<size_t> & delimiter);
void binary_process(environment & env, options & opt);
void training(environment & env,options & opt, size_t size);
void validation_parallel(environment & env,const options & opts,size_t size);
void binary_process(const char* in, const char* out,std::vector<size_t> & delimiter);
bool load_binary_training_disk(const char* dir,char* buffer,
							   const size_t size,const size_t offset);
#endif /* LANGUAGE_NETWORK_UTILITY_H_ */
