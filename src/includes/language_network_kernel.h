/*
 * language_network_kernel.h
 *
 *  Created on: 2013-12-22
 *      Author: Chen Pu
 */

#ifndef LANGUAGE_NETWORK_KERNEL_H_
#define LANGUAGE_NETWORK_KERNEL_H_
#include "language_network_typedefs.h"
#include "language_network_constants.h"
#include  "language_network_memory.hpp"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
__global__ void cbow_neural_network_async_validation_kernel(real* likelihood,const real* word_distribution,
															const real* point_distribution,const size_t corpus_size);

__global__ void cbow_neural_network_async_kernel(real* word_distribution,real* point_distribution,
													const size_t corpus_size,const size_t offset,const real learning_rate);

__device__ float sigmoid(const real numbers);
__device__ real neg_log_sigmoid(const real f);
__device__ indx_t indx( dim3 elem,dim3 blockD);

struct environment
{
	std::vector<size_t> corpus_delimiter;
	std::vector<huffman_code_t> huff_code_host;
	std::vector<huffman_point_t> huff_point_host;
	std::vector<huffman_indx_t> huff_indx_host;
	binder<huffman_code_t> code_binder;
	binder<huffman_point_t> point_binder;
	binder<huffman_indx_t> indx_binder;
	binder<word> corpus_binder;
	binder<window_t> window_binder;
	cuda_auto_ptr<huffman_code_t> huff_code_device;
	cuda_auto_ptr<huffman_point_t> huff_point_device;
	cuda_auto_ptr<huffman_indx_t> huff_indx_device;
	cuda_auto_ptr<word> corpus_device;
	cuda_auto_ptr<window_t> window_device;
	cuda_auto_ptr<real> word_feature;
	cuda_auto_ptr<real> point_feature;
	size_t file_ptr_word_count;
	environment();
	~environment();
};


#endif /* LANGUAGE_NETWORK_KERNEL_H_ */
