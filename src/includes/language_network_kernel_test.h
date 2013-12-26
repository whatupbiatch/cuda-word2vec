/*
 * language_network_kernel_test.h
 *
 *  Created on: 2013-12-24
 *      Author: Chen Pu
 */

#ifndef LANGUAGE_NETWORK_KERNEL_TEST_H_
#define LANGUAGE_NETWORK_KERNEL_TEST_H_
__global__ void t_corpus_texture_binding_kernel(word* out,size_t size);
__global__ void t_window_texture_binding_kernel(window_t* out,size_t size);
__global__ void t_huff_code_texture_binding_kernel(huffman_code_t* out,size_t size);
__global__ void t_huff_point_texture_binding_kernel(huffman_point_t* out, size_t size);
__global__ void t_huff_indx_texture_binding_kernel(huffman_indx_t* out, size_t size);
__global__ void t_mid_layer_activation(real* midLayer,const real* word_distribution,
				const size_t corpus_size);
__global__ void t_gradient_check_kernel(real* difference,const real* word_distribution,
		const real* point_distribution,const size_t corpus_size,
		const size_t pertub_cord,const real epsilong);
#endif /* LANGUAGE_NETWORK_KERNEL_TEST_H_ */
