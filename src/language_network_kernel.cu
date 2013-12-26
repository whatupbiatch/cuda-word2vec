#include "includes/language_network_typedefs.h"
#include "includes/language_network_constants.h"
#include "includes/language_network_kernel.h"
#include "includes/language_network_memory.hpp"
#include "includes/language_network_defaults.h"
#include <cuda.h>
#include <vector>

texture<huffman_indx_t> huffman_indx_texture;
texture<huffman_code_t> huffman_code_texture;
texture<huffman_point_t> huffman_point_texture;
texture<window_t> context_range_texture;
texture<word> corpus_texture;



__global__ void cbow_neural_network_async_validation_kernel(real* likelihood,
															const real* word_distribution,const real* point_distribution,
															const size_t corpus_size)
{
	const unsigned int FEATURESIZE = blockDim.x;
		// forward propagation, exactly the same as validation kernel
	extern __shared__ real out_layer_input[];
	const unsigned int block_id  = indx(blockIdx,gridDim);
	if(block_id >= corpus_size) return;
	const window_t range = tex1Dfetch(context_range_texture,block_id);
	if(range == 0)
	{
		return;
	}

	const unsigned int thread_id = threadIdx.x;


// add up context predictor vector
	real feature_i_mutable = 0;
	huffman_point_t context_indx;
	for(window_t i=1; i<= range; i++)
	{
	    context_indx = tex1Dfetch(corpus_texture,block_id+i);
	    feature_i_mutable += word_distribution[context_indx*FEATURESIZE + thread_id];
		context_indx = tex1Dfetch(corpus_texture,block_id-i);
		feature_i_mutable += word_distribution[context_indx*FEATURESIZE + thread_id];
	}
	const real feature_i = feature_i_mutable;

	// output the hidden activation

	//compute gradient for each huffman point activation
	context_indx = tex1Dfetch(corpus_texture,block_id);
	const huffman_indx_t start = tex1Dfetch(huffman_indx_texture, context_indx);
	const huffman_indx_t end = tex1Dfetch(huffman_indx_texture, context_indx+1);
	huffman_indx_t indicator = start;


	real likelihood_aggregate = 0;

	while(indicator < end)
	{
		const huffman_point_t point = tex1Dfetch(huffman_point_texture,indicator);
		const real point_feature_i = point_distribution[point*FEATURESIZE + thread_id];
		out_layer_input[thread_id] = feature_i*point_feature_i;
		__syncthreads();

		for (unsigned int i = FEATURESIZE/2; i >0; i >>= 1)
		{
			if(thread_id < i)
			{
				out_layer_input[thread_id] += out_layer_input[thread_id +i];
			}
			__syncthreads();
		}

		//compute negative loglikelihood in current corpus texture
		const huffman_code_t code = tex1Dfetch(huffman_code_texture,indicator);
		likelihood_aggregate += neg_log_sigmoid(out_layer_input[0]*(2*(real)code -1));
		indicator++;
	}
	likelihood[block_id] = likelihood_aggregate;

}


__global__ void cbow_neural_network_async_kernel(real* word_distribution,real* point_distribution,
													const size_t corpus_size,const size_t offset,const real learning_rate)
{
	
	const unsigned int FEATURESIZE = blockDim.x;
		// forward propagation, exactly the same as validation kernel
	extern __shared__ real out_layer_input[];
	const unsigned int block_id  = indx(blockIdx,gridDim);
	if(block_id >= corpus_size || block_id < offset) return;
	const window_t range = tex1Dfetch(context_range_texture,block_id);
	if(range == 0)
	{
		return;
	}

	const unsigned int thread_id = threadIdx.x;

// add up context predictor vector
	real feature_i_mutable = 0;
	huffman_point_t context_indx;
	for(window_t i=1; i<= range; i++)
	{
	    context_indx = tex1Dfetch(corpus_texture,block_id+i);
	    feature_i_mutable += word_distribution[context_indx*FEATURESIZE + thread_id];
		context_indx = tex1Dfetch(corpus_texture,block_id-i);
		feature_i_mutable += word_distribution[context_indx*FEATURESIZE + thread_id];
	}
	const real feature_i = feature_i_mutable;



	// output the hidden activation

	//compute gradient for each huffman point activation
	context_indx = tex1Dfetch(corpus_texture,block_id);
	const huffman_indx_t start = tex1Dfetch(huffman_indx_texture, context_indx);
	const huffman_indx_t end = tex1Dfetch(huffman_indx_texture, context_indx+1);


	real update_i = 0;
	for(huffman_indx_t indicator = start; indicator < end; indicator++)
	{
		const huffman_point_t point = tex1Dfetch(huffman_point_texture,indicator);
		const real point_feature_i = point_distribution[point*FEATURESIZE + thread_id];
		out_layer_input[thread_id] = feature_i*point_feature_i;

		__syncthreads();

		for (unsigned int i = FEATURESIZE/2; i >0; i >>= 1)
		{
			if(thread_id < i)
			{
				out_layer_input[thread_id] += out_layer_input[thread_id +i];
			}
			__syncthreads();
		}
		

		const huffman_code_t code = tex1Dfetch(huffman_code_texture,indicator);
		const real grad = sigmoid(out_layer_input[0]) - ((real)code);
		point_distribution[point*FEATURESIZE + thread_id] = point_feature_i - learning_rate*grad*feature_i;
		update_i += grad*point_feature_i;
	}

	update_i *= learning_rate;

	for(window_t i=1; i<= range; i++)
	{
		context_indx = tex1Dfetch(corpus_texture,block_id+i);
		word_distribution[context_indx*FEATURESIZE + thread_id] -= update_i;
	}

	for(window_t i=1; i<= range; i++)
	{
		context_indx = tex1Dfetch(corpus_texture,block_id-i);
		word_distribution[context_indx*FEATURESIZE + thread_id] -= update_i;
	}

}


__device__ float sigmoid(const real numbers)
{
	return 1/(1+expf(-numbers));
}


__device__ real neg_log_sigmoid(const real f)
{
	return logf(1+expf(-f));
}

__device__ indx_t indx( dim3 elem,dim3 blockD)
{
	return elem.x + blockD.x*elem.y + blockD.x*blockD.y*elem.z;
}



environment::environment():
				code_binder(huffman_code_texture),point_binder(huffman_point_texture),
				indx_binder(huffman_indx_texture),corpus_binder(corpus_texture),
				window_binder(context_range_texture),file_ptr_word_count(0) {}


environment::~environment()
{
		code_binder.unbind();
		point_binder.unbind();
		indx_binder.unbind();
		corpus_binder.unbind();
		window_binder.unbind();
}


//---------------------------tests------------------------------------------

/*
 * test kernels using global textures have to be defined here
 */



__global__ void t_corpus_texture_binding_kernel(word* out,size_t size)
{
	size_t index = indx(blockIdx,gridDim);
	if(index >= size) return;
	word val = tex1Dfetch(corpus_texture,index);
	out[index] = val;
}



__global__ void t_window_texture_binding_kernel(window_t* out,size_t size)
{
	size_t index = indx(blockIdx,gridDim);
	if(index >= size) return;
	window_t val = tex1Dfetch(context_range_texture,index);
	out[index] = val;
}

__global__ void t_huff_code_texture_binding_kernel(huffman_code_t* out,size_t size)
{
	size_t index = indx(blockIdx,gridDim);
	if(index >= size ) return;
	huffman_code_t val = tex1Dfetch(huffman_code_texture,index);
	out[index] = val;
}


__global__ void t_huff_point_texture_binding_kernel(huffman_point_t* out, size_t size)
{
	size_t index = indx(blockIdx,gridDim);
	if(index >= size ) return;
	huffman_point_t val = tex1Dfetch(huffman_point_texture,index);
	out[index] = val;
}


__global__ void t_huff_indx_texture_binding_kernel(huffman_indx_t* out, size_t size)
{
	size_t index = indx(blockIdx,gridDim);
	if(index >= size ) return;
	huffman_indx_t val = tex1Dfetch(huffman_indx_texture,index);
	out[index] = val;
}

__global__ void t_mid_layer_activation(real* midLayer,const real* word_distribution,const size_t corpus_size)
{

	const unsigned int FEATURESIZE = blockDim.x;
		// forward propagation, exactly the same as validation kernel
	extern __shared__ real out_layer_input[];
	const unsigned int block_id  = indx(blockIdx,gridDim);
	if(block_id >= corpus_size) return;
	const window_t range = tex1Dfetch(context_range_texture,block_id);
	if(range == 0)
	{
		return;
	}

	const unsigned int thread_id = threadIdx.x;


// add up context predictor vector( mid layer activation)
	real feature_i = 0;
	huffman_point_t context_indx;
	for(window_t i=1; i<= range; i++)
	{
	    context_indx = tex1Dfetch(corpus_texture,block_id+i);
		feature_i += word_distribution[context_indx*FEATURESIZE + thread_id];
		context_indx = tex1Dfetch(corpus_texture,block_id-i);
		feature_i += word_distribution[context_indx*FEATURESIZE + thread_id];
	}
	// output the hidden activation

	//compute gradient for each huffman point activation
	midLayer[threadIdx.x + FEATURESIZE*block_id] = feature_i;


}

/*
 * kernel implementing gradient_check
 */

__global__ void t_gradient_check_kernel(real* difference,const real* word_distribution,const real* point_distribution,
		const size_t corpus_size,const size_t pertub_cord,const real epsilong)
{

	const unsigned int FEATURESIZE = blockDim.x;
		// forward propagation, exactly the same as validation kernel
	extern __shared__ real out_layer_input[];
	const unsigned int block_id  = indx(blockIdx,gridDim);
	if(block_id >= corpus_size) return;
	const window_t range = tex1Dfetch(context_range_texture,block_id);
	if(range == 0)
	{
		return;
	}

	const unsigned int thread_id = threadIdx.x;


// add up context predictor vector
	real feature_i_mutable = 0;
	huffman_point_t context_indx;
	for(window_t i=1; i<= range; i++)
	{
	    context_indx = tex1Dfetch(corpus_texture,block_id+i);
	    feature_i_mutable += word_distribution[context_indx*FEATURESIZE + thread_id];
		context_indx = tex1Dfetch(corpus_texture,block_id-i);
		feature_i_mutable += word_distribution[context_indx*FEATURESIZE + thread_id];
	}

	const real feature_i = feature_i_mutable;
	// output the hidden activation

	//compute gradient for each huffman point activation
	context_indx = tex1Dfetch(corpus_texture,block_id);
	const huffman_indx_t start = tex1Dfetch(huffman_indx_texture, context_indx);
	const huffman_indx_t end = tex1Dfetch(huffman_indx_texture, context_indx+1);
	huffman_indx_t indicator = start;


	real aggregate_difference = 0;
	while(indicator < end)
	{
		const huffman_point_t point = tex1Dfetch(huffman_point_texture,indicator);
		const real point_feature_i = point_distribution[point*FEATURESIZE + thread_id];
		out_layer_input[thread_id] = feature_i*point_feature_i;
		__syncthreads();

		for (unsigned int i = FEATURESIZE/2; i >0; i >>= 1)
		{
			if(thread_id < i)
			{
				out_layer_input[thread_id] += out_layer_input[thread_id +i];
			}
			__syncthreads();
		}

		huffman_code_t code = tex1Dfetch(huffman_code_texture,start);
		real op = sigmoid(out_layer_input[0]);
		real grad = op - ((real)code);

		op = neg_log_sigmoid(out_layer_input[0]*(2*(real)code -1));

		// pertubation the ith coordinate of mid layer
		out_layer_input[thread_id] = feature_i*point_feature_i;
		if(thread_id==pertub_cord) out_layer_input[thread_id] +=epsilong*point_feature_i;

		for (unsigned int i = FEATURESIZE/2; i >0; i >>= 1)
		{
			if(thread_id < i)
			{
				out_layer_input[thread_id] += out_layer_input[thread_id +i];
			}
			__syncthreads();
		}

		real op_pertub = neg_log_sigmoid(out_layer_input[0]*(2*(real)code -1));
		real partial_derivative = (op_pertub - op)/epsilong;

		if(thread_id==pertub_cord)
			aggregate_difference += fabsf(partial_derivative - grad*point_feature_i);

		//pertubation of the ith coordinate of point_feature
		out_layer_input[thread_id] = feature_i*point_feature_i;
		if(thread_id==pertub_cord) out_layer_input[thread_id] +=epsilong*feature_i;

		for (unsigned int i = FEATURESIZE/2; i >0; i >>= 1)
		{
			if(thread_id < i)
			{
				out_layer_input[thread_id] += out_layer_input[thread_id +i];
			}
			__syncthreads();
		}
		op_pertub =  neg_log_sigmoid(out_layer_input[0]*(2*(real)code -1));
		partial_derivative = (op_pertub - op)/epsilong;

		if(thread_id==pertub_cord)
			aggregate_difference += fabsf(partial_derivative - grad*feature_i);


		indicator++;
	}
	if(thread_id==pertub_cord)
		difference[block_id] = aggregate_difference;


}







