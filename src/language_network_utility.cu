#include "includes/language_network_typedefs.h"
#include "includes/language_network_constants.h"
#include "includes/language_network_memory.hpp"
#include "includes/language_network_utility.h"
#include "includes/language_network_kernel.h"
#include "includes/language_network_defaults.h"
#include <fstream>
#include <vector>
#include <memory>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <string>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>


void random_window_cut(environment & env,const options & opts)
{
	random_window_cut(env.corpus_delimiter,opts.binary_window.c_str());
}

void random_window_cut(std::vector<size_t> & delimiter, std::vector<window_t> &window)
{
	size_t corpus_size = delimiter[delimiter.size()-1];
	size_t doc_counter = 0;
	window.clear();
	for(size_t i=0; i<corpus_size; i++)
	{
		if(i >= delimiter[doc_counter+1])
		{
			doc_counter++;
		}
		size_t doc_length = delimiter[doc_counter +1] - delimiter[doc_counter];
		size_t pos = i - delimiter[doc_counter];
		window_t range_mx =(window_t) std::min(std::min((size_t)MAXWINDOW-1,pos),(doc_length-pos-1));
		if(range_mx == 0)
		{
			window.push_back(0);
			continue;
		}
		window_t range = (window_t)(1 + (rand() % (unsigned int)(range_mx)));
		window.push_back(range);
	}
}


void random_window_cut(const std::vector<size_t> & delimiter,const char* dir)
{

	size_t corpus_size = delimiter[delimiter.size() -1];
	std::auto_ptr<window_t> buffer((window_t*)malloc(sizeof(window_t)*BUFFERSIZE));
	std::ofstream ostream(dir,std::ios::out | std::ios::binary);
	size_t buffer_counter = 0;
	size_t doc_counter = 0;
	for(size_t i=0; i< corpus_size; i++)
	{
		if(buffer_counter >= BUFFERSIZE)
		{
			ostream.write((char*)buffer.get(),sizeof(window_t)*BUFFERSIZE);
			buffer_counter = 0;
		}
		if( i>=delimiter[doc_counter+1])
		{
			doc_counter++;
		}
		size_t doc_length = delimiter[doc_counter+1] - delimiter[doc_counter];
		size_t pos = i- delimiter[doc_counter];
		window_t range_mx =(window_t) std::min(std::min((size_t)MAXWINDOW-1,pos),(doc_length-pos-1));
		if(range_mx == 0)
		{
		   buffer.get()[buffer_counter ++] = 0;
		   continue;
		}
		window_t range = (window_t)(1 + (rand() % (unsigned int)(range_mx)));
		buffer.get()[buffer_counter ++] = range;
	}

	ostream.write((char*)buffer.get(),sizeof(window_t)*buffer_counter);
	ostream.close();
}


void save_point_feature(const real* feature,options & opts)
{

	std::ofstream file(opts.point_feature.c_str());
	for(size_t i=0; i<opts.tree_size; i++)
	{
		for(size_t j=0; j<opts.feature_size; j++)
		{
			file << feature[i*opts.feature_size +j] << " ";
		}
		file << std::endl;
	}
	file.close();
}


void training(environment & env,options & opt, size_t size)
{
	cbow_neural_network_async_kernel<<<opt.grid_dim,opt.feature_size,opt.feature_size*sizeof(real)>>>(env.word_feature.get(),env.point_feature.get(),
		size,MAXWINDOW,opt.learning_rate);
}



void load_huffman_code(std::vector<huffman_code_t> & huffman_code,
					   std::vector<huffman_indx_t> &huffman_indx,const char* resource)
{
	std::ifstream file;
	file.open(resource, std::ios::in);
	std::string line;
	while(std::getline(file,line))
	{
		huffman_indx.push_back(huffman_code.size());
		std::stringstream ss(line);
		int code;
		while(ss >> code)
		{
			huffman_code.push_back(code);
		}
	}
	huffman_indx.push_back(huffman_code.size());
	file.close();
}


void load_huffman_point(std::vector<huffman_point_t> & huffman_point,
						std::vector<huffman_indx_t> & huffman_indx,const char* resource)
{
	std::ifstream file;
	file.open(resource,std::ios::in);
	std::string line;
	int count = 0;
	while(std::getline(file,line))
	{
		if(huffman_indx[count]!=huffman_point.size())
		{
			std::cerr<<"huffman files do not match";
			file.close();
			exit(EXIT_FAILURE);
		}
		count++;
		std::stringstream ss(line);
		int point;
		while(ss >> point)
		{
			huffman_point.push_back(point);
		}

	}
	if(huffman_indx[count]!=huffman_point.size())
	{
		std::cerr<<"huffman files do not match";
		file.close();
		exit(EXIT_FAILURE);
	}
	file.close();
}


void allocate_model_parameters(cuda_auto_ptr<real> & point_ptr,
		cuda_auto_ptr<real> & word_ptr, options & opts)
{
	size_t word_size= opts.feature_size*opts.vocab_size;
	size_t point_size = opts.feature_size*opts.tree_size;
	std::auto_ptr<real> rand_word_parameters((real*)malloc(sizeof(real)*word_size));
	std::auto_ptr<real> rand_point_parameters((real*)malloc(sizeof(real)*point_size));

	for(int i=0; i<word_size; i++)
	{
		rand_word_parameters.get()[i] = neural_net_parameter_initialization(opts.feature_size);
	}

	for(int i=0; i<point_size; i++)
	{
		rand_point_parameters.get()[i] = neural_net_parameter_initialization(opts.feature_size);
	}
	point_ptr.allocate_and_copy(point_size,rand_point_parameters.get(),"free point features allocation failed");
	word_ptr.allocate_and_copy(word_size,rand_word_parameters.get(),"free word features allocation failed");
}



void write_results(environment &env,const options & opts)
{
	std::auto_ptr<real> word_distribution (env.word_feature.fetch_raw());
	std::auto_ptr<real> point_distribution (env.point_feature.fetch_raw());
	std::ofstream out(opts.word_feature.c_str());
	std::cout<< "word size: " << opts.vocab_size <<" " << "point size: " << opts.tree_size << std::endl;
	for(size_t i=0; i< opts.vocab_size; i++)
	{
		for(size_t j=0; j< opts.feature_size; j++)
		{
			out << word_distribution.get()[i*opts.feature_size + j];
			out << " ";
		}
		out << "\n";
	}

	out.close();
	std::cout <<"word feature saved\n";

	std::ofstream point_out(opts.point_feature.c_str());

	for(size_t i=0; i< opts.tree_size; i++)
	{
		for(size_t j=0; j< opts.feature_size; j++)
		{
			point_out << point_distribution.get()[i*opts.feature_size + j];
			point_out << " ";
		}
		point_out << "\n";
	}
	point_out.close();

}

inline real neural_net_parameter_initialization(const size_t feature_size)
{
	return (((real)rand())/RAND_MAX - 0.5)/feature_size;
}


bool next_batch(environment & env, options & opt)
{
	bool ret = false;
	std::auto_ptr<word> buffer ((word*)malloc(sizeof(word)*opt.batch_size));
	if(!buffer.get())
	{
		std::cerr << "cannot allocate training buffer" << std::endl;
		return false;
	}
	ret = load_binary_training_disk(opt.binary_training.c_str(),(char*)buffer.get(),sizeof(word)*opt.batch_size,
		sizeof(word)*opt.batch_size*env.file_ptr_word_count);

	if(! ret) return ret;

	env.corpus_binder.unbind();
	ret = env.corpus_device.refill(buffer.get());
	if(! ret)
	{
		std::cerr << "refill corpus failed" << std::endl;
		return ret;
	}
   

	env.corpus_binder.rebind(env.corpus_device.get(),env.corpus_device.size());
	buffer.release();

	std::auto_ptr<window_t> window_buffer((window_t*)malloc(sizeof(window_t)*opt.batch_size));
	if(!window_buffer.get())
	{
		std::cerr << "window buffer allocation faied" << std::endl;
		return false;
	}
	ret = load_binary_training_disk(opt.binary_window.c_str(),(char*)window_buffer.get(),sizeof(window_t)*opt.batch_size,
		sizeof(window_t)*opt.batch_size*env.file_ptr_word_count);
	if(!ret) 
	{
		std::cerr << "window length and corpus length does not match, binary files corrupted "<<std::endl;
		return false;
	}
	env.window_binder.unbind();
	ret = env.window_device.refill(window_buffer.get());
	if(!ret)
	{
		std::cerr << "window refill failed" << std::endl;
		return false;
	}
	env.window_binder.rebind(env.window_device.get(),env.window_device.size());

	env.file_ptr_word_count++;
	return ret;
}




bool load_binary_training_disk(const char* dir,char* buffer,const size_t size,const size_t offset)
{
	std::ifstream stream(dir,std::ios::binary);
	stream.seekg(offset);
	size_t c_size = stream.tellg();
	stream.read(buffer,size);
	if(stream)
	{
		stream.close();
		return true;
	}

	stream.close();
	return false;
}

void binary_process(environment & env, options & opt)
{
	binary_process(opt.training.c_str(),opt.binary_training.c_str(),env.corpus_delimiter);
}


void binary_process(const char* in, const char* out,std::vector<size_t> & delimiter)
{
	std::ifstream istream(in);
	std::ofstream ostream(out,std::ios::out | std::ios::binary);
	size_t corpus_size = 0;
	delimiter.clear();
	std::string line;
	delimiter.push_back(corpus_size);
	std::vector<word> buffer;
	while(std::getline(istream,line))
	{
		std::stringstream ss(line);
		buffer.clear();
		word wd;
		while(ss >> wd)
		{
			buffer.push_back(wd);
			corpus_size++;
		}
		ostream.write((char*)&buffer[0],buffer.size()*sizeof(word));
		delimiter.push_back(corpus_size);
	}

	std::cout << "training loaded, size " << corpus_size << " doc size " << delimiter.size()-1 << std::endl;
	istream.close();
	ostream.close();
}


void validation_parallel(environment & env,const options & opts, size_t size)
{
	size_t grid_size = opts.grid_diagnal*opts.grid_diagnal;
	size = std::min(size,grid_size);
	cuda_auto_ptr<real> likelihood;
	likelihood.allocate(size,"likelihood free failed");
	likelihood.memset(0);
	cbow_neural_network_async_validation_kernel<<<opts.grid_dim,opts.feature_size,sizeof(real)*opts.feature_size>>>(likelihood.get(),
				env.word_feature.get(),env.point_feature.get(),size);

	thrust::device_ptr<real> dev_ptr = thrust::device_pointer_cast(likelihood.get());

	real sum = thrust::reduce(dev_ptr,dev_ptr + size,0.0, thrust::plus<real>());

	std:: cout << "the current negative log likelihood is: " << sum << std::endl;
}


options::options(options_mutable & opt):binary_training(binary_training_default),binary_window(opt.binary_window),
			training(opt.training),huff_code(opt.huff_code),huff_point(opt.huff_point)
			,point_feature(opt.point_feature),word_feature(opt.word_feature)
			,batch_size(opt.batch_size),iteration(opt.iteration),validation_ratio(opt.validation_ratio)
			,grid_diagnal(opt.grid_diagnal),grid_dim(opt.grid_dim),learning_rate(opt.learning_rate)
			,feature_size(opt.feature_size),tree_size(opt.tree_size),vocab_size(opt.vocab_size) {}


options_mutable::options_mutable():binary_training(binary_training_default),binary_window(binary_window_default),
	training(training_default),huff_code(huff_code_default),huff_point(huff_point_default),point_feature(point_feature_default),
		word_feature(word_feature_default),batch_size(batch_size_default),iteration(iteration_default),
		validation_ratio(validation_ratio_default),grid_diagnal(grid_diag_default),grid_dim(grid_dim_default),
		learning_rate(learning_rate_default),feature_size(feature_size_default),tree_size(tree_size_default),
		vocab_size(vocab_size_default){}

