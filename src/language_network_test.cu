#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include<device_functions.h>
#include "texture_fetch_functions.h"
#include "texture_types.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda_device_runtime_api.h"
#include <math.h>
#include <vector>
#include <memory>
#include <fstream>
#include <iterator>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "includes/language_network_constants.h"
#include "includes/language_network_typedefs.h"
#include "includes/language_network_memory.hpp"
#include "includes/language_network_kernel.h"
#include "includes/language_network_utility.h"
#include "includes/language_network_test_dir.h"
#include "includes/language_network_kernel_test.h"

void load_batch(std::vector<word> &corpus, std::vector<size_t> & delimiter, const char* dir)
{

	std::ifstream file(dir);
	std::string line;
	delimiter.push_back(corpus.size());
	while(std::getline(file,line))
	{
		std::stringstream ss(line);
		word wd;
		while(ss >> wd)
		{
			corpus.push_back(wd);
		}
		delimiter.push_back(corpus.size());
	}

	file.close();
}

void test_environment_setup(options & opt, environment & env,std::vector<word> & corpus, std::vector<window_t> & window)
{

		load_batch(corpus,env.corpus_delimiter,training_dir);
		random_window_cut(env.corpus_delimiter,window);
		env.corpus_device.allocate_and_copy(corpus,"corpus free failed");
		env.window_device.allocate_and_copy(window,"window free failed");
		env.corpus_binder.rebind(env.corpus_device.get(),env.corpus_device.size());
		env.window_binder.rebind(env.window_device.get(),env.window_device.size());
		load_huffman_code(env.huff_code_host,env.huff_indx_host,huff_code_dir);
		load_huffman_point(env.huff_point_host,env.huff_indx_host,huff_point_dir);
		env.huff_code_device.allocate_and_copy(env.huff_code_host);
		env.huff_point_device.allocate_and_copy(env.huff_point_host);
		env.huff_indx_device.allocate_and_copy(env.huff_indx_host);
		env.code_binder.rebind(env.huff_code_device.get(),env.huff_code_device.size());
		env.point_binder.rebind(env.huff_point_device.get(),env.huff_point_device.size());
		env.indx_binder.rebind(env.huff_indx_device.get(),env.huff_indx_device.size());
		allocate_model_parameters(env.point_feature,env.word_feature,opt);
}


void t_binary_process()
{
	std::vector<size_t> delimiter;
	std::vector<size_t> delimiter_control;
	std::vector<word> corpus_control;
	load_batch(corpus_control,delimiter_control,training_dir);
	std::cout << "control group loaded" << std::endl;
	binary_process(training_dir,training_binary,delimiter);
	std::cout << "test data loaded" << std::endl;
	for(size_t i=0; i<delimiter_control.size(); i++)
	{
		if(delimiter[i]!=delimiter_control[i]) std::cout << "binary test failed in delimiter check" <<std::endl;
	}
	std::cout << "delimiter test passed" << std::endl;
	size_t batch_size = 90000;

	std::auto_ptr<word> buffer((word*)malloc(sizeof(word)*batch_size));
	for(size_t i=0; i<corpus_control.size()/batch_size; i++)
	{
		std::cout << "testing batch " << i << std::endl;
		bool ret = load_binary_training_disk(training_binary,(char*)buffer.get(),sizeof(word)*batch_size,i*sizeof(word)*batch_size);
		if(! ret)
		{
			std::cout << "binary length error " << i <<" " << corpus_control.size()/batch_size << std::endl;
			break;
		}
		for(size_t j=0; j<batch_size; j++)
		{
			if(buffer.get()[j] != corpus_control[batch_size*i +j])
			{
				std::cout << "binary corpus check failed" << std::endl;
				return;
			} 
		}
	}
	std::cout << "binary processing check passed" << std::endl;
}

// add on disk random window to streaming implementation of next_batch

void t_random_window()
{

	std::vector<size_t> delimiter;
	std::vector<word> corpus;
	load_batch(corpus,delimiter,training_dir);
	std::vector<window_t> window;
	random_window_cut(delimiter,window);
	if(window.size()!=corpus.size()) std::cout << "window size check failed " << window.size() << " " <<corpus.size() << std::endl;
	if(window[0]!=0 || window[delimiter[delimiter.size()-1]-1]!=0)std::cout << "head and tail failed" << std::endl;
	for(size_t i=1; i < delimiter.size() -1; i++)
	{
		if(window[delimiter[i]]!=0 || window[delimiter[i] -1]!=0) std::cout << "window cut test failed" << std::endl;
	}

	for(size_t i=0; i< window.size(); i++)
	{
		if(window[i] > MAXWINDOW) std::cout << "window range failed" << std::endl;
	}
	std::cout << "in memory window test passed" << std::endl;

	std::auto_ptr<window_t> window_test((window_t*)malloc(sizeof(window_t)*window.size()));
	random_window_cut(delimiter,training_window_binary);
	bool ret = load_binary_training_disk(training_window_binary,(char*)window_test.get(),sizeof(window_t)*window.size(),0);
	if(! ret) std::cout << "binary window load test failed" << std::endl;

	if(window_test.get()[0]!=0 || window_test.get()[delimiter[delimiter.size()-1]-1]!=0)std::cout << "head and tail failed" << std::endl;
	for(size_t i=1; i < delimiter.size() -1; i++)
	{
		if(window_test.get()[delimiter[i]]!=0 || window_test.get()[delimiter[i] -1]!=0)
			std::cout << "window cut test failed" << std::endl;
	}

	for(size_t i=0; i< window.size(); i++)
	{
		if(window_test.get()[i] > MAXWINDOW) std::cout << "window range failed" << std::endl;
	}
	std::cout << "on disk window test passed" << std::endl;

}


void t_refill()
{
	cuda_auto_ptr<float> ptr;
	ptr.allocate(30);
	std::vector<float> vec(30,1);
	ptr.refill(&vec[0]);
	float* back = ptr.fetch_raw();
	for(size_t i=0; i<30; i++)
	{
		if(back[i]!=vec[i]) std::cout << "refill check failed" << std::endl;
	}
	std::cout<< "refill passed" << std::endl;
}


void t_corpus_texture()
{

	environment env;
	cuda_auto_ptr<word> test;
	std::vector<word> vec;
	for(size_t i=0; i<100; i++)
	{
		vec.push_back(i);
	}
	test.allocate_and_copy(vec,"texture test free failed");
	env.corpus_binder.rebind(test.get(),100);
	cuda_auto_ptr<word> test_out;
	test_out.allocate(100,"test out free failed");
	t_corpus_texture_binding_kernel<<<100,1>>>(test_out.get(),100);
	std::auto_ptr<word> result(test_out.fetch_raw());
	for(size_t i=0; i<100; i++)
	{
		if(vec[i]!=result.get()[i]) std::cerr<< "texture test failed"<< std::endl;
		break;
	}
	env.corpus_binder.unbind();
	std::cout << "corpus texture test done" << std::endl;
}

void t_corpus_window_texture()
{
	environment env;
	cuda_auto_ptr<window_t> test;
	std::vector<window_t> vec;
	for(size_t i=0; i<100; i++)
	{
		vec.push_back(i);
	}
	test.allocate_and_copy(vec);
	env.window_binder.rebind(test.get(),100);
	cuda_auto_ptr<window_t> test_out;
	test_out.allocate(100,"test out free failed");
	t_window_texture_binding_kernel<<<100,1>>>(test_out.get(),100);
	std::auto_ptr<window_t> result(test_out.fetch_raw());
	for(size_t i=0; i< 100; i++)
	{
		if(vec[i]!=result.get()[i]) std::cerr<< "texture test failed"<< std::endl;
		break;
	}
	env.window_binder.unbind();
	std::cout << "window texture test done" << std::endl;
}

void t_huffman_texture()
{
	std::cout << "test started" << std::endl;
	environment env;
	load_huffman_code(env.huff_code_host,env.huff_indx_host,huff_code_dir);
	load_huffman_point(env.huff_point_host,env.huff_indx_host,huff_point_dir);
	std::cout <<" code size" << env.huff_code_host.size() <<" point size " << env.huff_point_host.size() << std::endl;

	env.huff_code_device.allocate_and_copy(env.huff_code_host);
	if(env.huff_code_device.size()!=env.huff_code_host.size())
		std::cerr << "code device size check failed" << std::endl;
	std::cout << "resource allocated" << std::endl;
	env.code_binder.rebind(env.huff_code_device.get(),env.huff_code_device.size());
	std::cout << "resource binded" << std::endl;
	cuda_auto_ptr<huffman_code_t> code_out;
	code_out.allocate(env.huff_code_host.size());
	t_huff_code_texture_binding_kernel<<<env.huff_code_host.size(),1>>>
			(code_out.get(),env.huff_code_device.size());
	std::cout << "kernel returned" << std::endl;
	std::auto_ptr<huffman_code_t> code_result(code_out.fetch_raw());
	for(size_t i=0; i<code_out.size();i++)
	{
		if(code_result.get()[i]!=env.huff_code_host[i])
			std::cerr << "code check failed" << std::endl;
	}
	std::cout << "code check done" << std::endl;
	code_out.release();
	env.code_binder.unbind();
	env.huff_code_device.release();

	env.huff_point_device.allocate_and_copy(env.huff_point_host,"point memfree failed");
	cuda_auto_ptr<huffman_point_t> point_test;
	point_test.allocate(env.huff_point_host.size(),"point mem free failed");
	env.point_binder.rebind(env.huff_point_device.get(),env.huff_point_device.size());
	t_huff_point_texture_binding_kernel<<<env.huff_point_device.size(),1>>>
			(point_test.get(),env.huff_point_device.size());
	std::auto_ptr<huffman_point_t> point_result(point_test.fetch_raw());
	for(size_t i=0; i<point_test.size(); i++)
	{
		if(point_result.get()[i]!=env.huff_point_host[i])
			std::cerr << "point check failed" << std::endl;
	}
	std::cout <<"point check done" << std::endl;
	point_test.release();
	env.point_binder.unbind();
	env.huff_point_device.release();
}


void t_gradient_check()
{
	options_mutable mut;
	mut.feature_size = 8;
	options opt(mut);
	environment env;
	std::vector<word> corpus;
	std::vector<window_t> window;
	test_environment_setup(opt,env,corpus,window);
	cuda_auto_ptr<real> difference;
	difference.allocate(corpus.size());
	difference.memset(0);
	t_gradient_check_kernel<<<corpus.size(),opt.feature_size,sizeof(real)*opt.feature_size>>>(difference.get(),env.word_feature.get(),
			env.point_feature.get(),corpus.size(),1,0.01);
	std::auto_ptr<real> diff_host(difference.fetch_raw());
	real mx = 0;
	double average =0;
	for(size_t i=0; i<corpus.size(); i++)
	{
		real diff = diff_host.get()[i];
		mx = std::max(diff,mx);
		average+= diff;
	}
	average = average/corpus.size();
	std::cout << "gradient check done, averge error " << average << " max error" << mx << std::endl;
}

void t_mid_layer_activation()
{
	options_mutable mut;
	mut.feature_size = 8;
	options opt(mut);
	environment env;
	std::vector<word> corpus;
	std::vector<window_t> window;
	test_environment_setup(opt,env,corpus,window);
	env.point_feature.release();
	cuda_auto_ptr<real> midlayer;
	midlayer.allocate(corpus.size()*opt.feature_size);
	midlayer.memset(0);

	t_mid_layer_activation<<<corpus.size(),opt.feature_size,sizeof(real)*opt.feature_size>>>
			(midlayer.get(),env.word_feature.get(),corpus.size());


	std::auto_ptr<real> result_mid_layer(midlayer.fetch_raw());



	std::auto_ptr<real> word_feature_init(env.word_feature.fetch_raw());
	std::vector<real> host_layer(opt.feature_size,0);
	for(size_t i=0; i< corpus.size(); i++)
	{
		std::fill(host_layer.begin(),host_layer.end(),0);
		window_t range = window[i];
		for(size_t j=1; j<=range; j++)
		{
			for(size_t k=0; k<opt.feature_size; k++)
			{
				host_layer[k]+= word_feature_init.get()[k+opt.feature_size*corpus[i+j]];
				host_layer[k]+= word_feature_init.get()[k+opt.feature_size*corpus[i-j]];
			}
		}

		for(size_t k=0; k<opt.feature_size; k++)
		{
			if(host_layer[k]!=result_mid_layer.get()[k+opt.feature_size*i])
			{
				std::cerr << "mid layer check failed" << std::endl;
				std::cerr << "position " << i << " " << k <<" " <<host_layer[k]
				  <<" " <<result_mid_layer.get()[k+opt.feature_size*i] <<  std::endl;
				i = corpus.size();
				break;
			}

		}
	}

	std::cout << "env setup" << std::endl;
	return;

}

void t_likelihood_validation()
{ }


int main()
{
	t_gradient_check();
	cudaDeviceReset();
	t_mid_layer_activation();
	cudaDeviceReset();
	t_huffman_texture();
	cudaDeviceReset();
	t_corpus_window_texture();
	cudaDeviceReset();
	t_corpus_texture();
	cudaDeviceReset();
	t_refill();
	cudaDeviceReset();
	t_random_window();
	cudaDeviceReset();
	t_binary_process();
	return 0;
}
