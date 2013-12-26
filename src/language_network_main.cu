#include <string>
#include <fstream>
#include <math.h>
#include "includes/optParser/getoptpp/getopt_pp.h"
#include "includes/language_network_constants.h"
#include "includes/language_network_typedefs.h"
#include "includes/language_network_memory.hpp"
#include "includes/language_network_utility.h"
#include "includes/language_network_test_dir.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>


int run(options & opt);


int main(int argc,char** argv)
{
	options_mutable opts;
	GetOpt::GetOpt_pp parser(argc,argv);
	parser >> GetOpt::Option("train",opts.training) >> GetOpt::Option("code",opts.huff_code)
	>> GetOpt::Option("point",opts.huff_point) >> GetOpt::Option("pFeature",opts.point_feature)
	>> GetOpt::Option("wFeature",opts.word_feature) >> GetOpt::Option("batch",opts.batch_size)
	>>GetOpt::Option("vRatio",opts.validation_ratio) >> GetOpt::Option("iter",opts.iteration)
	>>GetOpt::Option("learningRate",opts.learning_rate) >>GetOpt::Option("binaryTraining",opts.binary_training)
	>> GetOpt::Option("binaryWindow",opts.binary_window) >> GetOpt::Option("vocabSize",opts.vocab_size)
	>> GetOpt::Option("treeSize",opts.tree_size);
	opts.grid_diagnal = (size_t)sqrt(double(opts.batch_size)) +1;
	opts.grid_dim = dim3(opts.grid_diagnal,opts.grid_diagnal,1);
	options opt_immutable(opts);
	std::cout << "launching kernels with configuration: " << std::endl;
	std::cout << "training directory: " <<  opt_immutable.training << std::endl;
	std::cout << "tree encoding of words: " << opt_immutable.huff_code << std::endl;
	std::cout << "tree point index of words: " << opt_immutable.huff_point << std::endl;
	std::cout << "number of feature: " << opt_immutable.feature_size << std::endl;
	std::cout << "size of vocabulary: " << opt_immutable.vocab_size << std::endl;
	std::cout << "size of tree internal node" << opt_immutable.tree_size << std::endl;
	std::cout << "word feature saving directory: " << opt_immutable.word_feature << std::endl;
	std::cout << "point feature saving directory: " << opt_immutable.point_feature << std::endl;
	std::cout << "batch size: " << opt_immutable.batch_size << std::endl;
	std::cout << "validation ratio: " << opt_immutable.validation_ratio << std::endl;
	std::cout << "iteration: " << opt_immutable.iteration << std::endl;
	std::cout << "learning rate: " << opt_immutable.learning_rate << std::endl;
	std::cout << "binary training file cache directory: " << opt_immutable.binary_training << std::endl;
	std::cout << "binary window file cache directory: " << opt_immutable.binary_window << std::endl;

	run(opt_immutable);
	return 0;
}


int main_test()
{
	options_mutable opts;
	opts.training = training_dir;
	opts.huff_code = huff_code_dir;
	opts.huff_point = huff_point_dir;
	opts.point_feature = point_feature_save_dir;
	opts.word_feature = word_feature_save_dir;
	opts.batch_size = 90000;
	opts.grid_diagnal = 300;
	opts.feature_size = 256;
	opts.grid_dim = dim3(opts.grid_diagnal,opts.grid_diagnal,1);
	opts.validation_ratio = 10;
	opts.iteration = 2;
	opts.learning_rate = 0.01;
	opts.binary_training = training_binary;
	opts.binary_window = training_window_binary;
	options opt_immutable(opts);
	run(opt_immutable);
	return 0;
}



void environment_setup(environment & env,options & opt)
{
	env.file_ptr_word_count = 0;
	load_huffman_code( env.huff_code_host,env.huff_indx_host,opt.huff_code.c_str());
	load_huffman_point(env.huff_point_host,env.huff_indx_host,opt.huff_point.c_str());
	binary_process(opt.training.c_str(),opt.binary_training.c_str(),env.corpus_delimiter);
	env.huff_code_device.allocate_and_copy(env.huff_code_host,"huff code free failed");
	env.huff_point_device.allocate_and_copy(env.huff_point_host,"huff point free failed");
	env.huff_indx_device.allocate_and_copy(env.huff_indx_host,"huff index free failed");
	env.corpus_device.allocate(opt.batch_size,"corpus free failed");
	env.window_device.allocate(opt.batch_size,"window free failed");
	env.code_binder.rebind(env.huff_code_device.get(),env.huff_code_device.size());
	env.point_binder.rebind(env.huff_point_device.get(),env.huff_point_device.size());
	env.indx_binder.rebind(env.huff_indx_device.get(),env.huff_indx_device.size());
	env.corpus_binder.rebind(env.corpus_device.get(),env.corpus_device.size());
	env.window_binder.rebind(env.window_device.get(),env.window_device.size());
	allocate_model_parameters(env.point_feature,env.word_feature,opt);
}



int run(options & opt)
{
	environment env;
	environment_setup(env,opt);
	std::cout <<"tree "<< opt.tree_size << std::endl;
	for(size_t i=0; i<opt.iteration;i++)
	{
		std::cout << "network training iteration " << i << std::endl;
		env.file_ptr_word_count = 0;
		random_window_cut(env,opt);
		size_t counter =0;
		while(next_batch(env,opt))
		{
			counter++;
			if(counter%opt.validation_ratio==0)
				validation_parallel(env,opt,env.corpus_device.size()-MAXWINDOW);
			else
				training(env,opt,env.corpus_device.size() - MAXWINDOW);
		}
	}
	std::cout <<"done" << std::endl;
	write_results(env,opt);
	return 0;
}
