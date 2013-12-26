cuda-word2vec
=============

cuda implementation of CBOW model

Features
---------

* parallel speedup with Nvidia GPU
* requires constant memory for arbitrarily large training set
* stream implementation requires no training data random access
* automatic validation and display validation data negative loglikelihood during training
* support custom word binary tree construction.
**improve model performance with your own binary tree constructed from any other unsupervised model**


Updates 0.1
---------
* optimize GPU memory IO efficiency exploiting locality
* optimize memory performance with cuda texture
* optimize disk to memory IO efficiency and save parsing overhead with binary preprocessing of training data
* back propagation gradient-check test completed
* implement automatic cuda memory management utilities to manage GPU resource


Requirement
---------
* cuda 5.5
* cuda Thrust
* nvcc compiler
* 1GB or more GPU memory ( 2GB+ is recommended)



Compilation
---------
* Linux: if nvcc command is available, compile with makefile 
* Windows: compile the source file manually with -arch=compute_30 and -code=sm_30
* Source Files: language_network_main.cu language_network_utility.cu language_network_kernel.cu includes/optParser/getopt_pp.cpp


Input Format
---------
* corpus should be preprocessed into a text file with each line a document.
* words in document should be convert into ints by any bijective mapping M [1-V] <-> vocabulary (store separtely as a convertable)
* Store word binary tree in two files: tree_point and tree_code. 
* line N of tree_point should be a path through internal nodes from root to leaf word-N.
* line N of tree_code should be a binary sequence cooresponding to the navigation of the above path (from root to leaf-N)
* each word id n must have a code line and a point line in the tree files
* a java implementation of huffman tree file constructor is provided, use if new tree construction is not needed


Usage
---------
command line options:

* --train path to training data
* --code path to tree_code data
* --point path to tree_point data
* --pFeature saving path of tree point feature
* --wFeature saving path of word feature
* --batch training batch size (length of corpus segment by words, use suitable batch size that fits in your GPU memory)
* --vRatio validation ratio ( integral value, validation will be performed after every vRatio training batch)
* --iter iterations of the training pass through training data
* --learningRate constant leanrning rate
* --binaryTraining cache path for binary preprocessing of the training set( delete manually after training)
* --binaryWindow cache path for random windows data (delete manually after training)
* --vocabSize vocabulary size
* --treeSize size of inernal nodes of the binary tree


Typedefs and Numerical Issues
---------

all types are found in src/includes/language_network_typedefs.h, modify if needed




Benchmarks and Profiling
---------
coming soon 


