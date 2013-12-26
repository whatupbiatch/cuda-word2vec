/*
 * language_network_memory.hpp
 *
 *  Created on: 2013-12-22
 *      Author: Chen Pu
 */


#ifndef LANGUAGE_NETWORK_MEMORY_HPP_
#define LANGUAGE_NETWORK_MEMORY_HPP_
#include <cuda.h>
#include <vector>
#include <memory>
#include <iostream>
#include "language_network_constants.h"
#include "language_network_typedefs.h"


template<typename T> class cuda_auto_ptr
{
	T* __internal;
	bool __released;
	char* __message;
	size_t __size;
public:
	cuda_auto_ptr(T* resource = NULL,size_t size = 0,char* message = NULL):
		__internal(resource),__released(true),__message(message),__size(size)
	{
		if(resource!=NULL)
		{
			__size = size;
			__released = false;
		}
	}

	~cuda_auto_ptr()
	{
		release();
	}

	T* get()
	{
		return __internal;
	}

	void allocate(size_t size,char* message = NULL)
	{
		release();
		 __message = message;
		cudaError_t error = cudaMalloc((void**)& __internal,size*sizeof(T));
		if(error != cudaSuccess)
		{
			if(__message !=NULL)
				std::cerr << __message  << std::endl;
			std::cerr << cudaGetErrorString(error) << std::endl;
		}
		__size = size;
		__released = false;
	}

	void memset(int value)
	{
		if(__internal!=NULL && !__released)
		{
			cudaError_t error = cudaMemset(__internal,value,sizeof(T)*__size);
		}
	}

	void release()
	{
		if(__released) return;
		cudaError_t error =  cudaFree(__internal);
		if(error!= cudaSuccess)
		{
			if(__message!=NULL)
				std::cerr << __message << std::endl;
			std::cerr << cudaGetErrorString(error) <<std::endl;
		}
		//test output, remove at release
		__size = 0;
		__released = true;
	}

	void allocate_and_copy(size_t size, T* cpu_ptr,char* message = "allocate and copy failed")
	{
		release();
		allocate(size,message);
		cudaError_t error = cudaMemcpy(__internal,cpu_ptr,sizeof(T)*size,cudaMemcpyHostToDevice);
		if(error!=cudaSuccess)
		{
			std::cerr << cudaGetErrorString(error) << std::endl;
			std::cerr << message << std::endl;
		}
		__size =size;
		__released = false;
	}

	bool refill(const T* cpu_ptr,char* message = "refill failed")
	{
		if(__released || __size ==0) return false;
		cudaError_t error = cudaMemcpy(__internal,cpu_ptr,sizeof(T)*__size,cudaMemcpyHostToDevice);
		if(error!=cudaSuccess)
		{
			std::cerr << cudaGetErrorString(error) << std::endl;
			std::cerr << message << std::endl;
			return false;
		}
		return true;

	}

	void allocate_and_copy(std::vector<T> & vec,char* message = NULL)
	{
		allocate_and_copy(vec.size(),&vec[0],message);
	}

	T* fetch_raw()
	{
		if(__size == 0) return 0;
		void* ret = malloc(sizeof(T)*__size);
		cudaError_t error = cudaMemcpy(ret,__internal,__size*sizeof(T),cudaMemcpyDeviceToHost);
		if(error != cudaSuccess)
		{
		   std::cerr << cudaGetErrorString(error) << std::endl;
		}

		return (T*)ret;
	}

	size_t size()
	{
		return __size;
	}

};

template<typename T> class binder
{
	texture<T> * __inner;
	bool unbinded;
	public:
	binder(texture<T> & tex,T* resource = NULL, size_t size =0):__inner(&tex),unbinded(false)
	{
		if( resource == NULL)
		{
			unbinded = true;
		}
		else
		{
			cudaError_t error = cudaBindTexture(NULL,tex,resource,size*sizeof(T));
			if(error != cudaSuccess)
			{
				std::cerr << cudaGetErrorString(error) << std::endl;
				std::cerr << "binder construction failed" << std::endl;
				unbinded = true;
			}
		}
	}

	~binder()
	{
		unbind();
	}

	void unbind()
	{
		if( unbinded) return;
		cudaUnbindTexture(*__inner);
		unbinded = true;
	}

	void rebind(T* resource, size_t size)
	{
		unbind();
		cudaError_t error = cudaBindTexture(NULL,*__inner,resource,size*sizeof(T));
		if(error != cudaSuccess)
		{
			std::cerr << cudaGetErrorString(error) << std::endl;
			std::cerr << "binder construction failed" << std::endl;
			unbinded = true;
		}
		unbinded = false;
	}
};


#endif /* LANGUAGE_NETWORK_MEMORY_HPP_ */
