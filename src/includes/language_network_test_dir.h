/*
 * language_network_test_dir.h
 *
 *  Created on: 2013-12-23
 *      Author: Chen Pu
 */

#ifndef LANGUAGE_NETWORK_TEST_DIR_H_
#define LANGUAGE_NETWORK_TEST_DIR_H_

static const  bool WINDOWS_OS = true;

static const char* huff_code_dir_windows = "C:\\Users\\Pu\\Downloads\\HuffmanCode1";
static const char* huff_point_dir_windows = "C:\\Users\\Pu\\Downloads\\HuffmanPoint1";
static const char* training_dir_windows = "C:\\Users\\Pu\\Downloads\\news.all.2012.nn.train";
static const char* word_feature_save_dir_windows = "C:\\Users\\Pu\\Downloads\\word_feature";
static const char* point_feature_save_dir_windows = "C:\\Users\\Pu\\Downloads\\point_feature";
static const char* training_dir_linux = "/Users/Pu/Downloads/news.all.2012.nn.train.head.h";
static const char* huff_code_dir_linux  = "/Users/Pu/Desktop/HuffmanCode1";
static const char* huff_point_dir_linux = "/Users/Pu/Desktop/HuffmanPoint1";
static const char* word_feature_save_dir_linux = "/Users/Pu/Desktop/word_feature";
static const char* point_feature_save_dir_linux = "/Users/Pu/Desktop/point_feature";
static const char* word_feature_save_dir_before = "C:\\Users\\Pu\\Downloads\\word_feature_b";
static const char* point_feature_save_dir_before = "C:\\Users\\Pu\\Downloads\\point_feature_b";
static const char* gradient_dir = "C:\\Users\\Pu\\Downloads\\nn.gradient";
static const char* hidden_dir = "C:\\Users\\Pu\\Downloads\\nn.hidden";
static const char* likelihood_dir = "C:\\Users\\Pu\\Downloads\\nn.likelihood";
static const char* huff_point_dir = WINDOWS_OS?huff_point_dir_windows:huff_point_dir_linux;
static const char* huff_code_dir = WINDOWS_OS?huff_code_dir_windows:huff_code_dir_linux;
static const char* training_dir = WINDOWS_OS?training_dir_windows:training_dir_linux;
static const char* word_feature_save_dir = WINDOWS_OS?word_feature_save_dir_windows:word_feature_save_dir_linux;
static const char* point_feature_save_dir = WINDOWS_OS?point_feature_save_dir_windows:point_feature_save_dir_linux;
static const char* training_binary = WINDOWS_OS?"C:\\Users\\Pu\\Downloads\\news.all.2012.nn.train.bin":"news.all.2012.nn.train.bin";
static const char* training_window_binary  =WINDOWS_OS?"C:\\Users\\Pu\\Downloads\\news.window.bin":"news.window.bin";


#endif /* LANGUAGE_NETWORK_TEST_DIR_H_ */
