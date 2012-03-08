#ifndef _CLASSIFIERS_H_
#define _CLASSIFIERS_H_

// Test Classifier
extern "C" void cuda_test_classifier(int* winX, int* winY, size_t stride, int winSize, int featSize, float* intImg, bool* results);

// Real Classifiers
extern "C" void cuda_classifier_1(int* winX, int* winY, size_t stride, int winSize, int featSize, float* intImg, bool* results);
extern "C" void cuda_classifier_2(int* winX, int* winY, size_t stride, int winSize, int featSize, float* intImg, bool* results);
extern "C" void cuda_classifier_3(int* winX, int* winY, size_t stride, int winSize, int featSize, float* intImg, bool* results);
extern "C" void cuda_classifier_4(int* winX, int* winY, size_t stride, int winSize, int featSize, float* intImg, bool* results);
extern "C" void cuda_classifier_5(int* winX, int* winY, size_t stride, int winSize, int featSize, float* intImg, bool* results);

#endif // _CLASSIFIERS_H_