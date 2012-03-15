#include <cv.h>
#include <highgui.h>
#include <iostream>

#include <time.h>

#include "integral.h"
#include "cuda_detect_faces.h"
#include "window_info.h"



using namespace cv;
using namespace std; 

Mat getImg(int argc, char** argv);
Mat integral_image(Mat &img);
void display_image(Mat &img, string title, int x=0, int y=0);
Mat detect_faces(Mat &integral_img);
Mat composite_heat_image(Mat &image, Mat &heat_map);



int main( int argc, char** argv )
{ 
	Mat image = getImg(argc, argv);
	Mat integral_img = integral_image(image);
	
	Mat heat_map = detect_faces(integral_img);
	
	normalize(integral_img, integral_img, 0, 1, NORM_MINMAX);
	normalize(heat_map, heat_map, 0, 1, NORM_MINMAX);
	
	Mat heat_img = composite_heat_image(image, heat_map);
	
	if(argc > 2 && (strcmp(argv[2], "-v") == 0)){
		display_image(image, "Original");
		display_image(integral_img, "Integral Image", image.cols);
		display_image(heat_map, "Heat Map", 0, image.rows+40);
		display_image(heat_img, "Heat Image", image.cols, image.rows+40);
	
	
		waitKey(0);											 // Wait for a keystroke in the window		
	}

	
	
	return 0;
}


Mat getImg(int argc, char** argv){
	// Check for filename
	if( argc < 2) {
	 cout <<" Usage: detect <image>" << endl;
	 exit(-1);
	}
	
	// Read the file
	Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	// Check for invalid input
	if(! image.data){
		cout <<  "Could not open or find the image" << std::endl ;
		exit(-1);
	}
	return image;
}


Mat integral_image(Mat &img){
	img.convertTo(img, CV_32F, 1.0/255);
	Mat int_img(img.rows+1, img.cols+1, CV_32F, 0.0f);
	Mat cv_int_img(img.rows+1, img.cols+1, CV_32F, 0.0f);
	Mat mask(int_img, Range(1, int_img.rows), Range(1, int_img.cols));
	img.copyTo(mask);
	
	// OpenCV Integral IMG
	clock_t start = clock();
	integral(img, cv_int_img, CV_32F);
	
	cout << "OpenCV Intergral time: " << ((double)clock() - start) / CLOCKS_PER_SEC << endl;
	
	// CUDA Integral IMG
	double ex_time = cuda_integrate_image((float*)int_img.data, int_img.rows, int_img.cols, int_img.step[0]/int_img.step[1]);
	
	cout << "CUDA Intergral time: " << ex_time << endl;
	
	return int_img;
}


void display_image(Mat &img, string title, int x, int y){
	namedWindow(title); 
	imshow(title, img );
	cvMoveWindow(title.c_str(), x, y);
}


Mat detect_faces(Mat &integral_img){
	int winSize = min(integral_img.rows, integral_img.cols)/4;
	cout 	<< "Window Size:     	" << winSize		<< endl;
	
	WindowInfo winInfo(integral_img, winSize);
	Mat heat_map(integral_img.rows, integral_img.cols, CV_32F, 0.0f);

	cout << endl << endl<< "CASCADING" << endl;
	cuda_detect_faces((float*)integral_img.data,integral_img.rows, integral_img.cols, integral_img.cols, winInfo.subWindowOffsets(), winInfo.totalWindows(), winInfo.windowSize(), (float*) heat_map.data);

	cout << endl << endl<< "BRUTE FORCE" << endl;
	cuda_detect_faces2((float*)integral_img.data,integral_img.rows, integral_img.cols, integral_img.cols, winInfo.subWindowOffsets(), winInfo.totalWindows(), winInfo.windowSize(), (float*) heat_map.data);
	
	return heat_map;
}

Mat composite_heat_image(Mat &image, Mat &heat_map){
	Mat ret = image.clone(); 
	for(size_t i = 0; i < image.rows; ++i){
		for(size_t j = 0; j < image.cols; ++j){
			ret.row(i).col(j) = image.row(i).col(j) * heat_map.row(i+1).col(j+1);
		}
	}
	return ret;
}



