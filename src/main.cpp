#include <cv.h>
#include <highgui.h>
#include <iostream>

#include <time.h>

#include "integral.h"
#include "window_info.h"


using namespace cv;
using namespace std; 

Mat getImg(int argc, char** argv);
Mat integral_image(Mat &img);
void display_image(Mat &img, string title, int x=0, int y=0);
void detect_faces(Mat &integral_img);


int main( int argc, char** argv )
{ 
	Mat image = getImg(argc, argv);
	Mat integral_img = integral_image(image);
	
	detect_faces(integral_img);
	
	
	display_image(image, "Original");
	display_image(integral_img, "Integral Image", image.rows);
	

	waitKey(0);											 // Wait for a keystroke in the window
	return 0;
}


Mat getImg(int argc, char** argv){
	// Check for filename
	if( argc != 2) {
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
	start = clock();
	cuda_integrate_image((float*)int_img.data, int_img.rows, int_img.cols, int_img.step[0]/int_img.step[1]);
	
	cout << "CUDA Intergral time: " << ((double)clock() - start) / CLOCKS_PER_SEC << endl;
	
	normalize(int_img, int_img, 0, 1, NORM_MINMAX);
	// normalize(cv_int_img, cv_int_img, 0, 1, NORM_MINMAX);
	// 
	// // Display Integral
	// 
	// 
	// display_image(int_img, "Computed Integral", img.cols);
	// display_image(cv_int_img, "OpenCV Integral", img.cols*2);
	
	return int_img;
}


void display_image(Mat &img, string title, int x, int y){
	namedWindow(title); 
	imshow(title, img );
	cvMoveWindow(title.c_str(), x, y);
}


void detect_faces(Mat &integral_img){
	WindowInfo winInfo(integral_img, 100);
	cout 	<< "Window Size:     	" << winInfo.windowSize()		<< endl
			<< "Number of Windows: 	" << winInfo.totalWindows()	<< endl
			<< "Number of xWindows: 	" << winInfo.xWindows()	<< endl
			<< "Number of yWindows: 	" << winInfo.yWindows()	<< endl
			<< "Number of xOffsetWindows: 	" << winInfo.xOffsetWindows()	<< endl
			<< "Number of yOffsetWindows: 	" << winInfo.yOffsetWindows()	<< endl
			<< "Number of xyOffsetWindows: 	" << winInfo.xyOffsetWindows()	<< endl;
			
	int* subWindows = winInfo.subWindowOffsets();
	float* img_data = (float*) integral_img.data;
	for(int i = 0; i < winInfo.totalWindows(); ++i){
		img_data[subWindows[i]] = 0.0;
	}
}




