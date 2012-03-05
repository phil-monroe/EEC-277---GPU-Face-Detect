#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "integral.h"

using namespace cv;
using namespace std; 

void integral_image(Mat &img);

int main( int argc, char** argv )
{ 
	if( argc != 2) 
	{
	 cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
	 return -1;
	}
	
	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);	// Read the file

	if(! image.data )                              // Check for invalid input
	{
		cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}
	integral_image(image);
	namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	imshow( "Display window", image );                   // Show our image inside it.

	waitKey(0);											 // Wait for a keystroke in the window
	return 0;
}

void integral_image(Mat &img){
	cout << test() << endl;
	// for(int row=0; row < img.rows; ++row){
	// 	for(int col=0; col < img.cols; ++col){
	// 		// flt_img.data[row*img.step[0]+col*flt_img.step[1]] = img.data[row*img.step+col] + img.data[row*img.step+col-1] + img.data[(row-1)*img.step+col] - img.data[(row-1)*img.step+col-1];
	// 		img.row(row).col(col) = 0.0;
	// 	}
	// }
	// 
	// img.row(100).col(100) = 1.0;
	// 
	// Mat flt_img(img.rows, img.cols, CV_64FC1);
	// Mat int_img(img.rows, img.cols, CV_64FC1);
	// 
	// img.convertTo(flt_img, CV_64FC1);
	// integral(img, int_img, CV_64FC1);
	//                 // Show our image inside it.
	// 
	// for(int row=1; row<img.rows; ++row){
	// 	flt_img.row(row).col(0) += flt_img.row(row-1).col(0);
	// }
	// 
	// for(int col=1; col<img.cols; ++col){
	// 	flt_img.row(0).col(col) += flt_img.row(0).col(col-1);
	// }
	// 
	// for(int row=1; row < img.rows; ++row){
	// 	for(int col=1; col < img.cols; ++col){
	// 		flt_img.row(row).col(col) = flt_img.row(row).col(col) + flt_img.row(row-1).col(col) + flt_img.row(row).col(col-1) - flt_img.row(row-1).col(col-1);
	// 	}
	// }
	// double m;
	// minMaxLoc(flt_img, NULL, &m);
	// cout << m << endl;
	// cout << numeric_limits<double>::max( ) << endl;
	// for(int row=1; row < img.rows; ++row){
	// 	for(int col=1; col < img.cols; ++col){
	// 		// flt_img.data[row*img.step[0]+col*flt_img.step[1]] = img.data[row*img.step+col] + img.data[row*img.step+col-1] + img.data[(row-1)*img.step+col] - img.data[(row-1)*img.step+col-1];
	// 		flt_img.row(row).col(col) = flt_img.row(row).col(col)/m;
	// 	}
	// }
	// 
	// minMaxLoc(int_img, NULL, &m);
	// cout << m << endl;
	// cout << numeric_limits<double>::max( ) << endl;
	// for(int row=1; row < img.rows; ++row){
	// 	for(int col=1; col < img.cols; ++col){
	// 		// flt_img.data[row*img.step[0]+col*flt_img.step[1]] = img.data[row*img.step+col] + img.data[row*img.step+col-1] + img.data[(row-1)*img.step+col] - img.data[(row-1)*img.step+col-1];
	// 		int_img.row(row).col(col) = int_img.row(row).col(col)/m;
	// 	}
	// }
	// 
	// namedWindow( "after Integral", CV_WINDOW_AUTOSIZE );// Create a window for display.
	// imshow( "after Integral", flt_img );                   // Show our image inside it.
	// 
	// namedWindow( "Integral", CV_WINDOW_AUTOSIZE );// Create a window for display.
	// imshow( "Integral", int_img );
}
