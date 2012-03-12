#include <cv.h>
#include <iostream>


class WindowInfo{
public:
	WindowInfo(const cv::Mat mat, int wSize);
	~WindowInfo();
	int xWindows();
	int yWindows();
	int xOffsetWindows();
	int yOffsetWindows();
	int xyOffsetWindows();
	int totalWindows();
	int windowSize();
	int windowOffset();
	int* subWindowOffsets();

private:	

	void computePositions();

	cv::Mat img;
	int winSize;
	int offset;
	
	int* offsets;
	

};