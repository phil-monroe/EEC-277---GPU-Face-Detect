#include <cv.h>
#include <iostream>


class WindowInfo{
public:
	WindowInfo(const cv::Mat mat, int wSize);
	~WindowInfo();
	int xWindows();
	int yWindows();
	int totalWindows();
	int windowSize();
	int* xPositions();
	int* yPositions();

private:	

	void computePositions();

	const cv::Mat img;
	int winSize;
	
	int* xPos;
	int* yPos;
	

};