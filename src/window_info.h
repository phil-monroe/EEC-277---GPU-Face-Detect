#include <cv.h>


class WindowInfo{
	WindowInfo(const cv::Mat mat, int wSize);
	~WindowInfo();
	int xWindows();
	int yWindows();
	int totalWindows();
	int windowSize();
	int* xPositions();
	int* yPositions();
	
private:
	const cv::Mat img;
	int winSize;
};