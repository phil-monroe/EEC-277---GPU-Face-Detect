
#include "window_info.h"


WindowInfo::WindowInfo(const cv::Mat mat, int wSize):img(mat){
	winSize = wSize;
	xPos = new int[totalWindows()];
	yPos = new int[totalWindows()];
	computePositions();
}


WindowInfo::~WindowInfo(){
	delete[] xPos;
	delete[] yPos;
}


int WindowInfo::xWindows(){
	return img.cols/windowSize();
}


int WindowInfo::yWindows(){
	return img.rows/windowSize();
}


int WindowInfo::totalWindows(){
	return xWindows() * yWindows();
}


int WindowInfo::windowSize(){
	return winSize;
}


int* WindowInfo::xPositions(){
	return xPos;
}


int* WindowInfo::yPositions(){
	return yPos;
}

void WindowInfo::computePositions(){
	for(int xWin = 0; xWin < xWindows(); ++xWin){
		for(int yWin = 0; yWin < yWindows(); ++yWin){
			int win = xWin*xWindows() + yWin;
			xPos[win] = xWin*windowSize();
			yPos[win] = yWin*windowSize();
			
			std::cout << "Window " << win << ": (" << xPos[win] << " , " << yPos[win] << ")" << std::endl;
		}
	}
}