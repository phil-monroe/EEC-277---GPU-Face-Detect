
#include "window_info.h"


WindowInfo::WindowInfo(cv::Mat mat, int wSize):img(mat){
	std::cout << "Image Size " << ": (" << img.rows << " , " << img.cols << ")" << std::endl;
	
	winSize = wSize;
	offset = winSize/2;
	offsets = new int[totalWindows()];
	computePositions();
}


WindowInfo::~WindowInfo(){
	delete offsets;
}


int WindowInfo::xWindows(){
	return img.cols/windowSize();
}


int WindowInfo::yWindows(){
	return img.rows/windowSize();
}


int WindowInfo::xOffsetWindows(){
	return (img.cols + offset)/windowSize() -1;
}


int WindowInfo::yOffsetWindows(){
	return (img.rows + offset)/windowSize()-1;
}

int WindowInfo::xyOffsetWindows(){
	return xOffsetWindows() * yOffsetWindows();
}


int WindowInfo::totalWindows(){
	return 	xWindows() * yWindows() 
			 + xOffsetWindows() * yWindows() 
			 + xWindows() * yOffsetWindows() 
			 + xyOffsetWindows();
}


int WindowInfo::windowSize(){
	return winSize;
}

int WindowInfo::windowOffset(){
	return offset;
}

int* WindowInfo::subWindowOffsets(){
	return offsets;
}

void WindowInfo::computePositions(){
	int win = 0;
	for(int i = 0; i <= img.rows - windowSize(); i+= windowOffset()){
		for(int j = 0; j <= img.cols - windowSize(); j += windowOffset()){
			
			offsets[win] = i*img.cols + j; 
			
			std::cout << "Window " << win << ": (" << i << ", " << j << ") " << offsets[win]  << std::endl;
			win++;
		}
	}
}