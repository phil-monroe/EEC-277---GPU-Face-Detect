CFLAGS	= $(shell pkg-config --cflags opencv)
CFLAGS  += -m32 -I./include
LIBS		= $(shell pkg-config --libs opencv)
LIBS	  += -L/usr/local/cuda/lib -lcudart

KERNELS = 	src/kernels/identify1.cu \
				src/kernels/identify2.cu \
				src/kernels/identify3.cu \
				src/kernels/identify4.cu \
				src/kernels/identify_glasses.cu \
				src/kernels/brute_merge_results.cu

all: detect
	
run: all test.jpg
	@echo "\nRunning...\n"
	./detect test.jpg
	
clean:
	- rm detect *.o a.out
	
main.o: src/main.cpp src/integral.h src/cuda_detect_faces.h
	@echo Compiling main.cpp
	@nvcc -c src/main.cpp $(CFLAGS)

window_info.o: src/window_info.cpp src/window_info.h
	@echo Compiling window_info.cpp
	@nvcc -c src/window_info.cpp $(CFLAGS)

integral.o: src/integral.cu src/integral.h cuda_helpers.o
	@echo "Compiling integral.cu"
	@nvcc -c src/integral.cu $(CFLAGS)
	
	
cuda_helpers.o: src/cuda_helpers.cu src/cuda_helpers.h
	@echo "Compiling cuda_helpers.cu"
	@nvcc -c src/cuda_helpers.cu $(CFLAGS)
	
cuda_detect_faces.o: src/cuda_detect_faces.h src/cuda_detect_faces.cu $(KERNELS)
	@echo "Compiling cuda_detect_faces.cu"
	@nvcc -c src/cuda_detect_faces.cu $(CFLAGS)

detect: main.o integral.o cuda_detect_faces.o cuda_helpers.o window_info.o
	@echo "\nLinking..."
	@nvcc main.o integral.o cuda_detect_faces.o cuda_helpers.o window_info.o $(LIBS) -o detect -m32
	