CFLAGS	= $(shell pkg-config --cflags opencv)
LIBS		= $(shell pkg-config --libs opencv)
LIBS	  += -L/usr/local/cuda/lib -lcudart

all: detect
	
run: all test.jpg
	@echo "\nRunning...\n"
	./detect test.jpg
	
clean:
	rm detect *.o a.out
	
main.o: src/main.cpp src/integral.h
	@echo Compiling main.cpp
	@g++ -c src/main.cpp $(CFLAGS) -m32

window_info.o: src/window_info.cpp src/window_info.h
	@echo Compiling window_info.cpp
	@g++ -c src/window_info.cpp $(CFLAGS) -m32

integral.o: src/integral.cu src/integral.h
	@echo "Compiling integral.cu"
	@nvcc -c src/integral.cu -m32
	
	
classifiers.o: src/classifiers.cu src/classifiers.h
	@echo "Compiling classifiers.cu"
	@nvcc -c src/classifiers.cu -m32
	
cuda_helpers.o: src/cuda_helpers.cu src/cuda_helpers.h
	@echo "Compiling cuda_helpers.cu"
	@nvcc -c src/cuda_helpers.cu -m32
	

detect: main.o integral.o classifiers.o cuda_helpers.o
	@echo "\nLinking..."
	@g++ main.o integral.o classifiers.o cuda_helpers.o $(LIBS) -o detect -m32
	