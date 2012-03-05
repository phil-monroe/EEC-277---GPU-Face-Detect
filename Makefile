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
	@g++ -c src/main.cpp $(CFLAGS)
	@echo "Compiling main.cpp"

integral.o: src/integral.cu src/integral.h
	@nvcc -c src/integral.cu -m64
	@echo "Compiling integral.cu"
	

detect: main.o integral.o
	@g++ main.o integral.o $(LIBS) -o detect
	@echo "\nLinking main.o, integral.o"
	