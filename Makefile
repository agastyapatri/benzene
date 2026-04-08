CC = clang++ 
CPPFLAGS = -std=c++20 -Wall -Wextra -g -O0 -DDEBUG 
SRCS = src/tensor.cpp main.cpp 
INCFLAGS = -Isrc/
all: main 
main: 
	clang++ -std=c++20 -Wall -Wextra -g -O0 -DDEBUG -Isrc/ src/tensor.cpp main.cpp -o main

clean: 
	rm main 
