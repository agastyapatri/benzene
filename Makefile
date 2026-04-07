CC = clang++ 
CPPFLAGS = -Wall -Wextra -g -O0 -DDEBUG 
SRCS = src/tensor.cpp main.cpp 
INCFLAGS = -Isrc/
all: main 
main: 
	$(CC) $(CFLAGS) $(INCFLAGS) $(SRCS) -o $@

clean: 
	rm main 
	rm *.o *.out
