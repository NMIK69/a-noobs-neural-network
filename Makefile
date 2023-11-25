CC=gcc
CFLAGS = -Wall -Wextra -MMD -MP -Wmissing-prototypes -std=c99 -pedantic -fopenmp -fopenmp-simd
OPTIMIZE=-O3 -ftree-vectorize -ffast-math
LDFLAGS = -lm -fopenmp -fopenmp-simd

SRCS=$(wildcard *.c)
OBJS=$(patsubst %.c, %.o, $(SRCS))

# header dependencies
DEPS=$(wildcard *.h)

TARGET=nn_example

all: $(TARGET)


debug: CFLAGS += -ggdb
debug: OPTIMIZE = -O0
debug: $(TARGET)

$(TARGET) : $(OBJS)
	$(CC) $(OPTIMIZE) $^ $(CFLAGS) $(LDFLAGS) -o $@

%.o : %.c $(DEPS)
	$(CC) $(OPTIMIZE) $(CFLAGS) -c $< $(LDFLAGS) -o $@


.PHONY : clean
clean :
	rm -f $(TARGET)
	rm -f *.o
	rm -f *.d
