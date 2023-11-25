#ifndef MNIST_H
#define MNIST_H

#define MNIST_NUM_PIXEL 784
#define MNIST_NUM_OUTCOMES 10

#include "neural_network.h"

struct Training_Data *load_mnist_from_file(const char *img_file,
					const char *lbl_file,
					int max_datasets);
#endif
