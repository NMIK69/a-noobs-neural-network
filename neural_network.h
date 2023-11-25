#ifndef SCRAPWORK_H
#define SCRAPWORK_H
#include <stddef.h>

#define NN_ENABLE_MULTI_PROCESSING 0

enum Activation_Functions {
	RELU,
	SIGMOID,
	TANH,
	SWISH
};

enum Learning_Rate_Behaviour {
	FIXED,
	SCHEDULE,
	ADAPTIVE
};

struct Neural_Network{
        int num_layers;
        int *layer_size;
        float **layer;
        
        float ***weight; 
        
	float **bias;

	float learning_rate;
	int iterations;

	/* The activation function used for the hidden layers */
	enum Activation_Functions af_hl; 

	/* The activation function used for the ouput layer */
	enum Activation_Functions af_ol; 

	enum Learning_Rate_Behaviour lr_behav;
};

struct Training_Data {
	float **input_data;
	float **expected_output;

	int num_entries;

	int input_size;
	int output_size;
};
void Training_Data_free(struct Training_Data *md);

struct Neural_Network* nn_create(int num_layers,
				 const int *layer_size);
void nn_free(struct Neural_Network *nn);
void nn_train(struct Neural_Network *nn, const struct Training_Data *td);
int nn_predict(struct Neural_Network *nn, float *input_data);

void nn_save_to_file(const struct Neural_Network *nn, const char *filename);
struct Neural_Network *nn_load_from_file(const char *filename);

#endif
