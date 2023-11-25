#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

/* back propagation data */
struct bp_data {
	float ***d_weight;
	float **error;
};
static struct bp_data *bp_data_create(const struct Neural_Network *nn);
static void free_bpd(struct bp_data *bpd, const struct Neural_Network *nn);
static void bp_data_make_error(struct bp_data *bpd, const struct Neural_Network *nn);
static void bp_data_make_d_weight(struct bp_data *bpd, const struct Neural_Network *nn) ;

/* activation functions and their derivatives */
static inline float act_func(enum Activation_Functions af, float n);
static inline float act_func_deriv(enum Activation_Functions af, float n);
static inline float sigmoid(float n);
static inline float sigmoid_deriv(float out_val);
static inline float act_tanh(float n);
static inline float act_tanh_deriv(float out_val);
static inline float relu(float n);
static inline float relu_deriv(float out_val);
static inline float swish(float n);
static inline float swish_deriv(float out_val);

/* cost function derivative with respect to the ouput layer*/
static inline float pcost_pout(float out_val, float expected_val);

/* learning rate adaption algorithm */
static inline void lr_schedule(struct Neural_Network *nn);
static inline void adept_lr(struct Neural_Network *nn);


/* used for creating the neural network */
static void make_layer_size(struct Neural_Network *nn, int num_layers, const int *layer_size);
static void make_layers(struct Neural_Network *nn);
static inline float uniform_random_float();
static void randomize_weights(struct Neural_Network *nn);
static void make_weights(struct Neural_Network *nn);
static int make_bias(struct Neural_Network *nn);

/* forward and back propagation */
static void fprop(struct Neural_Network *nn);
static void bprop(const struct Neural_Network *nn, int tdi,
		  const struct Training_Data *td,
		  struct bp_data *bpd);

static void bprop_out_layer(const struct Neural_Network *nn,
			    struct bp_data *bpd,
			    const struct Training_Data *td,
			    int tdi);

static void bprop_hidden_layer(const struct Neural_Network *nn, int rli,
			       struct bp_data *bpd);

static void update_weights(struct Neural_Network *nn, float ***d_weight);

static void update_bias(struct Neural_Network *nn, float **error);

static inline float error_out_layer(const struct Neural_Network *nn,
						    int node_index,
					  	    const struct Training_Data *td, 
						    int tdi);

static inline float error_hidden_layer(const struct Neural_Network *nn, 
				       const float *error, int rli, int node_index);


/********************************************************************/
void nn_train(struct Neural_Network *nn, const struct Training_Data *td)
{
	
	assert(td->input_size == nn->layer_size[0]);
	assert(td->output_size == nn->layer_size[nn->num_layers - 1]);

	struct bp_data *bpd = bp_data_create(nn);

	//float initial_lr = nn->learning_rate;

	for(int i = 0; i < nn->iterations; i++) {
		printf("EPOCH: %d\n", i + 1);

		for(int j = 0; j < td->num_entries; j++) {

			nn->layer[0] = td->input_data[j];
			fprop(nn);
			bprop(nn, j, td, bpd);
			update_weights(nn, bpd->d_weight);
			update_bias(nn, bpd->error);
		}	

		adept_lr(nn);
	}

	free_bpd(bpd, nn);
}


int nn_predict(struct Neural_Network *nn, float *input_data)
{
	
	nn->layer[0] = input_data;

	fprop(nn);
	
	float p_max = -1.0f;
	int res = -1; 

	int N = nn->num_layers;
	int out_layer_size = nn->layer_size[N - 1];
	for(int i = 0; i < out_layer_size; i++) {
		if(nn->layer[N - 1][i] > p_max) {
			res = i;
			p_max = nn->layer[N - 1][i];
		}
	}

	/* should never hapen */
	assert(res != -1);

	return res;
}


struct Neural_Network* nn_create(int num_layers, 
				const int *layer_size)
{

        struct Neural_Network *nn = malloc(sizeof(*nn));
	assert(nn != NULL);

	nn->af_hl = SIGMOID;
	nn->af_ol = SIGMOID;
	nn->lr_behav = FIXED;
	nn->learning_rate = 0.1f;
	nn->iterations = 0;

	make_layer_size(nn, num_layers, layer_size);
        make_layers(nn);
	make_weights(nn);
	make_bias(nn);

        return nn;
}


void nn_free(struct Neural_Network *nn)
{
	if(nn == NULL)
		return;
	
	int N = nn->num_layers;

	if(nn->layer != NULL) {
		for(int i = 1; i < N; i++) {
			free(nn->layer[i]);
		}
		free(nn->layer);
	}

	if(nn->bias != NULL) {	
		for(int i = 0; i < N - 1; i++) {
			free(nn->bias[i]);
		}
		free(nn->bias);
	}

	if(nn->weight != NULL) {
		int weight_size = nn->num_layers - 1;
		for(int i = 0; i < weight_size; i++) {
		
			int H = nn->layer_size[i + 1];
			if(nn->weight[i] == NULL)
				continue;

			for(int j = 0; j < H; j++) {
				free(nn->weight[i][j]);
			}
		
			free(nn->weight[i]);
		}
		free(nn->weight);
	}

	free(nn->layer_size);

	free(nn);
}


void nn_save_to_file(const struct Neural_Network *nn, const char *filename)
{
	FILE *file = fopen(filename, "wb");

	fwrite(&nn->num_layers, sizeof(int), 1, file);
	fwrite(nn->layer_size, sizeof(int), nn->num_layers, file);

	/* bias an weight size */
	int N = nn->num_layers - 1;

	for(int i = 0; i < N; i++) {
		fwrite(nn->bias[i], sizeof(float), nn->layer_size[i + 1], file);
	}


	for(int i = 0; i < N; i++) {
		int H = nn->layer_size[i + 1];
		int W = nn->layer_size[i];
		for(int j = 0; j < H; j++) {
			fwrite(nn->weight[i][j], sizeof(float), W, file);
		}
	}

	fclose(file);

}


struct Neural_Network *nn_load_from_file(const char *filename)
{
	FILE *file = fopen(filename, "rb");
	
	int num_layers;
	int res = fread(&num_layers, sizeof(int), 1, file);
	(void)res;

	int layer_size[num_layers];
	res = fread(layer_size, sizeof(int), num_layers, file);

	struct Neural_Network *nn = nn_create(num_layers, layer_size);
	assert(nn != NULL);
	
	
	/* bias and weight size  */
	int N = num_layers - 1;

	/* read bias */
	for(int i = 0; i < N; i++) {
		res = fread(nn->bias[i], sizeof(float), layer_size[i + 1], file);
	}

	/* read weights */
	for(int i = 0; i < N; i++) {
		int H = layer_size[i + 1];
		int W = layer_size[i];
		for(int j = 0; j < H; j++) {
			res = fread(nn->weight[i][j], sizeof(float), W, file);
		}
	}

	fclose(file);
	return nn;
}

void Training_Data_free(struct Training_Data *td)
{
	if(td->input_data == NULL)
		return;

	for(int i = 0; i < td->num_entries; i++) {
		free(td->input_data[i]);
	}
	free(td->input_data);

	if(td->expected_output == NULL)
		return;
	
	for(int i = 0; i < td->num_entries; i++) {
		free(td->expected_output[i]);
	}
	free(td->expected_output);
	
	free(td);
}


/*************** Activation Functions and its derivatives ****************/
static inline float sigmoid(float n)
{
	return 1.0f / (1.0f + expf(-n));
}

static inline float sigmoid_deriv(float out_val)
{
	return out_val * (1.0f - out_val);
}

static inline float act_tanh(float n)
{
	return tanhf(n);
}

static inline float act_tanh_deriv(float out_val)
{
	return 1.0f - (out_val * out_val);
}

static inline float relu(float n)
{
	return (n > 0.0f ? n : 0.0f);
}

static inline float relu_deriv(float out_val)
{
	return (out_val > 0.0f ? 1.0f : 0.0f);
}

static inline float swish(float n)
{
	return n * sigmoid(n);
}

static inline float swish_deriv(float out_val)
{
	/* https://arxiv.org/pdf/1801.07145.pdf */
	return out_val + sigmoid(out_val) * (1.0f - out_val);
}


static inline float act_func(enum Activation_Functions af, float n)
{
	switch(af) {
		case RELU:
			return relu(n);		
		case SIGMOID:
			return sigmoid(n);
		case TANH:
			return act_tanh(n);
		case SWISH:
			return swish(n);
	}

	return 0;
}

static inline float act_func_deriv(enum Activation_Functions af, float n)
{
	switch(af) {
		case RELU:
			return relu_deriv(n);		
		case SIGMOID:
			return sigmoid_deriv(n);
		case TANH:
			return act_tanh_deriv(n);
		case SWISH:
			return swish_deriv(n);
	}

	return 0;
}
/*********************************************************************/


/*************** Cost function and its derivatives *******************/

//static float cost_func(const struct Neural_Network *nn, 
//						const float *out_layer, 
//						const float *expected_value)
//{
//	int out_layer_size = nn->layer_size[nn->num_layers - 1];
//	float sum = 0.0f;
//	for(int i = 0; i < out_layer_size; i++) {
//		sum += (expected_value[i] - out_layer[i]) * 
//			(expected_value[i] - out_layer[i]);
//	}
//
//	return sum/2.0f;
//}

/*
* The partial derivative of the cost function
* with respect to the ouput neurons
*/
static inline float pcost_pout(float out_val, float expected_val)
{
	return (out_val - expected_val);	
}
/*********************************************************************/


/*************** Learning Rate Algos *********************************/
static inline void lr_schedule(struct Neural_Network *nn)
{
	nn->learning_rate -= nn->learning_rate / 2.0f;
	printf("%f\n", nn->learning_rate);
}

//static inline void lr_adaptive(float initial_lr, float avg_cost,
//						 float prev_avg_cost)
//{
//	
//}
static inline void adept_lr(struct Neural_Network *nn)
{
	switch(nn->lr_behav) {
		case FIXED:
			break;
		case SCHEDULE:
			lr_schedule(nn);
			break;
		case ADAPTIVE:
			break;
	}	
}
/*********************************************************************/


/******************* Network allocation functions *********************/
static void make_layer_size(struct Neural_Network *nn, 
			   int num_layers, 
                           const int *layer_size)
{
	nn->num_layers = num_layers;

        nn->layer_size = malloc(num_layers * sizeof(*nn->layer_size));
        assert(nn->layer_size != NULL);

        memcpy(nn->layer_size, layer_size, num_layers * sizeof(*nn->layer_size));
}

static void make_layers(struct Neural_Network *nn)
{
        nn->layer = malloc(nn->num_layers * sizeof(*nn->layer));
        assert(nn->layer != NULL);
        
        for(int i = 1; i < nn->num_layers; i++) {
                nn->layer[i] = malloc(nn->layer_size[i] * sizeof(**nn->layer));
                assert(nn->layer[i] != NULL);
        }
}


static inline float uniform_random_float() {
	return (2.0 * (float)rand() / (float)RAND_MAX) - 1.0;
}


static void randomize_weights(struct Neural_Network *nn) {
	
	srand(time(0));

	int weight_size = nn->num_layers - 1;
	for(int i = 0; i < weight_size; i++) {
		int H = nn->layer_size[i + 1];
		int W = nn->layer_size[i];
		for(int j = 0; j < H; j++) {
			for(int k = 0; k < W; k++) {
				nn->weight[i][j][k] = uniform_random_float();
			}
		}
	}
}

static void make_weights(struct Neural_Network *nn)
{
	
	/* A weight matrix connects the nodes between two 
	*  layers. Thats why there is one less weight matrix
	*  compared to the number of layers.
	*/
	int weight_size = nn->num_layers - 1;

	nn->weight = malloc(weight_size * sizeof(*nn->weight));
	assert(nn->weight != NULL);

	for(int i = 0; i < weight_size; i++) {
		// weight matrix height (rows)
		int H = nn->layer_size[i + 1];
		// weight matrix width (columns)
		int W = nn->layer_size[i];

		nn->weight[i] = malloc(H * sizeof(**nn->weight));
		assert(nn->weight[i] != NULL);

		for(int j = 0; j < H; j++) {
			nn->weight[i][j] = malloc(W * sizeof(***nn->weight));
			assert(nn->weight[i][j] != NULL);
		}
	}

	randomize_weights(nn);
}

static int make_bias(struct Neural_Network *nn)
{
	/* There is one bias for each node in each layer except
	*  for the input layer, which has no bias. That is the
	*  reason why there is one less bias vector compared to
	*  the number of total layers.
	*/
	int bias_size = nn->num_layers - 1;

        nn->bias = malloc(bias_size * sizeof(*nn->bias));
        assert(nn->bias != NULL);
        
        for(int i = 0; i < bias_size; i++) {
                nn->bias[i] = calloc(sizeof(**nn->bias), nn->layer_size[i+1]);
                assert(nn->bias[i] != NULL);
        }

        return 0;
}
/*********************************************************************/


/******************** forward and backwards propagation ****************/
static void fprop(struct Neural_Network *nn)
{	
	int N = nn->num_layers;
	for(int i = 1; i < N; i++) {

		#if NN_ENABLE_MULTI_PROCESSING == 1
		#pragma omp parallel for simd
		#endif
		for(int j = 0; j < nn->layer_size[i]; j++) {
			float sum = 0.0f;
			for(int k = 0; k < nn->layer_size[i-1]; k++) {
				sum += nn->layer[i-1][k] *
					nn->weight[i-1][j][k];
			}
			/*
			* The first layer has no bias. The bias vector
			* is one less then the total number of layers.
			* Thats why the bias[i-1] is the bias for layer[i].
			*/
			sum += nn->bias[i - 1][j];

			
			/* ouput layer (ol) might have a different activation
			*  function then the hidden layers (hl).
			*/
			if(i == (N - 1)) {				
				nn->layer[i][j] = act_func(nn->af_ol, sum);	
			} 
			else {
				nn->layer[i][j] = act_func(nn->af_hl, sum);
			}
		}
	}
}


static inline float error_out_layer(const struct Neural_Network *nn,
						    int node_index,
					  	    const struct Training_Data *td, 
						    int tdi)
{
	float node_value = nn->layer[nn->num_layers - 1][node_index];
	float expected_value = td->expected_output[tdi][node_index];

	return  act_func_deriv(nn->af_ol, node_value) * 
			pcost_pout(node_value, expected_value);
}

static void bprop_out_layer(const struct Neural_Network *nn,
						struct bp_data *bpd,
						const struct Training_Data *td,
						int tdi)
{
	int N = nn->num_layers;
	float *oe = bpd->error[N - 2];
	
	// weight size
	int ws = N - 1;
	// right layer size
	int rls = nn->layer_size[N - 1];
	// left layer size
	int lls = nn->layer_size[N - 2]; 
	#if NN_ENABLE_MULTI_PROCESSING == 1
	#pragma omp parallel for
	#endif
	for(int k = 0; k < rls; k++) {
		oe[k] = error_out_layer(nn, k, td, tdi);
		for(int j = 0; j < lls; j++) {
			bpd->d_weight[ws - 1][k][j] = oe[k] *
				nn->layer[N - 2][j];
		}
	}
}


static inline float error_hidden_layer(const struct Neural_Network *nn, 
				const float *error, int rli, int node_index)
{
	// rli = right layer index

	int N = nn->layer_size[rli + 1];

	float sum = 0.0f;
	for(int k = 0; k < N; k++) {
		sum += error[k] * nn->weight[rli][k][node_index];
	}

	float out_val = nn->layer[rli][node_index];
	return sum * act_func_deriv(nn->af_hl, out_val); 
	
}

static void bprop_hidden_layer(const struct Neural_Network *nn, int rli,
						      struct bp_data *bpd)
{
	// rli = right layer index

	// rls = right layer size
	int rls = nn->layer_size[rli];
	// lls = left layer size
	int lls = nn->layer_size[rli - 1];
	
	// he = hidden layer error
	float *he = bpd->error[rli - 1];

	#if NN_ENABLE_MULTI_PROCESSING == 1
	#pragma omp parallel for
	#endif
	for(int j = 0; j < rls; j++) {
		he[j] = error_hidden_layer(nn, bpd->error[rli], rli, j);	
		for(int i = 0; i < lls; i++) {
			bpd->d_weight[rli - 1][j][i] = he[j] * nn->layer[rli-1][i];	
		}
	}
}

static void bprop(const struct Neural_Network *nn, int tdi,
					const struct Training_Data *td,
					struct bp_data *bpd)
{
	
	int N = nn->num_layers;
	bprop_out_layer(nn, bpd, td, tdi);
	for(int i = N - 2; i > 0; i--) {
		bprop_hidden_layer(nn, i, bpd);
	}
}

static void update_weights(struct Neural_Network *nn, float ***d_weight)
{
	// weight size
	int ws = nn->num_layers - 1;
	
	for(int i = 0; i < ws; i++) {
		int H = nn->layer_size[i+1];
		int W = nn->layer_size[i]; 

		#if NN_ENABLE_MULTI_PROCESSING == 1
		#pragma omp parallel for
		#endif
		for(int j = 0; j < H; j++) {
			for(int k = 0; k < W; k++) {
				nn->weight[i][j][k] -= (d_weight[i][j][k] *
							nn->learning_rate);
			}
		}
	}
}

static void update_bias(struct Neural_Network *nn, float **error)
{
	/* N = bias size = error size */
	int N = nn->num_layers - 1;
	for(int i = 0; i < N; i++) {

		/* The first layer of the network has no bias.
		*  The bias and error start at the second layer.
		*  That is why the first array of bias values has
		*  the size of the second layer (M).
		*/
		int M = nn->layer_size[i + 1];

		#if NN_ENABLE_MULTI_PROCESSING == 1
		#pragma omp parallel for
		#endif
		for(int j = 0; j < M; j++) {
			nn->bias[i][j] -= error[i][j] * nn->learning_rate;
		}
	}
}


static void free_bpd(struct bp_data *bpd, const struct Neural_Network *nn)
{

	if(bpd == NULL)
		return;

	int N = nn->num_layers - 1;

	if(bpd->error != NULL) {
        	for(int i = 0; i < N; i++) {
			free(bpd->error[i]);
        	}
	}
	free(bpd->error);


	if(bpd->d_weight == NULL) 
		return;

	for(int i = 0; i < N; i++) {
		int H = nn->layer_size[i + 1];

		if(bpd->d_weight[i] == NULL)
			continue;

		for(int j = 0; j < H; j++) {
			free(bpd->d_weight[i][j]);
		}
		free(bpd->d_weight[i]);
	}
	free(bpd->d_weight);

	free(bpd);
}


static void bp_data_make_error(struct bp_data *bpd, const struct Neural_Network *nn)
{

	int error_size = nn->num_layers - 1;
	bpd->error = malloc(error_size * sizeof(*bpd->error));
        assert(bpd->error != NULL);
        
        for(int i = 0; i < error_size; i++) {
                bpd->error[i] = malloc(nn->layer_size[i + 1] * sizeof(**bpd->error));
                assert(bpd->error[i] != NULL);
        }
}

static void bp_data_make_d_weight(struct bp_data *bpd, const struct Neural_Network *nn) 
{
	int weight_size = nn->num_layers - 1;

	bpd->d_weight = malloc(weight_size * sizeof(*bpd->d_weight));
	assert(bpd->d_weight != NULL);

	for(int i = 0; i < weight_size; i++) {
		int H = nn->layer_size[i + 1];
		int W = nn->layer_size[i];

		bpd->d_weight[i] = malloc(H * sizeof(**bpd->d_weight));
		assert(bpd->d_weight[i] != NULL);

		for(int j = 0; j < H; j++) {
			bpd->d_weight[i][j] = malloc(W * sizeof(***bpd->d_weight));
			assert(bpd->d_weight[i][j] != NULL);
		}
	}
}

static struct bp_data *bp_data_create(const struct Neural_Network *nn)
{
	struct bp_data *bpd = malloc(sizeof(*bpd));
	assert(bpd != NULL);
	bp_data_make_error(bpd, nn);
	bp_data_make_d_weight(bpd, nn);

	return bpd;
}

