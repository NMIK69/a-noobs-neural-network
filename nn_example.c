#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mnist.h"
#include "neural_network.h"

int main() {

	/* --------- Creating a new network ----------*/

	/* Determine the number of nodes in each layer */
        int layer_size[] = {784, 64, 10};
	/* Determine the number of total layers */
	int num_layers = sizeof(layer_size) / sizeof(*layer_size);

	/* Create the network */
        struct Neural_Network *nn = nn_create(num_layers, layer_size);


	/* --------- Loading the trainig and validation data ----------*/
	
	/* How many entries shall be read from the data file */
	/* A value of -1 implies to read all present entries. */
	int tdata_entries = -1;
	/* load all the training data entires from image and label file.*/
	struct Training_Data *tdata = load_mnist_from_file("data/train-images.idx3-ubyte", 
							"data/train-labels.idx1-ubyte",
							tdata_entries);
	int vdata_entries = -1;
	/* Load all the vaidation data entries from image and label file.*/
	struct Training_Data *vdata = load_mnist_from_file("data/t10k-images.idx3-ubyte",
				       			"data/t10k-labels.idx1-ubyte", 
							vdata_entries);

	/* --------- Training the network ----------*/

	/* Set activation function for the output layer */
	/* Default value = SIGMOID */
	nn->af_ol = SIGMOID;

	/* Set activation function for all hidden layers */
	/* Default value = SIGMOID */
	nn->af_hl = SIGMOID;

	/* Set learning rate behaviour alogrithm (experimental) */
	/* Default value = FIXED */
	nn->lr_behav = FIXED;

	/* Set number of iterations */
	/* Default value = 0 */
	nn->iterations = 5;

	/* Set learning rate */
	/* Default value = 0.1 */
	nn->learning_rate = 0.1;

	printf("Training on %d datapoints\n", tdata->num_entries);
	printf("Training for %d iterations with a learning rate of %f\n", nn->iterations, nn->learning_rate);

	/* Train the network on the training data (tdata). */
	nn_train(nn, tdata);


	printf("Training done...\n");
	printf("Determining performance on %d datapoints...\n", vdata->num_entries);


	/* --------- Evaluate performance on the validation dataset ----------*/

	/* count the number or correct prediction (cp_count) on the validation
	 * data (vdata)*/
	int cp_count = 0;
	for(int i = 0; i < vdata->num_entries; i++) {
		int prediction = nn_predict(nn, vdata->input_data[i]);

		assert(prediction >= 0);

		if(vdata->expected_output[i][prediction] == 1.0f) {
			cp_count += 1;
		}
		
	}

	/* evaluate performance */
	float perf = ((float)cp_count/(float)vdata->num_entries) * 100.0f;
	printf("Successfull prediction rate: %.3f\n", perf);


	/* --------- Save network to file ----------*/

	nn_save_to_file(nn, "nn1.bin");


	/* --------- Release memory ----------*/

	/* free Training_Data memory */
	Training_Data_free(tdata);
	Training_Data_free(vdata);

	/* free Neural_Network memory */
	nn_free(nn);

        return 0;
}
