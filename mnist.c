#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mnist.h"

static int fetch_images(FILE *file, int read_limit);
static int fetch_labels(FILE *file, int read_limit);

static void load_images(FILE *file, struct Training_Data *td);
static void load_labels(FILE *file, struct Training_Data *td);

static void allocate_img_data(struct Training_Data *td);
static void allocate_lbl_data(struct Training_Data *td);

static long int get_filesize(FILE *file);


struct Training_Data *load_mnist_from_file(const char *img_file, 
					const char *lbl_file, 
					int max_datasets)
{
	struct Training_Data *td = malloc(sizeof(*td));
	assert(td != NULL);

	FILE *img_fp = fopen(img_file, "rb");	
	assert(img_fp != NULL);

	FILE *lbl_fp = fopen(lbl_file, "rb");	
	assert(lbl_fp != NULL);

	int num_images = fetch_images(img_fp, max_datasets);
	int num_labels = fetch_labels(lbl_fp, max_datasets);
	
	assert(num_images == num_labels);
	td->num_entries = num_images;
	td->input_size = MNIST_NUM_PIXEL;
	td->output_size = MNIST_NUM_OUTCOMES;

	allocate_img_data(td);
	allocate_lbl_data(td);

	load_images(img_fp, td);
	load_labels(lbl_fp, td);

	fclose(img_fp);
	fclose(lbl_fp);

	return td;
}

static void allocate_img_data(struct Training_Data *td)
{
	td->input_data = malloc(td->num_entries * sizeof(*td->input_data));
	assert(td->input_data != NULL);

	for(int i = 0; i < td->num_entries; i++) {
		td->input_data[i] = malloc(MNIST_NUM_PIXEL * sizeof(**td->input_data));	
		assert(td->input_data != NULL);
	}
}


static void allocate_lbl_data(struct Training_Data *td)
{
	td->expected_output = malloc(td->num_entries * sizeof(*td->expected_output));
	assert(td->expected_output != NULL);

	for(int i = 0; i < td->num_entries; i++) {
		td->expected_output[i] = calloc(sizeof(**td->expected_output), MNIST_NUM_OUTCOMES);	
		assert(td->expected_output[i] != NULL);
	}
}

static long int get_filesize(FILE *file)
{
	int res = fseek(file, 0, SEEK_END);
	assert(res == 0);

	long int filesize = ftell(file);
	assert(filesize != -1L);
	
	return filesize;
}

static void load_images(FILE *file, struct Training_Data *td)
{
	/* set fp to right after file header */
	
	int res = fseek(file, 16, SEEK_SET);
	assert(res == 0);

	for(int i = 0; i < td->num_entries; i++) {
		for(int j = 0; j < MNIST_NUM_PIXEL; j++) {
			int value = fgetc(file);
			assert(!feof(file));
			assert(!ferror(file));

			td->input_data[i][j] = (float)value / 255.0f;
		}
	}
}


static void load_labels(FILE *file, struct Training_Data *td)
{
	/* set fp to right after file header */
	int res = fseek(file, 8, SEEK_SET);
	assert(res == 0);

	for(int i = 0; i < td->num_entries; i++) {
		int value = fgetc(file);
		assert(!feof(file));
		assert(!ferror(file));
		assert(value < 10 && value >= 0);

		td->expected_output[i][value] = 1.0f;
	}
}

static int fetch_images(FILE *file, int read_limit)
{
	long int filesize = get_filesize(file);
	assert(filesize > 0);

	int num_images = (filesize - 16) / MNIST_NUM_PIXEL;	
	if(read_limit != -1 && read_limit < num_images)
		num_images = read_limit;
	
	return num_images;
}

static int fetch_labels(FILE *file, int read_limit)
{
	long int filesize = get_filesize(file);
	assert(filesize > 0);
	
	int num_labels = filesize - 8;
	if(read_limit != -1 && read_limit < num_labels)
		num_labels = read_limit;
	
	return num_labels;
}

