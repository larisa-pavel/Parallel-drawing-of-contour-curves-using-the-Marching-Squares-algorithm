// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.

int min(int a, int b) {
    if (a < b) {
        return a;
    }
    return b;
}

typedef struct my_args {
    int P_mare; //number of threads
    int thread_id;
    int step_x;
    int step_y;
    pthread_barrier_t *barrier;
    ppm_image *new_image;
    ppm_image *image;
    ppm_image **contour_map;
    unsigned char **grid;
} my_args;


ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Corresponds to step 1 of the marching squares algorithm, which focuses on sampling the image.
// Builds a p x q grid of points with values which can be either 0 or 1, depending on how the
// pixel values compare to the `sigma` reference value. The points are taken at equal distances
// in the original image, based on the `step_x` and `step_y` arguments.
void sample_grid(ppm_image *image, int step_x, int step_y, unsigned char sigma, int p, int q, unsigned char **grid, int thread_id, int P_mare) {
    int start = thread_id * (double)p / P_mare;
    int end = min((thread_id + 1) * (double)p / P_mare, p);
    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = image->data[i * step_x * image->y + j * step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > sigma) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }
    grid[p][q] = 0;

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = start; i < end; i++) {
        ppm_pixel curr_pixel = image->data[i * step_x * image->y + image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[i][q] = 0;
        } else {
            grid[i][q] = 1;
        }
    }
    start = thread_id * (double)q / P_mare;
    end = min((thread_id + 1) * (double)q / P_mare, p);
    for (int j = start; j < end; j++) {
        ppm_pixel curr_pixel = image->data[(image->x - 1) * image->y + j * step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[p][j] = 0;
        } else {
            grid[p][j] = 1;
        }
    }
}

// Corresponds to step 2 of the marching squares algorithm, which focuses on identifying the
// type of contour which corresponds to each subgrid. It determines the binary value of each
// sample fragment of the original image and replaces the pixels in the original image with
// the pixels of the corresponding contour image accordingly.
void march(ppm_image *image, unsigned char **grid, ppm_image **contour_map, int step_x, int step_y, int thread_id, int P_mare) {
    int p = image->x / step_x;
    int q = image->y / step_y;
    int start = thread_id * (double)p / P_mare;
    int end = min((thread_id + 1) * (double)p / P_mare, p);
    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(image, contour_map[k], i * step_x, j * step_y);
        }
    }
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

void rescale_image(ppm_image *image, ppm_image *new_image, int thread_id, int P_mare) {
    uint8_t sample[3];

    // we only rescale downwards
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
        *new_image = *image;
        return;
    }

    int N = new_image->x;
	int start = thread_id * (double)N / P_mare;
    int end = min((thread_id + 1) * (double)N / P_mare, N);

    // use bicubic interpolation for scaling
    for (int i = start; i < end; i++) {
        for (int j = 0; j < new_image->y; j++) {
            float u = (float)i / (float)(new_image->x - 1);
            float v = (float)j / (float)(new_image->y - 1);
            sample_bicubic(image, u, v, sample);

            new_image->data[i * new_image->y + j].red = sample[0];
            new_image->data[i * new_image->y + j].green = sample[1];
            new_image->data[i * new_image->y + j].blue = sample[2];
        }
    }
    return;
}

void *my_function(void *arg) {
    my_args *my_thread = (my_args *)arg;
    rescale_image(my_thread->image, my_thread->new_image, my_thread->thread_id, my_thread->P_mare);
    pthread_barrier_wait(my_thread->barrier);
    int p = my_thread->new_image->x / my_thread->step_x;
    int q = my_thread->new_image->y / my_thread->step_y;
    sample_grid(my_thread->new_image, my_thread->step_x, my_thread->step_y, SIGMA, p, q, my_thread->grid, my_thread->thread_id, my_thread->P_mare);
    pthread_barrier_wait(my_thread->barrier);
    march(my_thread->new_image, my_thread->grid, my_thread->contour_map, my_thread->step_x, my_thread->step_y, my_thread->thread_id, my_thread->P_mare);
    pthread_barrier_wait(my_thread->barrier);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }
    //moved all the allocations in the main function for more efficiency
    ppm_image *image = read_ppm(argv[1]);
    int step_x = STEP;
    int step_y = STEP;

    ppm_image **contour_map = init_contour_map();

    ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
    if (!new_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    new_image->x = RESCALE_X;
    new_image->y = RESCALE_Y;
    int p = new_image->x / step_x;
    int q = new_image->y / step_y;

    new_image->data = (ppm_pixel*)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));
    if (!new_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char*));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++) {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }
    //number of threads
    int P = atoi(argv[3]);
    //vector of P threads
    my_args my_thread[P];
    pthread_t threads[P];
    pthread_barrier_t my_barrier;
    pthread_barrier_init(&my_barrier, NULL, P);
    for (int i = 0; i < P; i++) {
        my_thread[i].thread_id = i;
        my_thread[i].image = image;
        my_thread[i].step_x = step_x;
        my_thread[i].step_y = step_y;
        my_thread[i].contour_map = contour_map;
        my_thread[i].barrier = &my_barrier;
        my_thread[i].grid = grid;
        my_thread[i].new_image = new_image;
        my_thread[i].P_mare = P;
        pthread_create(&threads[i], NULL, my_function, &my_thread[i]);
    }

    for (int i = 0; i < P; i++) {
		pthread_join(threads[i], NULL);
	}
    write_ppm(new_image, argv[2]);

    if (image == new_image) {
        free(image->data);
        free(image);
    }

    free_resources(new_image, contour_map, grid, step_x);
    
    pthread_barrier_destroy(&my_barrier);
    return 0;
}
