#include <cuda.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>

#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <ctime>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


const int VECTOR_SIZE = 3;
const int TIME = 2000;
const int MAP_SIZE = 1000;


class Neuron
{
public:
	int x;
	int y;
	double vector[VECTOR_SIZE];
	Neuron() : x(0), y(0) { 
		vector[0] = 0;
		vector[1] = 0;
		vector[2] = 0;
	}
	Neuron(int x, int y) : x(x), y(y) {
		for (int i = 0; i < VECTOR_SIZE; ++i)
			vector[i] = ((double)rand() / (RAND_MAX));
	}
	void init_by_coords(int x, int y) {
		this->x = x;
		this->y = y;
		for (int i = 0; i < VECTOR_SIZE; ++i)
			vector[i] = ((double)rand() / (RAND_MAX));
	}
	Neuron(double r, double g, double b) {
		vector[0] = r;
		vector[1] = g;
		vector[2] = b;
	}
};

double weight_metric(Neuron& neuron1, Neuron& neuron2) {
	return sqrt((neuron1.vector[0] - neuron2.vector[0])*(neuron1.vector[0] - neuron2.vector[0]) +
		(neuron1.vector[1] - neuron2.vector[1])*(neuron1.vector[1] - neuron2.vector[1]) +
		(neuron1.vector[2] - neuron2.vector[2])*(neuron1.vector[2] - neuron2.vector[2]));
}

class Map
{
public:
	int size;
	Neuron* matrix;
	Map(int size) : size(size) {
		this->matrix = new Neuron[size*size];
		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < size; ++j)
				this->matrix[i*size + j].init_by_coords(i, j);
		}
	}
	Neuron& get_most_simular(Neuron& neuron) {
		Neuron* simular_neuron = new Neuron(0, 0);
		double min_metric = 4;
#pragma omp parallel 
		{
		Neuron* private_simular_neuron = new Neuron(0, 0);
		double private_min_metric = 400;
		double m = 0;
#pragma omp for nowait
		for (int i = 0; i < this->size; ++i) {
			for (int j = 0; j < this->size; ++j) {
				m = weight_metric(this->matrix[i*size + j], neuron);
				if (m < private_min_metric) {
					private_min_metric = m;
					private_simular_neuron = &this->matrix[i*size + j];
				}
			}
		}
#pragma omp critical 
		if (private_min_metric < min_metric) {
			min_metric = private_min_metric;
			simular_neuron = private_simular_neuron;
		}
		}
		return *simular_neuron;
	}

	void print_x_coords() {
		for (int i = 0; i < this->size; ++i)
			for (int j = 0; j < this->size; ++j)
				printf("%d ", this->matrix[i*size + j].x);
		printf("\n");
	}
	void print_y_coords() {
		for (int i = 0; i < this->size; ++i)
			for (int j = 0; j < this->size; ++j)
				printf("%d ", this->matrix[i*size + j].y);
		printf("\n");
	}
	void print_rgbs() {
		for (int i = 0; i < this->size; ++i)
			for (int j = 0; j < this->size; ++j)
				printf("%f %f %f|", this->matrix[i*size + j].vector[0],
					this->matrix[i*size + j].vector[1],
					this->matrix[i*size + j].vector[2]);
		printf("\n");
	}
};

__global__ void process_neuron(Neuron* map, Neuron* neuron, int t, double eta, double sigma) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	double ro = (float)((map[x].x - neuron->x)*(map[x].x - neuron->x) +
		                     (map[x].y - neuron->y)*(map[x].y - neuron->y));
	if (ro >= sigma * sigma)
		return;

	double h = exp(-(ro) / (2 * sigma * sigma));

	for (int i = 0; i < VECTOR_SIZE; ++i)		
		map[x].vector[i] += eta * h * (neuron->vector[i] - map[x].vector[i]);
}

int main()
{
	srand(time(0));

	Map* map = new Map(MAP_SIZE);

	std::vector<Neuron> learning;
	learning.push_back(Neuron(1, 0, 0));
	learning.push_back(Neuron(0, 1, 0));
	learning.push_back(Neuron(0, 0, 1));
	learning.push_back(Neuron(1, 1, 0));
	learning.push_back(Neuron(1, 0, 1));
	learning.push_back(Neuron(0, 1, 1));
	learning.push_back(Neuron(1, 0.5, 0));

	Neuron* neuron = new Neuron();
	int t = 1;

	Map* dev_map = new Map(MAP_SIZE);
	Neuron* dev_neuron;

	gpuErrchk(cudaMalloc((void**)&dev_map->matrix, MAP_SIZE * sizeof(Neuron)*MAP_SIZE));
	gpuErrchk(cudaMalloc((void**)&dev_neuron, sizeof(Neuron)));

	//копируем ввод на device
	gpuErrchk(cudaMemcpy(dev_map->matrix, map->matrix, MAP_SIZE*MAP_SIZE * sizeof(Neuron), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_neuron, neuron, sizeof(Neuron), cudaMemcpyHostToDevice));

	double eta, sigma, lambda;

	double start_time = omp_get_wtime();
	while (t < TIME) {
		if (t != 1) {
			gpuErrchk(cudaMemcpy(map->matrix, dev_map->matrix, MAP_SIZE*MAP_SIZE * sizeof(Neuron), cudaMemcpyDeviceToHost));
		}
		neuron = &map->get_most_simular(learning[rand() % 7]);
		gpuErrchk(cudaMemcpy(dev_neuron, neuron, sizeof(Neuron), cudaMemcpyHostToDevice));
		lambda = ((double)TIME) / log(((double)MAP_SIZE) / 2);
		eta = 0.1 * exp(-(double)t / lambda);
		sigma = (((double)MAP_SIZE) / 2) * exp(-(double)t / lambda);
		process_neuron << <MAP_SIZE, MAP_SIZE >> > (dev_map->matrix, dev_neuron, t, eta, sigma);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		++t;
	}
	printf("%f\n", omp_get_wtime() - start_time);

	cudaMemcpy(map->matrix, dev_map->matrix, MAP_SIZE*MAP_SIZE * sizeof(Neuron), cudaMemcpyDeviceToHost);

	map->print_x_coords();
	map->print_y_coords();
	map->print_rgbs();

	cudaFree(dev_map->matrix);
	cudaFree(dev_map);
	cudaFree(dev_neuron);
}
