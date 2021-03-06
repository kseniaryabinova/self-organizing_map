// SOM.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <cstdlib>
#include <math.h>
#include <vector>
#include <ctime>
#include <omp.h>

const int VECTOR_SIZE = 3;
const int TIME = 2000;
const int MAP_SIZE = 100;


class Neuron
{
public:
	int x;
	int y;
	double vector[VECTOR_SIZE];
	Neuron(): x(0), y(0) { }
	Neuron(int x, int y): x(x), y(y) {
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

void process_neuron(Neuron* neuron1, Neuron* neuron2, int t) {
	double lambda = ((double)TIME) / log(((double)MAP_SIZE) / 2);
	double sigma = (((double)MAP_SIZE) / 2) * exp(-(double)t / lambda);
	double ro = (neuron1->x - neuron2->x)*(neuron1->x - neuron2->x) +
				(neuron1->y - neuron2->y)*(neuron1->y - neuron2->y);
	if (ro >= sigma * sigma)
		return;
	double eta = 0.1 * exp(-(double)t / lambda);
	double h = exp(-(ro) / (2 * sigma * sigma));
	for (int i = 0; i < VECTOR_SIZE; ++i)
		neuron1->vector[i] += eta * h * (neuron2->vector[i] - neuron1->vector[i]);
}

class Map
{
public:
	int size;
	Neuron** matrix;
	Map(int size) : size(size) {
		this->matrix = new Neuron*[size];
		for (int i = 0; i<size; ++i){
			this->matrix[i] = new Neuron[size];
			for (int j = 0; j < size; ++j)
				this->matrix[i][j].init_by_coords(i, j);
		}
	}
	void process(Neuron* neuron, int t) {
		int a = 0;
#pragma omp parallel for 
		for (int i = 0; i < MAP_SIZE; ++i)
			for (int j = 0; j < this->size; ++j)
				process_neuron(&this->matrix[i][j], neuron, t);	
	}
	Neuron& get_most_simular(Neuron& neuron) {
		Neuron* simular_neuron = new Neuron(0,0);
		double min_metric = this->size*this->size*this->size;
		for (int i= 0; i<this->size; ++i)
			for (int j = 0; j < this->size; ++j) {
				double m = weight_metric(this->matrix[i][j], neuron);
				if (m < min_metric) {
					min_metric = m;
					simular_neuron = &this->matrix[i][j];
				}
			}
		return *simular_neuron;
	}
	void print_x_coords() {
		for (int i = 0; i < this->size; ++i)
			for (int j = 0; j < this->size; ++j)
				printf("%d ", this->matrix[i][j].x);
		printf("\n");
	}
	void print_y_coords() {
		for (int i = 0; i < this->size; ++i)
			for (int j = 0; j < this->size; ++j)
				printf("%d ", this->matrix[i][j].y);
		printf("\n");
	}
	void print_rgbs() {
		for (int i = 0; i < this->size; ++i)
			for (int j = 0; j < this->size; ++j)
				printf("%f %f %f|", this->matrix[i][j].vector[0], 
					                this->matrix[i][j].vector[1], 
					                this->matrix[i][j].vector[2]);
		printf("\n");
	}
};


int main()
{
	srand(time(0));

	Map map(MAP_SIZE);

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

	double start_time = omp_get_wtime();
	while (t < TIME) {
		neuron = &map.get_most_simular(learning[rand() % 7]);
		map.process(neuron, t);
		++t;		
	}
	printf("%f\n", omp_get_wtime() - start_time);

	map.print_x_coords();
	map.print_y_coords();
	map.print_rgbs();
}

