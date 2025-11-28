#include "net.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <string>

void Neuron::initialize_parameters_he(float *array, unsigned short in, unsigned short out) 
{
    float std_dev = sqrt(2.0f / in);
    
    for (unsigned int i = 0; i < (unsigned int)in * out; i++) 
    {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float rand_std_normal = sqrt(-2.0f * log(u1)) * sin(2.0f * acos(-1.0f) * u2);
        array[i] = rand_std_normal * std_dev;
    }
}

Neuron::Neuron(unsigned short number_of_inputs, unsigned short number_of_neurons) 
{
    counter = 0;
    this->number_of_inputs = number_of_inputs;
    this->number_of_neurons = number_of_neurons;

    weights = new float[(unsigned int)number_of_inputs * number_of_neurons];
    biases = new float[number_of_neurons];
    outputs = new float[number_of_neurons];
    weight_gradients = new float[(unsigned int)number_of_inputs * number_of_neurons];
    bias_gradients = new float[number_of_neurons];
    d_inputs = new float[number_of_inputs];

    initialize_parameters_he(weights, number_of_inputs, number_of_neurons);

    for (unsigned short i = 0; i < number_of_neurons; ++i) 
    {
        biases[i] = 0.0f;
        bias_gradients[i] = 0.0f;
    }

     for (unsigned int i = 0; i < (unsigned int)number_of_inputs * number_of_neurons; i++) 
        weight_gradients[i] = 0.0f;
}

Neuron::~Neuron(void) 
{
    delete[] weights;
    delete[] biases;
    delete[] outputs;
    delete[] weight_gradients;
    delete[] bias_gradients;
    delete[] d_inputs;
}

void Neuron::forward(const float *inputs) 
{
    counter++;
    this->inputs = inputs;

    for (unsigned short i = 0; i < number_of_neurons; i++) 
    {
        outputs[i] = biases[i];
        for (unsigned short j = 0; j < number_of_inputs; ++j)
            outputs[i] += inputs[j] * weights[i * number_of_inputs + j];

        if (outputs[i] < 0)
            outputs[i] = 0;
        
    }

    if (counter >= 1000)
        counter = 0;
}

void Neuron::backward(const float *d_values) 
{
    for (unsigned short i = 0; i < number_of_inputs; i++)
        d_inputs[i] = 0.0f;

    for (unsigned short i = 0; i < number_of_neurons; i++) 
    {
        if (outputs[i] > 0) 
        {
            bias_gradients[i] += d_values[i];

            for (unsigned short j = 0; j < number_of_inputs; j++) 
            {
                weight_gradients[i * number_of_inputs + j] += inputs[j] * d_values[i];
                d_inputs[j] += weights[i * number_of_inputs + j] * d_values[i];
            }
        }
    }
}

void Neuron::update(float learning_rate) 
{
    for (unsigned int i = 0; i < (unsigned int) number_of_inputs * number_of_neurons; i++)
        weights[i] -= learning_rate * weight_gradients[i];

    for (unsigned short i = 0; i < number_of_neurons; i++)
        biases[i] -= learning_rate * bias_gradients[i];
    
    for (unsigned int i = 0; i < (unsigned int)number_of_inputs * number_of_neurons; i++)
        weight_gradients[i] = 0.0f;

    for (unsigned short i = 0; i < number_of_neurons; i++)
        bias_gradients[i] = 0.0f;
}
