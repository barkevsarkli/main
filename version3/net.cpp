#include "net.h"
#include "data.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <string>

void Neuron::initialize_parameters_random(float *array, unsigned short in, unsigned short out) 
{
    float limit = sqrt(1.0f / in);
    
    for (unsigned int i = 0; i < (unsigned int)in * out; i++) 
    {
        array[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
}

Neuron::Neuron(unsigned short number_of_inputs, unsigned short number_of_neurons,
               const std::string& act_func1, const std::string& act_func2) 
{
    counter = 0;
    this->number_of_inputs = number_of_inputs;
    this->number_of_neurons = number_of_neurons;
    this->activation_function_type1 = act_func1;
    this->activation_function_type2 = act_func2;
    
    std::cout << "Neuron layer created: " << number_of_inputs << " -> " << number_of_neurons 
              << " (Even neurons use: " << act_func1 << ", Odd neurons use: " << act_func2 << ")" << std::endl;

    weights = new float[(unsigned int)number_of_inputs * number_of_neurons];
    biases = new float[number_of_neurons];
    outputs = new float[number_of_neurons];
    weight_gradients = new float[(unsigned int)number_of_inputs * number_of_neurons];
    bias_gradients = new float[number_of_neurons];
    d_inputs = new float[number_of_inputs];

    std::cout << "Using Kaiming Uniform initialization (PyTorch default)" << std::endl;
    initialize_parameters_random(weights, number_of_inputs, number_of_neurons);

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

        if (outputs[i] > 20.0f) outputs[i] = 20.0f;
        if (outputs[i] < -20.0f) outputs[i] = -20.0f;

        if ((i % 2) == 0)
        {
            if (activation_function_type1 == "relu") {
                if (outputs[i] < 0)
                    outputs[i] = 0;
            }

            else if (activation_function_type1 == "leaky_relu")
            {
                if (outputs[i] < 0)
                    outputs[i] *= 0.1;
            }

            else if (activation_function_type1 == "tanh") {
                outputs[i] = tanh(outputs[i]);
            }
            else if (activation_function_type1 == "sigmoid")
            {
                if (outputs[i] >= 0) {
                    float z = exp(-outputs[i]);
                    outputs[i] = 1.0f / (1.0f + z);
                } else {
                    float z = exp(outputs[i]);
                    outputs[i] = z / (1.0f + z);
                }
            }
            
            else 
            {
                if (outputs[i] < 0)
                    outputs[i] = 0;
            }
        }

        else
        {
            if (activation_function_type2 == "relu")
            {
                if (outputs[i] < 0)
                    outputs[i] = 0;
            }

            else if (activation_function_type2 == "leaky_relu")
            {
                if (outputs[i] < 0)
                    outputs[i] *= 0.01;
            }

            else if (activation_function_type2 == "tanh")
            {
                outputs[i] = tanh(outputs[i]);
            }

            else if (activation_function_type2 == "sigmoid")
            {
                if (outputs[i] >= 0) 
                {
                    float z = exp(-outputs[i]);
                    outputs[i] = 1.0f / (1.0f + z);
                }
                
                else 
                {
                    float z = exp(outputs[i]);
                    outputs[i] = z / (1.0f + z);
                }
            }

            else 
            {
                if (outputs[i] < 0)
                    outputs[i] = 0;
            }
        }
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
        float activation_derivative;
        
        if ((i % 2) == 0) 
        {
            if (activation_function_type1 == "relu")
                activation_derivative = (outputs[i] > 0) ? 1.0f : 0.0f;

            else if (activation_function_type1 == "leaky_relu")
                activation_derivative = (outputs[i] > 0) ? 1.0f : 0.1;

            else if (activation_function_type1 == "tanh")
                activation_derivative = 1.0f - outputs[i] * outputs[i];

            else if (activation_function_type1 == "sigmoid")
                activation_derivative = outputs[i] * (1.0f - outputs[i]);

            else
                activation_derivative = (outputs[i] > 0) ? 1.0f : 0.0f;
        }
        else
        {
            if (activation_function_type2 == "relu") {
                activation_derivative = (outputs[i] > 0) ? 1.0f : 0.0f;
            }

            else if (activation_function_type2 == "leaky_relu")
                activation_derivative = (outputs[i] > 0) ? 1.0f : 0.1;

            else if (activation_function_type2 == "tanh") {
                activation_derivative = 1.0f - outputs[i] * outputs[i];
            }
            else if (activation_function_type2 == "sigmoid") {
                activation_derivative = outputs[i] * (1.0f - outputs[i]);
            }
            else {
                activation_derivative = (outputs[i] > 0) ? 1.0f : 0.0f;
            }
        }
        
        float d_output = d_values[i] * activation_derivative;
        
        if (d_output > 10.0f) d_output = 10.0f;
        if (d_output < -10.0f) d_output = -10.0f;
        
        bias_gradients[i] += d_output;

        for (unsigned short j = 0; j < number_of_inputs; j++) 
        {
            weight_gradients[i * number_of_inputs + j] += inputs[j] * d_output;
            d_inputs[j] += weights[i * number_of_inputs + j] * d_output;
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
