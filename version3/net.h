#ifndef NET_H
#define NET_H

#include <string>

class Neuron 
{
    private:
        short counter;
        float *weights;
        float *biases;
        float *outputs;
        float *weight_gradients;
        float *bias_gradients;
        float *d_inputs;

        const float *inputs;
        unsigned short number_of_inputs;
        unsigned short number_of_neurons;
        
        std::string activation_function_type1;
        std::string activation_function_type2;

        void initialize_parameters_random(float *array, unsigned short in, unsigned short out);

    public:
        Neuron(unsigned short number_of_inputs, unsigned short number_of_neurons, 
               const std::string& act_func1 = "relu", const std::string& act_func2 = "relu");
        ~Neuron(void);

        void forward(const float *inputs);
        void backward(const float *d_values);

        void update(float learning_rate);

        void apply_convolution(void);


        const float *get_weights() { return weights; }
        const float* get_outputs() const { return outputs; }
        const float* get_d_inputs() const { return d_inputs; }
};

#endif 