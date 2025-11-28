#ifndef NET_H
#define NET_H

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

        void initialize_parameters_he(float *array, unsigned short in, unsigned short out);

    public:
        Neuron(unsigned short number_of_inputs, unsigned short number_of_neurons);
        ~Neuron(void);

        void forward(const float *inputs);
        void backward(const float *d_values);

        void update(float learning_rate);

        void apply_convolution(void);

        const float* get_outputs() const { return outputs; }
        const float* get_d_inputs() const { return d_inputs; }
};

#endif 