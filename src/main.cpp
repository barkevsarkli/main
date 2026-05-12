#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "data.h"
#include "net.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 12
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 10

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                << " <func1> <func2> <seed>" << std::endl;
        return 1;
    }

    std::srand(std::stoi(argv[3]));

    std::string activation_type1 = argv[1];
    std::string activation_type2 = argv[2];

    float learning_rate = LEARNING_RATE;

    std::cout << "main() | Using activation functions: " << activation_type1 << " and " << activation_type2 << std::endl;
    std::cout << "main() | Hidden layer size: " << HIDDEN_SIZE << ", Learning rate: " << learning_rate << ", Epochs: " << EPOCHS << std::endl;

    Data train_images_data("./data/train-images/train-images-idx3-ubyte",
                           "./data/train-images/train-labels-idx1-ubyte", 0.2);

    Neuron hidden_layer1(INPUT_SIZE, HIDDEN_SIZE, activation_type1, activation_type2);
    Neuron hidden_layer2(HIDDEN_SIZE, HIDDEN_SIZE, activation_type1, activation_type2);
    Neuron output_layer(HIDDEN_SIZE, OUTPUT_SIZE, activation_type1, activation_type2);

    std::cout << "main() | Using the same initials for all runs." << std::endl;

    for (unsigned short epoch = 0; epoch < EPOCHS; epoch++)
    {
        float total_loss = 0;

        for (unsigned short i = 0; i < train_images_data.get_train_size(); i++)
        {
            Image current_image = train_images_data.get_train_images()[i];

            hidden_layer1.forward(current_image.normalized_pixels);
            hidden_layer2.forward(hidden_layer1.get_outputs());
            output_layer.forward(hidden_layer2.get_outputs());

            const float *predictions = output_layer.get_outputs();
            float loss = 0;
            std::vector<float> d_loss(OUTPUT_SIZE);

            for (unsigned short j = 0; j < OUTPUT_SIZE; j++)
            {
                float error = predictions[j] - current_image.one_hot_encoded_label[j];
                loss += error * error;
                d_loss[j] = 2 * error;
            }

            total_loss += loss / OUTPUT_SIZE;

            output_layer.backward(d_loss.data());
            hidden_layer2.backward(output_layer.get_d_inputs());
            hidden_layer1.backward(hidden_layer2.get_d_inputs());

            hidden_layer1.update(learning_rate);
            hidden_layer2.update(learning_rate);
            output_layer.update(learning_rate);
        }

        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                  << ", Loss: " << total_loss / train_images_data.get_train_size() << std::endl;
    }

    return 0;
}
