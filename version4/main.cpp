#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "cifar_data.h"
#include "net.h"

#define INPUT_SIZE 3072
#define HIDDEN_SIZE 258   
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001f

int main(int argc, char** argv)
{
    if (argc < 5) 
    {
        std::cerr << "Usage: " << argv[0]
                << " <func1> <func2> <seed> <epochs>" << std::endl;
        return 1;
    }

    std::srand(std::stoi(argv[3]));
    std::string activation_type1 = argv[1];
    std::string activation_type2 = argv[2];
    unsigned short epochs = std::stoi(argv[4]);

    float learning_rate = LEARNING_RATE;
    if (activation_type1 == "sigmoid" || activation_type2 == "sigmoid") 
        learning_rate = 0.01f; 

    std::string filename = "cifar_results_" + activation_type1 + "_" + activation_type2 + ".csv";
    std::ofstream out_file(filename, std::ios::app);

    std::cout << "Initializing CIFAR-10 Data Loader..." << std::endl;
    
    CIFAR_DATA data("/Users/khaos/Desktop/main-main/CIFAR10/train_images/", 
                    "/Users/khaos/Desktop/main-main/CIFAR10/test_images/test_batch.bin");

    data.print_stats();

    Neuron hidden_layer1(INPUT_SIZE, HIDDEN_SIZE, activation_type1, activation_type2);
    Neuron hidden_layer2(HIDDEN_SIZE, HIDDEN_SIZE, activation_type1, activation_type2);
    Neuron hidden_layer3(HIDDEN_SIZE, HIDDEN_SIZE, activation_type1, activation_type2);
    Neuron output_layer(HIDDEN_SIZE, OUTPUT_SIZE, activation_type1, activation_type2);

    std::cout << "Training started (" << epochs << " epochs)" << std::endl;

    for (unsigned short epoch = 0; epoch < epochs; epoch++) 
    {
        float total_loss = 0;
        unsigned int correct_train_predictions = 0;
        unsigned int train_size = data.get_train_size();

        for (unsigned int i = 0; i < train_size; i++) 
        {
            float *current_input = &data.train_images_flat[i * INPUT_SIZE];
            float *current_target = &data.train_labels_encoded[i * OUTPUT_SIZE];
            int true_label = (int)data.train_labels[i];

            hidden_layer1.forward(current_input);
            hidden_layer2.forward(hidden_layer1.get_outputs());
            hidden_layer3.forward(hidden_layer2.get_outputs());
            output_layer.forward(hidden_layer3.get_outputs());

            const float *predictions = output_layer.get_outputs();
            float loss = 0;
            std::vector<float> d_loss(OUTPUT_SIZE);

            int predicted_label = 0;
            float max_output = -1.0f;

            for (unsigned short j = 0; j < OUTPUT_SIZE; j++) 
            {
                float error = predictions[j] - current_target[j];
                
                loss += error * error;
                d_loss[j] = 2 * error;

                if (predictions[j] > max_output) 
                {
                    max_output = predictions[j];
                    predicted_label = j;
                }
            }

            if (predicted_label == true_label)
                correct_train_predictions++;

            total_loss += loss / OUTPUT_SIZE;

            output_layer.backward(d_loss.data());
            hidden_layer3.backward(output_layer.get_d_inputs());
            hidden_layer2.backward(hidden_layer3.get_d_inputs());
            hidden_layer1.backward(hidden_layer2.get_d_inputs());

            hidden_layer1.update(learning_rate);
            hidden_layer2.update(learning_rate);
            hidden_layer3.update(learning_rate);
            output_layer.update(learning_rate);
        }

        float train_accuracy = (float)correct_train_predictions / train_size * 100.0f;
        float average_loss = total_loss / train_size;
        
        printf("Epoch %2d/%2d | Loss: %.5f | Acc: %f%%\n", epoch + 1, epochs, average_loss, train_accuracy);
        
        out_file << average_loss << ", ";
    }

   std::cout << "Training is finished" << std::endl;
   std::cout << "Testing trained network" << std::endl;

   unsigned int correct_predictions = 0;
   unsigned int test_size = data.get_test_size();
   
   for (unsigned int i = 0; i < test_size; i++) 
   {
       float *current_input = &data.test_images_flat[i * INPUT_SIZE];
       int true_label = (int)data.test_labels[i];

       hidden_layer1.forward(current_input);
       hidden_layer2.forward(hidden_layer1.get_outputs());
       hidden_layer3.forward(hidden_layer2.get_outputs());
       output_layer.forward(hidden_layer3.get_outputs());

       const float *predictions = output_layer.get_outputs();
       int predicted_label = 0;
       float max_output = -1.0f;

       for (unsigned short j = 0; j < OUTPUT_SIZE; j++) 
       {
           if (predictions[j] > max_output) 
           {
               max_output = predictions[j];
               predicted_label = j;
           }
       }

       if (predicted_label == true_label)
           correct_predictions++;
   }

   float accuracy = (float)correct_predictions / test_size * 100.0f;

   std::cout << "Test Set Accuracy: " << accuracy << "%" << std::endl;
   out_file << accuracy << std::endl;
   out_file.close();

   return 0;
}
