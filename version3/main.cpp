#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "data.h"
#include "net.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                << " <func1> <func2> <seed> <epochs>" << std::endl;
        return 1;
    }

    std::srand(std::stoi(argv[3]));

    std::string activation_type1 = argv[1];
    std::string activation_type2 = argv[2];
    unsigned short epochs = std::stoi(argv[4]);

    float learning_rate = LEARNING_RATE;
    if (activation_type1 == "sigmoid" || activation_type2 == "sigmoid") {
        learning_rate = 0.01f;  // Same as ReLU but with clipping protection
        std::cout << "Using learning rate for sigmoid: " << learning_rate << std::endl;
    }
    else if (activation_type1 == "tanh" || activation_type2 == "tanh") {
        learning_rate = 0.01f;  // Same as ReLU
        std::cout << "Using learning rate for tanh: " << learning_rate << std::endl;
    }

    std::string filename = "results_" + activation_type1 + "_" + activation_type2 + "_" + std::to_string(HIDDEN_SIZE) + ".csv";

    std::ofstream out_file(filename, std::ios::app);
    if (!out_file.is_open()) 
    {
        std::cerr << "main() | Failed to open output file." << std::endl;
        return 1;
    }

    std::cout << "main() | Using activation functions: " << activation_type1 << " and " << activation_type2 << std::endl;
    std::cout << "main() | Hidden layer size: " << HIDDEN_SIZE << ", Actual learning rate: " << learning_rate << ", epochs: " << epochs << std::endl;
    out_file << argv[3] << "," << activation_type1 << "," << activation_type2 << ",";

    Data train_images_data("/Users/khaos/Desktop/main-main/train-images-idx3-ubyte/train-images-idx3-ubyte", 
        "/Users/khaos/Desktop/main-main/train-labels-idx1-ubyte/train-labels-idx1-ubyte", 0.2);

    Neuron hidden_layer1(INPUT_SIZE, HIDDEN_SIZE, activation_type1, activation_type2);
    Neuron hidden_layer2(HIDDEN_SIZE, HIDDEN_SIZE, activation_type1, activation_type2);
    Neuron output_layer(HIDDEN_SIZE, OUTPUT_SIZE, activation_type1, activation_type2);

    std::cout << "main() | Using the same initials for all runs." << std::endl;

    for (unsigned short epoch = 0; epoch < epochs; epoch++) 
    {
        float total_loss = 0;
        unsigned short correct_train_predictions = 0;

        for (unsigned short i = 0; i < train_images_data.get_train_size(); i++) 
        {
            Image current_image = train_images_data.get_train_images()[i];
            
            hidden_layer1.forward(current_image.normalized_pixels);
            hidden_layer2.forward(hidden_layer1.get_outputs());
            output_layer.forward(hidden_layer2.get_outputs());

            const float *predictions = output_layer.get_outputs();
            float loss = 0;
            std::vector<float> d_loss(OUTPUT_SIZE);

            int predicted_label = 0;
            float max_output = -1.0f;

            for (unsigned short j = 0; j < OUTPUT_SIZE; j++) 
            {
                float error = predictions[j] - current_image.one_hot_encoded_label[j];
                
                // Check for NaN in predictions
                if (std::isnan(predictions[j])) {
                    std::cerr << "NaN detected in prediction at epoch " << epoch + 1 << ", sample " << i << std::endl;
                    return 1;
                }
                
                loss += error * error;
                d_loss[j] = 2 * error;

                // --- CORRECTED ACCURACY ---
                // Find the index (label) with the highest prediction value
                if (predictions[j] > max_output) 
                {
                    max_output = predictions[j];
                    predicted_label = j;
                }
                // --- END CORRECTION ---
            }

            // --- CORRECTED ACCURACY ---
            // Compare the predicted label to the true label
            if (predicted_label == current_image.label)
            {
                correct_train_predictions++;
            }
            // --- END CORRECTION ---

            total_loss += loss / OUTPUT_SIZE;

            output_layer.backward(d_loss.data());
            hidden_layer2.backward(output_layer.get_d_inputs());
            hidden_layer1.backward(hidden_layer2.get_d_inputs());

            hidden_layer1.update(learning_rate);
            hidden_layer2.update(learning_rate);
            output_layer.update(learning_rate);
        }

        float train_accuracy = (float)correct_train_predictions / train_images_data.get_train_size() * 100.0f;

        float average_loss = total_loss / train_images_data.get_train_size();
        
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << ", Average Loss: " << average_loss << ", Accuracy: " << train_accuracy << "%" << std::endl;
        
        out_file << average_loss <<", ";
    }

   std::cout << "main() | Training finished" << std::endl;
   std::cout << "main() | Starting Evaluation on Test Set" << std::endl;

   unsigned short correct_predictions = 0;
   
   for (unsigned short i = 0; i < train_images_data.get_test_size(); i++) 
   {
       Image current_image = train_images_data.get_test_images()[i];

       hidden_layer1.forward(current_image.normalized_pixels);
       hidden_layer2.forward(hidden_layer1.get_outputs());
       output_layer.forward(hidden_layer2.get_outputs());

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

       if (predicted_label == current_image.label)
           correct_predictions++;
   }

   float accuracy = (float)correct_predictions / train_images_data.get_test_size() * 100.0f;

   std::cout << "Test Set Accuracy: " << accuracy << "%" << std::endl;
   std::cout << "Correctly predicted " << correct_predictions << " out of " 
             << train_images_data.get_test_size() << " images." << std::endl;

   out_file << accuracy << std::endl;
        
   out_file.close();

   return 0;
}