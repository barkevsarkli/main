#include <iostream>
#include "data.cpp"
#include <fstream>
#include "net.cpp"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 10
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.0001f
#define EPOCHS 25

int main(void)
{
    srand(time(NULL));

    Data train_images_data("/home/bsarkli/microsoft_vscode/Supervised_Learning/CNNs/train-images-idx3-ubyte/train-images-idx3-ubyte", 
        "/home/bsarkli/microsoft_vscode/Supervised_Learning/CNNs/train-labels-idx1-ubyte/train-labels-idx1-ubyte", 0.2);

    Neuron hidden_layer(INPUT_SIZE, HIDDEN_SIZE);
    Neuron output_layer(HIDDEN_SIZE, OUTPUT_SIZE);
    std::cout << "main() | Training started" << std::endl;

    for (unsigned short epoch = 0; epoch < EPOCHS; epoch++) 
    {
        float total_loss = 0;

        for (unsigned short i = 0; i < train_images_data.get_train_size(); i++) 
        {
            Image current_image = train_images_data.get_train_images()[i];
            
            hidden_layer.forward(current_image.normalized_pixels);
            output_layer.forward(hidden_layer.get_outputs());

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
            hidden_layer.backward(output_layer.get_d_inputs());

            hidden_layer.update(LEARNING_RATE);
            output_layer.update(LEARNING_RATE);
        }

        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS 
                  << ", Average Loss: " << total_loss / train_images_data.get_train_size() << std::endl;
    }

   std::cout << "main() | Training finished" << std::endl;
   std::cout << "main() | Starting Evaluation on Test Set" << std::endl;

   unsigned short correct_predictions = 0;
   
   for (unsigned short i = 0; i < train_images_data.get_test_size(); i++) 
   {
       Image current_image = train_images_data.get_test_images()[i];

       hidden_layer.forward(current_image.normalized_pixels);
       output_layer.forward(hidden_layer.get_outputs());

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

   return 0;
}