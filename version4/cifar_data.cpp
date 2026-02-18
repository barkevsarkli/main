#include "cifar_data.h"
#include <fstream>
#include <iostream>
#include <string>


CIFAR_DATA::CIFAR_DATA(const std::string &train_foldername, const std::string &test_filename) 
{  
    this->train_foldername = train_foldername;
    this->test_filename = test_filename;
    
    total_train_image_count = 50000;
    total_test_image_count = 10000;

    train_images_raw = new Image[total_train_image_count];
    test_images_raw = new Image[total_test_image_count];

    train_images_flat = new float[total_train_image_count * PIXELS_PER_IMAGE];
    train_labels_encoded = new float[total_train_image_count * NUM_CLASSES];
    train_labels = new BYTE[total_train_image_count];
    
    test_images_flat = new float[total_test_image_count * PIXELS_PER_IMAGE];
    test_labels_encoded = new float[total_test_image_count * NUM_CLASSES];
    test_labels = new BYTE[total_test_image_count];

    std::cout << "Loading CIFAR-10..." << std::endl;
    load_train_images_raw();
    load_test_images_raw();

    normalize_data(train_images_raw, train_images_flat, total_train_image_count);
    normalize_data(test_images_raw, test_images_flat, total_test_image_count);

    for(unsigned int i=0; i<total_train_image_count; i++) train_labels[i] = train_images_raw[i].label;
    for(unsigned int i=0; i<total_test_image_count; i++) test_labels[i] = test_images_raw[i].label;

    one_hot_encode(train_labels, train_labels_encoded, total_train_image_count);
    one_hot_encode(test_labels, test_labels_encoded, total_test_image_count);

    delete[] train_images_raw;
    delete[] test_images_raw;
    
    train_images_raw = nullptr;
    test_images_raw = nullptr;
    
    std::cout << "CIFAR-10 Loaded successfully. Raw memory freed." << std::endl;
}

CIFAR_DATA::~CIFAR_DATA(void) 
{
    delete[] train_images_flat;
    delete[] test_images_flat;
    delete[] train_labels_encoded;
    delete[] test_labels_encoded;
    delete[] train_labels;
    delete[] test_labels;
}

void CIFAR_DATA::get_image_batch(Image *images, const std::string filename, unsigned int start_index)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Error: " << filename << " couldn't open" << std::endl;
        exit(1); 
    }

    for (unsigned int i = start_index; i < start_index + 10000; i++) {
        images[i].label = (BYTE)file.get();
        file.read((char*)images[i].r, 32 * 32);
        file.read((char*)images[i].g, 32 * 32);
        file.read((char*)images[i].b, 32 * 32);
    }
    file.close();
}

void CIFAR_DATA::load_train_images_raw(void)
{
    get_image_batch(train_images_raw, train_foldername + "data_batch_1.bin", 0);
    get_image_batch(train_images_raw, train_foldername + "data_batch_2.bin", 10000);
    get_image_batch(train_images_raw, train_foldername + "data_batch_3.bin", 20000);
    get_image_batch(train_images_raw, train_foldername + "data_batch_4.bin", 30000);
    get_image_batch(train_images_raw, train_foldername + "data_batch_5.bin", 40000);
}

void CIFAR_DATA::load_test_images_raw(void)
{
    get_image_batch(test_images_raw, test_filename, 0);
}

void CIFAR_DATA::normalize_data(Image *images, float *float_output, unsigned int count)
{
    unsigned long long float_idx = 0;
    for (unsigned int i = 0; i < count; i++) {
        for (int j = 0; j < 1024; j++) float_output[float_idx++] = images[i].r[j] / 255.0f;
        for (int j = 0; j < 1024; j++) float_output[float_idx++] = images[i].g[j] / 255.0f;
        for (int j = 0; j < 1024; j++) float_output[float_idx++] = images[i].b[j] / 255.0f;
    }
}

void CIFAR_DATA::one_hot_encode(BYTE *raw_labels, float *encoded_output, unsigned int count)
{
    for (unsigned int i = 0; i < count; i++) {
        int label = (int)raw_labels[i];
        unsigned int base_index = i * NUM_CLASSES;
        for (int j = 0; j < NUM_CLASSES; j++) {
            encoded_output[base_index + j] = 0.0f;
        }
        encoded_output[base_index + label] = 1.0f;
    }
}

void CIFAR_DATA::print_stats() 
{
    std::cout << "--- Data Stats ---" << std::endl;
    std::cout << "Train Images: " << total_train_image_count << std::endl;
    std::cout << "Test Images:  " << total_test_image_count << std::endl;
}
