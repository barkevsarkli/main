#ifndef DATA_H
#define DATA_H

#include <string>
#include <fstream>
#include <cstdint>

struct Image {
    uint8_t pixels[28 * 28];
    float normalized_pixels[28 * 28];
    uint8_t one_hot_encoded_label[10];
    uint8_t label;
    uint32_t id;
};

class Data {
private:
    Image* images;
    Image* images_train;
    Image* images_test;
    uint32_t number_of_images;
    uint32_t number_of_labels;
    uint32_t train_size;
    float test_size;

    void load_images(const std::string &filename);
    void load_labels(const std::string &filename);
    void split_data(float test_size);
    void normalize_data();
    void one_hot_encode_labels();
    void one_hot_encode(uint8_t label, uint8_t* output_vector);

public:
    Data(const std::string &filename_images, const std::string &filename_labels, float test_size);
    ~Data();
    
    void display_image(const Image& image, const std::string& filename);
    
    // Getters for accessing data
    Image* get_train_images() { return images_train; }
    Image* get_test_images() { return images_test; }
    uint32_t get_train_size() { return train_size; }
    uint32_t get_test_size() { return number_of_images - train_size; }
    uint32_t get_number_of_images() { return number_of_images; }
};

// Helper function
uint32_t read_big_endian_integer(std::ifstream &file);

#endif // DATA_H
