#include "data.h"
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <stdexcept>

Data::Data(const std::string &filename_images, const std::string &filename_labels, float test_size)
{
    this->number_of_images = 0;
    this->number_of_labels = 0;
    this->train_size = 0;
    this->test_size = test_size;
    load_images(filename_images);
    load_labels(filename_labels);
    //flags[0] = 0;
    //flags[1] = 0;
    //flags[2] = 0;
    split_data(test_size);
    normalize_data();
    one_hot_encode_labels();
}

Data::~Data()
{
    delete[] images_train;
    delete[] images_test;
}

void Data::load_images(const std::string &filename /*, const std::string &filename2*/)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("load_images() | Failed to open file: " + filename);

    uint32_t magic_number = read_big_endian_integer(file);

    if (magic_number != 2051)
        throw std::runtime_error("load_images() | Invalid magic number: " + std::to_string(magic_number));

    uint32_t number_of_images = read_big_endian_integer(file);
    this->number_of_images = number_of_images;

    uint32_t number_of_rows = read_big_endian_integer(file);

    if (number_of_rows != 28)
        throw std::runtime_error("load_images() | Invalid number of rows: " + std::to_string(number_of_rows));

    uint32_t number_of_columns = read_big_endian_integer(file);

    if (number_of_columns != 28)
        throw std::runtime_error("load_images() | Invalid number of columns: " + std::to_string(number_of_columns));

    images = new Image[number_of_images];

    for (uint32_t i = 0; i < number_of_images; i++)
    {
        file.read(reinterpret_cast<char*>(images[i].pixels), number_of_rows * number_of_columns);
        images[i].id = i;
    }

    std::cout << "Images loaded successfully" << std::endl;
    std::cout << "Number of images: " << number_of_images << std::endl;
    file.close();
}

void Data::load_labels(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("load_labels() | Failed to open file: " + filename);

    uint32_t magic_number = read_big_endian_integer(file);

    if (magic_number != 2049)
        throw std::runtime_error("load_labels() | Invalid magic number: " + std::to_string(magic_number));

    uint32_t number_of_labels = read_big_endian_integer(file);
    this->number_of_labels = number_of_labels;

    for (uint32_t i = 0; i < number_of_labels; i++)
        images[i].label = file.get();

    file.close();
    std::cout << "Labels loaded successfully" << std::endl;
    std::cout << "Number of labels: " << number_of_labels << std::endl;
}

uint32_t read_big_endian_integer(std::ifstream &file)
{
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

void Data::split_data(float test_size)
{
    if (test_size < 0 || test_size > 1)
        throw std::runtime_error("split_data() | Invalid test size: " + std::to_string(test_size));

    this->train_size = number_of_images * (1 - test_size);
    images_train = new Image[train_size];
    images_test = new Image[number_of_images - train_size];

    for (uint32_t i = 0; i < this->train_size; i++)
        images_train[i] = images[i];

    for (uint32_t i = this->train_size; i < this->number_of_images; i++)
        images_test[i - this->train_size] = images[i];

    std::cout << "Data split successfully" << std::endl;
    std::cout << "Train size: " << this->train_size << std::endl;
    std::cout << "Test size: " << this->number_of_images - this->train_size << std::endl;

    delete[] images;
}

void Data::normalize_data()
{
    for (uint32_t i = 0; i < this->train_size; i++)
        for (uint32_t j = 0; j < 28 * 28; j++)
            images_train[i].normalized_pixels[j] = images_train[i].pixels[j] / 255.0f;

    for (uint32_t i = 0; i < this->number_of_images - this->train_size; i++)
        for (uint32_t j = 0; j < 28 * 28; j++)
            images_test[i].normalized_pixels[j] = images_test[i].pixels[j] / 255.0f; // because of these steps our memory became 5 times larger

    std::cout << "Data normalized successfully" << std::endl;
}

void Data::one_hot_encode_labels()
{
    for (uint32_t i = 0; i < this->train_size; i++)
        one_hot_encode(images_train[i].label, images_train[i].one_hot_encoded_label);

    for (uint32_t i = 0; i < this->number_of_images - this->train_size; i++)
        one_hot_encode(images_test[i].label, images_test[i].one_hot_encoded_label);

    std::cout << "Labels one-hot encoded successfully" << std::endl;
}

void Data::one_hot_encode(uint8_t label, uint8_t* output_vector)
{
    for (uint8_t i = 0; i < 10; i++)
        output_vector[i] = 0;

    if (label < 10)
        output_vector[label] = 1;
}

void Data::display_image(const Image& image, const std::string& filename)
{
    const int width = 28;
    const int height = 28;
    const int row_padded = (width * 3 + 3) & (~3);
    const int filesize = 14 + 40 + row_padded * height;

    unsigned char bmpfileheader[14] = {
        'B','M',
        0,0,0,0,
        0,0,
        0,0,
        54,0,0,0
    };

    unsigned char bmpinfoheader[40] = {
        40,0,0,0,
        0,0,0,0,
        0,0,0,0,
        1,0,
        24,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0
    };

    bmpfileheader[2] = (unsigned char)(filesize);
    bmpfileheader[3] = (unsigned char)(filesize>> 8);
    bmpfileheader[4] = (unsigned char)(filesize>>16);
    bmpfileheader[5] = (unsigned char)(filesize>>24);

    bmpinfoheader[4] = (unsigned char)(width);
    bmpinfoheader[5] = (unsigned char)(width>> 8);
    bmpinfoheader[6] = (unsigned char)(width>>16);
    bmpinfoheader[7] = (unsigned char)(width>>24);

    bmpinfoheader[8]  = (unsigned char)(height);
    bmpinfoheader[9]  = (unsigned char)(height>> 8);
    bmpinfoheader[10] = (unsigned char)(height>>16);
    bmpinfoheader[11] = (unsigned char)(height>>24);

    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("display_image() | Failed to open file: " + filename);

    file.write(reinterpret_cast<char*>(bmpfileheader), 14);
    file.write(reinterpret_cast<char*>(bmpinfoheader), 40);

    unsigned char row[row_padded];
    for (int y = height - 1; y >= 0; y--) {
        int idx = 0;
        for (int x = 0; x < width; x++) {
            uint8_t pixel = image.pixels[y * width + x];
            row[idx++] = pixel;
            row[idx++] = pixel;
            row[idx++] = pixel;
        }
        while (idx < row_padded) row[idx++] = 0;
        file.write(reinterpret_cast<char*>(row), row_padded);
    }
    file.close();
}
