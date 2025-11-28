#ifndef CIFAR_DATA_H
#define CIFAR_DATA_H

#include <fstream>
#include <iostream>
#include <string>

#define SIZE 32
#define CHANNELS 3
#define PIXELS_PER_IMAGE (SIZE * SIZE * CHANNELS)
#define NUM_CLASSES 10

typedef unsigned char BYTE;

struct Image {
    BYTE label;
    BYTE r[32 * 32];
    BYTE g[32 * 32];
    BYTE b[32 * 32];
};

class CIFAR_DATA 
{
    private: 
        // Temporary Raw Data (Deleted after constructor)
        Image *train_images_raw;
        Image *test_images_raw;

        std::string train_foldername;
        std::string test_filename;

        // Helper functions
        void get_image_batch(Image *images, const std::string filename, unsigned int start_index);
        void load_train_images_raw(void);
        void load_test_images_raw(void);
        
        // Data processing
        void normalize_data(Image *images, float *float_output, unsigned int count);
        void one_hot_encode(BYTE *raw_labels, float *encoded_output, unsigned int count);

    public:
        // Final Processed Data
        float *train_images_flat;     // Inputs (0.0 - 1.0)
        float *train_labels_encoded;  // Targets (One-Hot Vectors)
        BYTE *train_labels;           // Raw integers (for checking)
        unsigned int total_train_image_count;

        float *test_images_flat;
        float *test_labels_encoded;
        BYTE *test_labels;
        unsigned int total_test_image_count;

        CIFAR_DATA(const std::string &train_foldername, const std::string &test_filename);
        ~CIFAR_DATA(void);
        
        // --- COMPATIBILITY HELPERS FOR MAIN.CPP ---
        unsigned int get_train_size() const { return total_train_image_count; }
        unsigned int get_test_size() const { return total_test_image_count; }
        
        void print_stats();
};

#endif