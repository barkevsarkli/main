#ifndef MAPPED_FILE_H
#define MAPPED_FILE_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// Read-only view of a whole file, behind one small platform layer.
//
// version5 mapped its datasets with POSIX mmap so that many worker processes share one
// copy in the page cache instead of a private copy each (on an 8 GB machine six private
// CIFAR-10 copies pushed the system into swap). version6 keeps that, and adds
//   * CreateFileMapping / MapViewOfFile on Windows (_WIN32), and
//   * a plain std::ifstream read into a private buffer, selected with --no-mmap.
// All three hand the same bytes to the loaders; tests/test_loader.cpp checks that the
// normalised floats are identical for every MNIST and CIFAR-10 image.
class MappedFile
{
public:
    MappedFile() = default;
    MappedFile(const std::string& path, bool use_mmap) { open(path, use_mmap); }
    ~MappedFile() { close(); }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& o) noexcept { *this = std::move(o); }
    MappedFile& operator=(MappedFile&& o) noexcept;

    void open(const std::string& path, bool use_mmap);
    void close();

    const uint8_t* data() const { return base; }
    size_t         size() const { return len; }
    bool           mapped() const { return is_mapped; }

private:
    const uint8_t*       base = nullptr;
    size_t               len = 0;
    bool                 is_mapped = false;
    std::vector<uint8_t> buffer;          // --no-mmap backing store
#ifdef _WIN32
    void* file_handle = nullptr;          // HANDLE
    void* map_handle  = nullptr;          // HANDLE
#endif
};

#endif
