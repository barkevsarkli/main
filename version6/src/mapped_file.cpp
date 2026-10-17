#include "mapped_file.h"
#include <fstream>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <unistd.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
#endif

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept
{
    if (this != &o) {
        close();
        buffer = std::move(o.buffer);
        len = o.len;
        is_mapped = o.is_mapped;
        base = is_mapped ? o.base : buffer.data();
#ifdef _WIN32
        file_handle = o.file_handle; map_handle = o.map_handle;
        o.file_handle = nullptr; o.map_handle = nullptr;
#endif
        o.base = nullptr; o.len = 0; o.is_mapped = false;
    }
    return *this;
}

void MappedFile::open(const std::string& path, bool use_mmap)
{
    close();
    if (!use_mmap) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("MappedFile | cannot open " + path);
        const std::streamoff n = f.tellg();
        if (n < 0) throw std::runtime_error("MappedFile | cannot size " + path);
        buffer.resize((size_t)n);
        f.seekg(0);
        if (n > 0 && !f.read(reinterpret_cast<char*>(buffer.data()), n))
            throw std::runtime_error("MappedFile | short read on " + path);
        base = buffer.data();
        len = buffer.size();
        is_mapped = false;
        return;
    }

#ifdef _WIN32
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (fh == INVALID_HANDLE_VALUE) throw std::runtime_error("MappedFile | cannot open " + path);
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) { CloseHandle(fh); throw std::runtime_error("MappedFile | cannot size " + path); }
    if (sz.QuadPart == 0) { CloseHandle(fh); base = nullptr; len = 0; return; }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) { CloseHandle(fh); throw std::runtime_error("MappedFile | CreateFileMapping failed on " + path); }
    void* p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(mh); CloseHandle(fh); throw std::runtime_error("MappedFile | MapViewOfFile failed on " + path); }
    file_handle = fh;
    map_handle = mh;
    base = static_cast<const uint8_t*>(p);
    len = (size_t)sz.QuadPart;
    is_mapped = true;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("MappedFile | cannot open " + path);
    struct stat st{};
    if (fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("MappedFile | fstat failed on " + path); }
    if (st.st_size == 0) { ::close(fd); base = nullptr; len = 0; return; }
    void* p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);                    // the mapping keeps its own reference to the file
    if (p == MAP_FAILED) throw std::runtime_error("MappedFile | mmap failed on " + path);
    // Training reads samples in shuffled order, so read-ahead tuned for streaming is wasted.
    madvise(p, (size_t)st.st_size, MADV_RANDOM);
    base = static_cast<const uint8_t*>(p);
    len = (size_t)st.st_size;
    is_mapped = true;
#endif
}

void MappedFile::close()
{
    if (is_mapped && base) {
#ifdef _WIN32
        UnmapViewOfFile(base);
        if (map_handle)  CloseHandle((HANDLE)map_handle);
        if (file_handle) CloseHandle((HANDLE)file_handle);
        map_handle = file_handle = nullptr;
#else
        munmap(const_cast<uint8_t*>(base), len);
#endif
    }
    buffer.clear();
    buffer.shrink_to_fit();
    base = nullptr;
    len = 0;
    is_mapped = false;
}
