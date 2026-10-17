#include "util.h"
#include <filesystem>
#include <fstream>
#include <system_error>

namespace util {

bool write_atomic(const std::string& path, const std::string& content)
{
    namespace fs = std::filesystem;
    const fs::path target(path);
    if (target.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(target.parent_path(), ec);
    }
    fs::path tmp = target;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f << content;
        f.flush();
        if (!f) return false;
    }
    std::error_code ec;
    fs::rename(tmp, target, ec);        // atomic on POSIX; replaces the target on Windows too
    if (ec) {
        fs::remove(target, ec);
        fs::rename(tmp, target, ec);
    }
    return !ec;
}

}  // namespace util
