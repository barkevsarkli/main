#ifndef CHECK_H
#define CHECK_H

#include <cstdio>
#include <cstdlib>
#include <string>

// Minimal test harness: CHECK records a failure and keeps going; finish() sets the exit code.
static int g_failures = 0, g_checks = 0;

#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s | ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                                  \
            std::fprintf(stderr, "\n");                                         \
        }                                                                       \
    } while (0)

static int finish(const char* name)
{
    std::printf("%s: %d checks, %d failures -> %s\n", name, g_checks, g_failures, g_failures ? "FAIL" : "PASS");
    return g_failures ? 1 : 0;
}

#endif
