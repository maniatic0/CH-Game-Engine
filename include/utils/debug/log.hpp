/*
 * Debug Loggin utilities
 *
 */

#ifndef LOG_HEADER_GUARD
#define LOG_HEADER_GUARD

#ifdef NDEBUG

// More Detailed Debug Message, Only defined if NDEBUG is not defined
#define LONGLOG(message, ...)

// Short Debug Message, Only defined if NDEBUG is not defined
#define LOG(message, ...)

// Iterate More Detailed Debug Message, Only defined if NDEBUG is not defined
#define FORLONGLOG(expression, message, ...)

// Iterate Short Debug Message, Only defined if NDEBUG is not defined
#define FORLOG(expression, message, ...)

#else  // !NDEBUG

#include <cstdio>

// Macro to String Expansion
// https://stackoverflow.com/questions/19343205/c-concatenating-file-and-line-macros
#define STRING_1(x) #x
#define STRING_2(x) STRING_1(x)
#define STRING_3(x) STRING_2(x)

// See this for VARIADIC trick
// https://stackoverflow.com/questions/5891221/variadic-macros-with-zero-arguments

// More Detailed Debug Message, Only defined if NDEBUG is not defined
#define LONGLOG(message, ...)                                      \
  std::printf(__FILE__ ":" STRING_2(__LINE__) ":%s \n", __func__); \
  std::printf(message, ##__VA_ARGS__)

// Short Debug Message, Only defined if NDEBUG is not defined
#define LOG(message, ...) std::printf(message, ##__VA_ARGS__)

// Iterate More Detailed Debug Message, Only defined if NDEBUG is not defined
#define FORLONGLOG(expression, message, ...) \
  for (expression) {                         \
    LONGLOG(message, ##__VA_ARGS__);         \
  }

// Iterate Short Debug Message, Only defined if NDEBUG is not defined
#define FORLOG(expression, message, ...) \
  for (expression) {                     \
    LOG(message, ##__VA_ARGS__);         \
  }

#endif  // NDEBUG

#endif  // LOG_HEADER_GUARD
