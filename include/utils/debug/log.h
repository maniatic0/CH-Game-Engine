/*
 * Debug Loggin utilities
 *
 */

#ifndef LOG_HEADER_GUARD
#define LOG_HEADER_GUARD

#ifdef NDEBUG

// More Detailed Debug Message, Only defined if NDEBUG is not defined
#define LONGLOG(message)

// Short Debug Message, Only defined if NDEBUG is not defined
#define LOG(message)

// Iterate More Detailed Debug Message, Only defined if NDEBUG is not defined
#define FORLONGLOG(expression, message)

// Iterate Short Debug Message, Only defined if NDEBUG is not defined
#define FORLOG(expression, message)

#else  // !NDEBUG

#include <iostream>

// More Detailed Debug Message, Only defined if NDEBUG is not defined
#define LONGLOG(message)                                             \
  std::cout << "(" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
            << "): \n"                                               \
            << message << std::endl

// Short Debug Message, Only defined if NDEBUG is not defined
#define LOG(message) std::cout << message << std::endl

// Iterate More Detailed Debug Message, Only defined if NDEBUG is not defined
#define FORLONGLOG(expression, message) \
  for (expression) {                    \
    LONGLOG(message);                   \
  }

// Iterate Short Debug Message, Only defined if NDEBUG is not defined
#define FORLOG(expression, message) \
  for (expression) {                \
    LOG(message);                   \
  }

#endif  // NDEBUG

#endif  // LOG_HEADER_GUARD
