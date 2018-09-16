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

#else  // !NDEBUG

#include <iostream>

// More Detailed Debug Message, Only defined if NDEBUG is not defined
#define LONGLOG(message)                                             \
  std::cout << "(" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
            << "): \n\t" << message << std::endl

// Short Debug Message, Only defined if NDEBUG is not defined
#define LOG(message) std::cout << message << std::endl

#endif  // NDEBUG

#endif  // LOG_HEADER_GUARD
