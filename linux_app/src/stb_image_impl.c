/**
 * stb_image implementation file
 * 
 * This file provides the implementation of stb_image library.
 * It should be compiled separately and linked with the main application.
 */

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#define STBI_NO_SIMD  // Disable SIMD for ARM compatibility
#include "stb_image.h"
