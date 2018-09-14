#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pCallback) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pCallback);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT callback,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, callback, pAllocator);
  }
}

class HelloTriangleApplication {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  // GLFW Config
  GLFWwindow *window;
  const int WIDTH = 800;
  const int HEIGHT = 600;

  // Vulkan Config
  VkInstance instance;
  VkDebugUtilsMessengerEXT callback;

  const std::vector<const char *> validationLayers = {
      "VK_LAYER_LUNARG_standard_validation"};

#ifdef NDEBUG
  const bool enableValidationLayers = false;
#else   // !NDEBUG
  const bool enableValidationLayers = true;
#endif  // NDEBUG

  void initWindow() {
#ifdef _DEBUG
    std::cout << "Window Init Started" << std::endl;
#endif  // _DEBUG
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
#ifdef _DEBUG
    std::cout << "Window Init Successful" << std::endl;
#endif  // _DEBUG
  }

  bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers) {
      bool layerFound = false;

      for (const auto &layerProperties : availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }

      if (!layerFound) {
        return false;
      }
    }

    return true;
  }

  std::vector<const char *> getRequiredExtensions() {
    // VK Extension Checking
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                           extensions.data());

#ifdef _DEBUG
    std::cout << "Vulkan available extensions:" << std::endl;

    for (const auto &extension : extensions) {
      std::cout << "\t" << extension.extensionName << std::endl;
    }
#endif  // _DEBUG

    // GLFW required Extensions
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> reqExtensions(
        glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef _DEBUG
    std::cout << "GLFW Required Vulkan extensions:" << std::endl;

    for (int i = 0; i < glfwExtensionCount; i++) {
      std::cout << "\t" << glfwExtensions[i] << std::endl;
    }
#endif  // _DEBUG

    if (enableValidationLayers) {
      reqExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#ifdef _DEBUG
      std::cout << "Validation Layers Required Vulkan extensions:" << std::endl;
      std::cout << "\t" << VK_EXT_DEBUG_UTILS_EXTENSION_NAME << std::endl;
#endif  // _DEBUG
    }

// Extension Check
#ifdef _DEBUG
    std::cout << "Checking required Vulkan Extensions:" << std::endl;
#endif  // _DEBUG
    int maxSize = reqExtensions.size();
    std::vector<bool> extensionCheck(maxSize, false);

    int extIter = 0, maxiter = 0;
    for (const auto &extension : extensions) {
      for (extIter = 0; extIter < maxSize; extIter++) {
        if (!extensionCheck[extIter] &&
            strcmp(extension.extensionName, reqExtensions[extIter]) == 0) {
          extensionCheck[extIter] = true;
          maxiter++;
#ifdef _DEBUG
          std::cout << "\tFound: " << extension.extensionName << std::endl;
#endif  // _DEBUG
          if (maxiter == maxSize) {
#ifdef _DEBUG
            std::cout << "All required Vulkan Extensions found" << std::endl;
#endif  // _DEBUG
            break;
          }
        }
      }
    }

    if (maxiter != maxSize) {
      throw std::runtime_error("failed to get the required Vulkan extensions!");
    }

    return reqExtensions;
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData) {
    std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
  }

  void setupDebugCallback() {
    if (!enableValidationLayers) return;

#ifdef _DEBUG
    std::cout << "Vulkan Debug Callback Init Started" << std::endl;
#endif  // _DEBUG

    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;  // Optional

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr,
                                     &callback) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug callback!");
    }
#ifdef _DEBUG
    std::cout << "Vulkan Debug Callback Init Successful" << std::endl;
#endif  // _DEBUG
  }

  void createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "validation layers requested, but not available!");
    }

#ifdef _DEBUG
    std::cout << "Vulkan Instance Init Started" << std::endl;
#endif  // _DEBUG

    // VK Optional App Config
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    // VK Required Extensions and Validation Layers Config
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    /*
     As you'll see, the general pattern that object creation function parameters
     in Vulkan follow is:
      * Pointer to struct with creation info
      * Pointer to custom allocator callbacks, always nullptr in this tutorial
      * Pointer to the variable that stores the handle to the new object
    */
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }
#ifdef _DEBUG
    std::cout << "Vulkan Instance Init Successful" << std::endl;
#endif  // _DEBUG
  }

  void initVulkan() {
#ifdef _DEBUG
    std::cout << "Vulkan Init Started" << std::endl;
#endif  // _DEBUG
    createInstance();
    setupDebugCallback();
#ifdef _DEBUG
    std::cout << "Vulkan Init Successful" << std::endl;
#endif  // _DEBUG
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
#ifdef _DEBUG
    std::cout << "Cleanup Started" << std::endl;
#endif  // _DEBUG

    if (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
    }

    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();

#ifdef _DEBUG
    std::cout << "Cleanup Successful" << std::endl;
#endif  // _DEBUG
  }
};

int main() {
#ifdef _DEBUG
  std::cout << "_DEBUG MACRO ACTIVE" << std::endl;
#endif

#ifndef NDEBUG
  std::cout << "NDEBUG MACRO NOT ACTIVE" << std::endl;
#endif

  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}