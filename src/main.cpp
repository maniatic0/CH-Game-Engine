#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>

// GLFW Config
const int WIDTH = 800;
const int HEIGHT = 600;

// TODO: Create DebugUtils Wrapper Class
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

const std::vector<const char *> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else   // !NDEBUG
const bool enableValidationLayers = true;
#endif  // NDEBUG

struct QueueFamilyIndices {
  int graphicsFamily = -1;

  bool isComplete() { return graphicsFamily >= 0; }
};

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

  // Vulkan Config
  VkInstance instance;
  VkDebugUtilsMessengerEXT callback;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

  void initWindow() {
#ifndef NDEBUG
    std::cout << "Window Init Started" << std::endl;
#endif  // !NDEBUG
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
#ifndef NDEBUG
    std::cout << "Window Init Successful" << std::endl;
#endif  // !NDEBUG
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

#ifndef NDEBUG
    std::cout << "Vulkan available extensions:" << std::endl;

    for (const auto &extension : extensions) {
      std::cout << "\t" << extension.extensionName << std::endl;
    }
#endif  // !NDEBUG

    // GLFW required Extensions
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> reqExtensions(
        glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifndef NDEBUG
    std::cout << "GLFW Required Vulkan extensions:" << std::endl;

    for (int i = 0; i < glfwExtensionCount; i++) {
      std::cout << "\t" << glfwExtensions[i] << std::endl;
    }
#endif  // !NDEBUG

    if (enableValidationLayers) {
      reqExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#ifndef NDEBUG
      std::cout << "Validation Layers Required Vulkan extensions:" << std::endl;
      std::cout << "\t" << VK_EXT_DEBUG_UTILS_EXTENSION_NAME << std::endl;
#endif  // !NDEBUG
    }

// Extension Check
#ifndef NDEBUG
    std::cout << "Checking required Vulkan Extensions:" << std::endl;
#endif  // !NDEBUG
    int maxSize = reqExtensions.size();
    std::vector<bool> extensionCheck(maxSize, false);

    int extIter = 0, maxiter = 0;
    for (const auto &extension : extensions) {
      for (extIter = 0; extIter < maxSize; extIter++) {
        if (!extensionCheck[extIter] &&
            strcmp(extension.extensionName, reqExtensions[extIter]) == 0) {
          extensionCheck[extIter] = true;
          maxiter++;
#ifndef NDEBUG
          std::cout << "\tFound: " << extension.extensionName << std::endl;
#endif  // !NDEBUG
          if (maxiter == maxSize) {
#ifndef NDEBUG
            std::cout << "All required Vulkan Extensions found" << std::endl;
#endif  // !NDEBUG
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

#ifndef NDEBUG
    std::cout << "Vulkan Debug Callback Init Started" << std::endl;
#endif  // !NDEBUG

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
#ifndef NDEBUG
    std::cout << "Vulkan Debug Callback Init Successful" << std::endl;
#endif  // !NDEBUG
  }

  void createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "validation layers requested, but not available!");
    }

#ifndef NDEBUG
    std::cout << "Vulkan Instance Init Started" << std::endl;
#endif  // !NDEBUG

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
#ifndef NDEBUG
    std::cout << "Vulkan Instance Init Successful" << std::endl;
#endif  // !NDEBUG
  }

  void initVulkan() {
#ifndef NDEBUG
    std::cout << "Vulkan Init Started" << std::endl;
#endif  // !NDEBUG
    createInstance();
    setupDebugCallback();
    pickPhysicalDevice();
#ifndef NDEBUG
    std::cout << "Vulkan Init Successful" << std::endl;
#endif  // !NDEBUG
  }

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueCount > 0 &&
          queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphicsFamily = i;
      }

      if (indices.isComplete()) {
        break;
      }

      i++;
    }

    return indices;
  }

  // Only Vulkan Support Required
  bool isDeviceSuitableOnlyVulkan(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    return indices.isComplete();
  }

  // Only Vulkan Support, Dedicated GPU and Geometry Shaders Required
  bool isDeviceSuitable(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    QueueFamilyIndices indices = findQueueFamilies(device);

    return deviceProperties.deviceType ==
               VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
           deviceFeatures.geometryShader && indices.isComplete();
  }

  // Better Rating Way
  int rateDeviceSuitability(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    QueueFamilyIndices indices = findQueueFamilies(device);

    int score = 0;

    if (!indices.isComplete()) {
#ifndef NDEBUG
      std::cout << "\t" << deviceProperties.deviceName << " score: " << 0
                << std::endl;
#endif  // !NDEBUG
      return 0;
    }

    // Application can't function without geometry shaders
    if (!deviceFeatures.geometryShader) {
#ifndef NDEBUG
      std::cout << "\t" << deviceProperties.deviceName << " score: " << 0
                << std::endl;
#endif  // !NDEBUG
      return 0;
    }

    // Discrete GPUs have a significant performance advantage
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      score += 1000;
    }

    // Maximum possible size of textures affects graphics quality
    score += deviceProperties.limits.maxImageDimension2D;

#ifndef NDEBUG
    std::cout << "\t" << deviceProperties.deviceName << " score: " << score
              << std::endl;
#endif  // !NDEBUG

    return score;
  }

  void pickPhysicalDevice() {
#ifndef NDEBUG
    std::cout << "Vulkan Physical Device Selection Started" << std::endl;
#endif  // !NDEBUG
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

#ifndef NDEBUG
    std::cout << "Rating Vulkan Physical Devices: " << std::endl;
#endif  // !NDEBUG

    /* Only bool selection
    for (const auto &device : devices) {
      if (isDeviceSuitableOnlyVulkan(device)) {
        physicalDevice = device;
        break;
      }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable GPU!");
    } */

    // Use an ordered map to automatically sort candidates by increasing score
    std::multimap<int, VkPhysicalDevice> candidates;

    for (const auto &device : devices) {
      int score = rateDeviceSuitability(device);
      candidates.insert(std::make_pair(score, device));
    }

    // Check if the best candidate is suitable at all
    if (candidates.rbegin()->first > 0) {
      physicalDevice = candidates.rbegin()->second;
    } else {
      throw std::runtime_error("failed to find a suitable GPU!");
    }

#ifndef NDEBUG
    std::cout << "Vulkan Physical Device Selection Successful" << std::endl;
#endif  // !NDEBUG
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
#ifndef NDEBUG
    std::cout << "Cleanup Started" << std::endl;
#endif  // !NDEBUG

    if (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
    }

    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();

#ifndef NDEBUG
    std::cout << "Cleanup Successful" << std::endl;
#endif  // !NDEBUG
  }
};

int main() {
#ifndef NDEBUG
  std::cout << "NDEBUG MACRO NOT DEFINED" << std::endl;
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