#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_set>

#include <utils/files/files_config.h>
#include <utils/debug/log.hpp>
#include <utils/files/binary_loader.hpp>
#include <utils/files/file_path.hpp>

// TODO: Move to other File
// Camera Uniforms
struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

// TODO: Move Vertex Config to new file
// Vertex Config
struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static VkVertexInputBindingDescription getBindingDescription() {
    /*
    All of our per-vertex data is packed together in one array, so we're only
    going to have one binding. The binding parameter specifies the index of the
    binding in the array of bindings. The stride parameter specifies the number
    of bytes from one entry to the next, and the inputRate parameter can have
    one of the following values:

    VK_VERTEX_INPUT_RATE_VERTEX: Move to the next data entry after each vertex
    VK_VERTEX_INPUT_RATE_INSTANCE: Move to the next data entry after each
    instance We're not going to use instanced rendering, so we'll stick to
    per-vertex data.
    */

    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 3>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
  }
};

const std::vector<Vertex> vertices = {  //
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}} //
};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

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
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
  std::optional<uint32_t> transferFamily;

  bool isComplete() {
    return graphicsFamily.has_value() && presentFamily.has_value() &&
           transferFamily.has_value();
  }
};

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

const int MAX_FRAMES_IN_FLIGHT = 2;

struct CommandPoolFamily {
  VkCommandPool commandPool;
  VkQueue queue;
};

struct CommandBufferFamily {
  VkCommandBuffer commandBuffer;
  CommandPoolFamily *commandPoolFamily;
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
  VkSurfaceKHR surface;

  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device;

  VkQueue graphicsQueue;
  VkQueue presentQueue;
  VkQueue transferQueue;

  VkSwapchainKHR swapChain;
  std::vector<VkImage> swapChainImages;
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;

  std::vector<VkImageView> swapChainImageViews;

  VkRenderPass renderPass;

  VkDescriptorSetLayout descriptorSetLayout;
  VkPipelineLayout pipelineLayout;
  VkPipeline graphicsPipeline;

  std::vector<VkFramebuffer> swapChainFramebuffers;

  CommandPoolFamily graphicsCommandPool;
  CommandPoolFamily transferCommandPool;
  std::vector<VkCommandBuffer> commandBuffers;

  // Drawing
  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  std::vector<VkFence> inFlightFences;
  size_t currentFrame = 0;

  bool framebufferResized = false;

  // Vertex Buffer
  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;

  // Index Buffer
  VkBuffer indexBuffer;
  VkDeviceMemory indexBufferMemory;

  // Uniform Buffers
  std::vector<VkBuffer> uniformBuffers;
  std::vector<VkDeviceMemory> uniformBuffersMemory;

  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet> descriptorSets;

  // Image
  VkImage textureImage;
  VkDeviceMemory textureImageMemory;
  VkImageView textureImageView;
  VkSampler textureSampler;

  void initWindow() {
    LOG("Window Init Started");
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    // Arbitrary pointer, we use it to send the instance
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    LOG("Window Init Successful");
  }

  static void framebufferResizeCallback(GLFWwindow *window, int width,
                                        int height) {
    // Get the pointer we setup with the instance
    auto app = reinterpret_cast<HelloTriangleApplication *>(
        glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
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

    LOG("Vulkan available extensions:");
    FORLOG(const auto &extension : extensions, "\t" << extension.extensionName);

    // GLFW required Extensions
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> reqExtensions(
        glfwExtensions, glfwExtensions + glfwExtensionCount);

    LOG("GLFW Required Vulkan extensions:");
    FORLOG(int i = 0; i < glfwExtensionCount; i++, "\t" << glfwExtensions[i]);

    if (enableValidationLayers) {
      reqExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

      LOG("Validation Layers Required Vulkan extensions:");
      LOG("\t" << VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Extension Check
    LOG("Checking required Vulkan Extensions:");

    int maxSize = reqExtensions.size();
    std::vector<bool> extensionCheck(maxSize, false);

    int extIter = 0, maxiter = 0;
    for (const auto &extension : extensions) {
      for (extIter = 0; extIter < maxSize; extIter++) {
        if (!extensionCheck[extIter] &&
            strcmp(extension.extensionName, reqExtensions[extIter]) == 0) {
          extensionCheck[extIter] = true;
          maxiter++;

          LOG("\tFound: " << extension.extensionName);
          if (maxiter == maxSize) {
            LOG("All required Vulkan Extensions found!");
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

    LOG("Vulkan Debug Callback Init Started");

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
    LOG("Vulkan Debug Callback Init Successful");
  }

  void createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "validation layers requested, but not available!");
    }

    LOG("Vulkan Instance Init Started");

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

    LOG("Vulkan Instance Init Successful");
  }

  void createSurface() {
    LOG("Vulkan Surface Creation Started");
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }

    LOG("Vulkan Surface Creation Successful");
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
    VkBool32 presentSupport = false;
    bool separateTransferQueue = false;
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueCount > 0) {
        // Check if it is a graphics queue
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
          indices.graphicsFamily = i;
          if (!indices.transferFamily.has_value()) {
            indices.transferFamily = i;
            separateTransferQueue = false;
          }
          // Check if it is not a graphics queue and it is a transfer queue
        } else if (queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) {
          indices.transferFamily = i;
          separateTransferQueue = true;
        }
      }

      presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

      if (queueFamily.queueCount > 0 && presentSupport) {
        indices.presentFamily = i;
      }

      if (separateTransferQueue && indices.isComplete()) {
        break;
      }

      i++;
    }

    return indices;
  }

  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                              &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                           details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                                              &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
  }

  // Only Vulkan Support Required
  bool isDeviceSuitableOnlyVulkan(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      swapChainAdequate = !swapChainSupport.formats.empty() &&
                          !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
  }

  // Only Vulkan Support, Dedicated GPU and Geometry Shaders Required
  bool isDeviceSuitable(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    QueueFamilyIndices indices = findQueueFamilies(device);

    LOG("\t" << deviceProperties.deviceName << " supported Extensions:");
    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      swapChainAdequate = !swapChainSupport.formats.empty() &&
                          !swapChainSupport.presentModes.empty();
    }

    return deviceProperties.deviceType ==
               VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
           deviceFeatures.geometryShader && indices.isComplete() &&
           extensionsSupported && swapChainAdequate;
  }

  // Better Rating Way
  int rateDeviceSuitability(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    QueueFamilyIndices indices = findQueueFamilies(device);

    LOG("\t" << deviceProperties.deviceName << " supported Extensions:");
    bool extensionsSupported = checkDeviceExtensionSupport(device);

    int score = 0;

    // Application can't function without the extensions we need
    if (!extensionsSupported) {
      LOG("\t" << deviceProperties.deviceName << " score: " << 0);
      return 0;
    }

    // Check that swapchain supports at least one format and one present mode
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
    bool swapChainAdequate = !swapChainSupport.formats.empty() &&
                             !swapChainSupport.presentModes.empty();

    if (!swapChainAdequate) {
      LOG("\t" << deviceProperties.deviceName << " score: " << 0);
      return 0;
    }

    // Application can't function without the queues we need
    if (!indices.isComplete()) {
      LOG("\t" << deviceProperties.deviceName << " score: " << 0);
      return 0;
    }

    // Application can't function without geometry shaders
    if (!deviceFeatures.geometryShader) {
      LOG("\t" << deviceProperties.deviceName << " score: " << 0);
      return 0;
    }

    // Discrete GPUs have a significant performance advantage
    LOG("\t\tDiscrete GPU: "
        << (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                ? "Found"
                : "Not Found"));
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      score += 1000;
    }

    // Support for Anisotropy Samplers
    LOG("\t\tSupport for Anisotropy Samplers: "
        << (deviceFeatures.samplerAnisotropy ? "Found" : "Not Found"));
    if (deviceFeatures.samplerAnisotropy) {
      return 500;
    }

    // Maximum possible size of textures affects graphics quality
    score += deviceProperties.limits.maxImageDimension2D;

    LOG("\t" << deviceProperties.deviceName << " score: " << score);
    return score;
  }

  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    std::unordered_set<std::string> requiredExtensions(deviceExtensions.begin(),
                                                       deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
      LOG("\t\t" << extension.extensionName);
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  void pickPhysicalDevice() {
    LOG("Vulkan Physical Device Selection Started");

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    LOG("Rating Vulkan Physical Devices: ");

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

    LOG("Vulkan Physical Device Selection Successful");
  }

  void createLogicalDevice() {
    LOG("Vulkan Logical Device Creation Started");

    // Logical Device Queue Create Info
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::unordered_set<uint32_t> uniqueQueueFamilies = {
        indices.graphicsFamily.value(), indices.presentFamily.value(),
        indices.transferFamily.value()};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo = {};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    // Required Device Features (I.e geometry shader)
    VkPhysicalDeviceFeatures deviceFeatures = {};
    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);
    if (supportedFeatures.samplerAnisotropy) {
      deviceFeatures.samplerAnisotropy = VK_TRUE;
    } else {
      deviceFeatures.samplerAnisotropy = VK_FALSE;
    }

    // Logical Device Create Info
    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;

    // Required Extensions and Validation Layers
    LOG("Logical and Physical Device Required Extensions");
    FORLOG(auto &extension : deviceExtensions, "\t Found: " << extension);
    LOG("All Logical and Physical Device Required Extensions Found");
    // This is true if we are here

    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    // Get Graphics Queue
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

    // Get Transfer Quere
    vkGetDeviceQueue(device, indices.transferFamily.value(), 0, &transferQueue);

    LOG("Vulkan Logical Device Creation Successful");
  }

  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    // We want 8 bits per color, BRGA, stored non lineary
    // No Preferred Format
    if (availableFormats.size() == 1 &&
        availableFormats[0].format == VK_FORMAT_UNDEFINED) {
      return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }

    // Check for the format that supports what we want
    for (const auto &availableFormat : availableFormats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return availableFormat;
      }
    }

    // Return what is available
    return availableFormats[0];
  }

  VkPresentModeKHR chooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> availablePresentModes) {
    // Double Buffering
    VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

    for (const auto &availablePresentMode : availablePresentModes) {
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
        // Triple Buffering (less tearing and low latency)
        return availablePresentMode;
      } else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
        // Worst mode, may cause tearing
        bestMode = availablePresentMode;
      }
    }

    return bestMode;
  }

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
    // The swap extent is the resolution of the swap chain images and it's
    // almost always exactly equal to the resolution of the window that we're
    // drawing to
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    } else {
      // Get Current Window Size
      int width, height;
      glfwGetFramebufferSize(window, &width, &height);

      VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                                 static_cast<uint32_t>(height)};

      // Get the biggest resolution possible clamped by min/max extent
      actualExtent.width =
          std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                     capabilities.maxImageExtent.width);
      actualExtent.height =
          std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                     capabilities.maxImageExtent.height);

      return actualExtent;
    }
  }

  void createSwapChain() {
    LOG("Vulkan SwapChain Creation Started");
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    // Number of images in swapchain queue
    // Should be the min plus one just in case
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    // swapChainSupport.capabilities.maxImageCount = 0 implies no limit
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    // Always 1, except in  stereoscopic 3D application
    createInfo.imageArrayLayers = 1;

    // Direct Rendering
    // For postprocessing In that case you may use a value like
    // VK_IMAGE_USAGE_TRANSFER_DST_BIT instead and use a memory operation to
    // transfer the rendered image to a swap chain image.
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    /**
     * VK_SHARING_MODE_EXCLUSIVE: An image is owned by one queue family at a
     * time and ownership must be explicitly transfered before using it in
     * another queue family. This option offers the best performance.
     * VK_SHARING_MODE_CONCURRENT: Images can be used across multiple queue
     * families without explicit ownership transfers.
     */

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                     indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      createInfo.queueFamilyIndexCount = 0;      // Optional
      createInfo.pQueueFamilyIndices = nullptr;  // Optional
    }

    // If we want transforms like 90 degrees rotation. Current tranform to apply
    // no transform
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

    // The compositeAlpha field specifies if the alpha channel should be used
    // for blending with other windows in the window system. You'll almost
    // always want to simply ignore the alpha channel, hence
    // VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR.
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    createInfo.presentMode = presentMode;
    // If the clipped member is set to VK_TRUE then that means that we don't
    // care about the color of pixels that are obscured, for example because
    // another window is in front of them. Unless you really need to be able to
    // read these pixels back and get predictable results, you'll get the best
    // performance by enabling clipping.
    createInfo.clipped = VK_TRUE;

    // That leaves one last field, oldSwapChain. With Vulkan it's possible that
    // your swap chain becomes invalid or unoptimized while your application is
    // running, for example because the window was resized. In that case the
    // swap chain actually needs to be recreated from scratch and a reference to
    // the old one must be specified in this field.
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                            swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;

    LOG("Vulkan SwapChain Creation Successful");
  }

  VkImageView createImageView(VkImage image, VkFormat format) {
    VkImageViewCreateInfo viewInfo = {};

    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;

    // The viewType and format fields specify how the image data should be
    // interpreted. The viewType parameter allows you to treat images as 1D
    // textures, 2D textures, 3D textures and cube maps.
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;

    // The components field allows you to swizzle the color channels around.
    // For example, you can map all of the channels to the red channel for a
    // monochrome texture. You can also map constant values of 0 and 1 to a
    // channel. In our case we'll stick to the default mapping.
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    // The subresourceRange field describes what the image's purpose is and
    // which part of the image should be accessed. Our images will be used as
    // color targets without any mipmapping levels or multiple layers.
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
  }

  void createImageViews() {
    LOG("Vulkan Image View Creation Started");
    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      swapChainImageViews[i] =
          createImageView(swapChainImages[i], swapChainImageFormat);
    }

    LOG("Vulkan Image View Creation Successful");
  }

  VkShaderModule createShaderModule(const std::vector<char> &code) {
    // Creating a shader module is simple, we only need to specify a pointer to
    // the buffer with the bytecode and the length of it. This information is
    // specified in a VkShaderModuleCreateInfo structure. The one catch is that
    // the size of the bytecode is specified in bytes, but the bytecode pointer
    // is a uint32_t pointer rather than a char pointer. Therefore we will need
    // to cast the pointer with reinterpret_cast as shown below. When you
    // perform a cast like this, you also need to ensure that the data satisfies
    // the alignment requirements of uint32_t. Lucky for us, the data is stored
    // in an std::vector where the default allocator already ensures that the
    // data satisfies the worst case alignment requirements.
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
  }

  void createRenderPass() {
    LOG("Vulkan Render Pass Started");

    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

    // Clear color and depth from frame buffer
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    // Don't do anything with the stencil buffer
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    // Textures and framebuffers in Vulkan are represented by VkImage objects
    // with a certain pixel format, however the layout of the pixels in memory
    // can change based on what you're trying to do with an image.
    // VK_IMAGE_LAYOUT_UNDEFINED for initialLayout means that we don't care what
    // previous layout the image was in
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR = Images to be presented in the swap
    // chain
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Attachments config
    // The attachment parameter specifies which attachment to reference by its
    // index in the attachment descriptions array. Our array consists of a
    // single VkAttachmentDescription, so its index is 0. The layout specifies
    // which layout we would like the attachment to have during a subpass that
    // uses this reference. Vulkan will automatically transition the attachment
    // to this layout when the subpass is started. We intend to use the
    // attachment to function as a color buffer and the
    // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL layout will give us the best
    // performance, as its name implies.
    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Subpass
    VkSubpassDescription subpass = {};
    // Vulkan may also support compute subpasses in the future, so we have to be
    // explicit about this being a graphics subpass. Next, we specify the
    // reference to the color attachment:
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

    // The index of the attachment in this array is directly referenced from the
    // fragment shader with the layout(location = 0) out vec4 outColor
    // directive!
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    // Render Pass Config

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;

    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;

    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }

    LOG("Vulkan Render Pass Successful");
  }

  void createDescriptorSetLayout() {
    LOG("Vulkan Description Set Layout Creation Started");

    /*
     * The first two fields specify the binding used in the shader and the type
     * of descriptor, which is a uniform buffer object. It is possible for the
     * shader variable to represent an array of uniform buffer objects, and
     * descriptorCount specifies the number of values in the array. This could
     * be used to specify a transformation for each of the bones in a skeleton
     * for skeletal animation, for example. Our MVP transformation is in a
     * single uniform buffer object, so we're using a descriptorCount of 1.
     */
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;

    /*
     * We also need to specify in which shader stages the descriptor is going to
     * be referenced. The stageFlags field can be a combination of
     * VkShaderStageFlagBits values or the value VK_SHADER_STAGE_ALL_GRAPHICS.
     * In our case, we're only referencing the descriptor from the vertex
     * shader.
     */
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    uboLayoutBinding.pImmutableSamplers = nullptr;  // Optional, for images

    VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        uboLayoutBinding, samplerLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                    &descriptorSetLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor set layout!");
    }

    LOG("Vulkan Description Set Layout Creation Successful");
  }

  void createGraphicsPipeline() {
    LOG("Vulkan Graphics Pipeline Creation Started");
    auto vertShaderCode = utils::readAssetFile("triangle.vert.glsl.spv", true);
    auto fragShaderCode = utils::readAssetFile("triangle.frag.glsl.spv", true);

    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;

    vertShaderModule = createShaderModule(vertShaderCode);
    fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;

    // The next two members specify the shader module containing the code, and
    // the function to invoke. That means that it's possible to combine multiple
    // fragment shaders into a single shader module and use different entry
    // points to differentiate between their behaviors. In this case we'll stick
    // to the standard main, however.
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    // There is one more (optional) member, pSpecializationInfo, which we won't
    // be using here, but is worth discussing. It allows you to specify values
    // for shader constants. You can use a single shader module where its
    // behavior can be configured at pipeline creation by specifying different
    // values for the constants used in it. This is more efficient than
    // configuring the shader using variables at render time, because the
    // compiler can do optimizations like eliminating if statements that depend
    // on these values. If you don't have any constants like that, then you can
    // set the member to nullptr, which our struct initialization does
    // automatically.
    // vertShaderStageInfo.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

    // Hardcoded vertex input description
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    /*
    For no vertex data
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr;  // Optional
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;  // Optional
    */

    // Vertex Data Description
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable =
        VK_FALSE;  // Same vertex used by two triangles for example

    // Viewport size (normally all the window but it could be different)
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;  // Framebuffer depthbuffer
    viewport.maxDepth = 1.0f;  // Framebuffer depthbuffer

    // If we want to clip the framebuffer
    // It works by dropping pixels that fall outside of it in the rasterizer
    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    // Viewport Config
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // Rasterizer Config
    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    // Clamp depth rather than discard (useful for shadow mapping). Requires GPU
    // Feature
    rasterizer.depthClampEnable = VK_FALSE;

    // If we want to drop everything and not let geometry pass
    rasterizer.rasterizerDiscardEnable = VK_FALSE;

    // If we want point cloud, wireframe or filling. Point cloud and wireframe
    // requires GPU Feature
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

    // The lineWidth member is straightforward, it describes the thickness of
    // lines in terms of number of fragments. The maximum line width that is
    // supported depends on the hardware and any line thicker than 1.0f requires
    // you to enable the wideLines GPU feature.
    rasterizer.lineWidth = 1.0f;

    // Culling and Vertex Winding direction
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    // rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.frontFace =
        VK_FRONT_FACE_COUNTER_CLOCKWISE;  // OpenGL to Vulkan Fix

    // Depth Bias (Sometimes used with ShadowMapping)
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;  // Optional
    rasterizer.depthBiasClamp = 0.0f;           // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f;     // Optional

    // MSAA (disabled for now)
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;           // Optional
    multisampling.pSampleMask = nullptr;             // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE;  // Optional
    multisampling.alphaToOneEnable = VK_FALSE;       // Optional

    // If you are using a depth and/or stencil buffer, then you also need to
    // configure the depth and stencil tests using
    // VkPipelineDepthStencilStateCreateInfo. We don't have one right now, so we
    // can simply pass a nullptr instead of a pointer to such a struct. We'll
    // get back to it in the depth buffering chapter.

    // Color Blending per framebuffer
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;  // No blending
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
    colorBlendAttachment.dstColorBlendFactor =
        VK_BLEND_FACTOR_ZERO;                                        // Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;             // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
    colorBlendAttachment.dstAlphaBlendFactor =
        VK_BLEND_FACTOR_ZERO;                             // Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;  // Optional

    // Alpha blending example
    // colorBlendAttachment.blendEnable = VK_TRUE;
    // colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    // colorBlendAttachment.dstColorBlendFactor =
    // VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; colorBlendAttachment.colorBlendOp =
    // VK_BLEND_OP_ADD; colorBlendAttachment.srcAlphaBlendFactor =
    // VK_BLEND_FACTOR_ONE; colorBlendAttachment.dstAlphaBlendFactor =
    // VK_BLEND_FACTOR_ZERO; colorBlendAttachment.alphaBlendOp =
    // VK_BLEND_OP_ADD;

    // The second structure references the array of structures for all of the
    // framebuffers and allows you to set blend constants that you can use as
    // blend factors in the aforementioned calculations.
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;  // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;  // Optional
    colorBlending.blendConstants[1] = 0.0f;  // Optional
    colorBlending.blendConstants[2] = 0.0f;  // Optional
    colorBlending.blendConstants[3] = 0.0f;  // Optional
    // If you want to use the second method of blending (bitwise combination),
    // then you should set logicOpEnable to VK_TRUE. The bitwise operation can
    // then be specified in the logicOp field. Note that this will automatically
    // disable the first method, as if you had set blendEnable to VK_FALSE for
    // every attached framebuffer! The colorWriteMask will also be used in this
    // mode to determine which channels in the framebuffer will actually be
    // affected. It is also possible to disable both modes, as we've done here,
    // in which case the fragment colors will be written to the framebuffer
    // unmodified.

    // A limited amount of the state that we've specified in the previous
    // structs can actually be changed without recreating the pipeline. Examples
    // are the size of the viewport, line width and blend constants. If you want
    // to do that, then you'll have to fill in a
    // VkPipelineDynamicStateCreateInfo structure like this:

    /* VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                      VK_DYNAMIC_STATE_LINE_WIDTH};

    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates; */

    // You can use uniform values in shaders, which are globals similar to
    // dynamic state variables that can be changed at drawing time to alter the
    // behavior of your shaders without having to recreate them. They are
    // commonly used to pass the transformation matrix to the vertex shader, or
    // to create texture samplers in the fragment shader.
    // These uniform values need to be specified during pipeline creation by
    // creating a VkPipelineLayout object. Even though we won't be using them
    // until a future chapter, we are still required to create an empty pipeline
    // layout The structure also specifies push constants, which are another way
    // of passing dynamic values to shaders that we may get into in a future
    // chapter.
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;  // Optional
    pipelineLayoutInfo.pSetLayouts =
        &descriptorSetLayout;                          // Optional, Uniforms
    pipelineLayoutInfo.pushConstantRangeCount = 0;     // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr;  // Optional

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    // We start by referencing the array of VkPipelineShaderStageCreateInfo
    // structs.
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;

    // Then we reference all of the structures describing the fixed-function
    // stage.
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr;  // Optional
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;  // Optional

    // After that comes the pipeline layout, which is a Vulkan handle rather
    // than a struct pointer.
    pipelineInfo.layout = pipelineLayout;

    // And finally we have the reference to the render pass and the index of the
    // sub pass where this graphics pipeline will be used. It is also possible
    // to use other render passes with this pipeline instead of this specific
    // instance, but they have to be compatible with renderPass. The
    // requirements for compatibility are described here, but we won't be using
    // that feature in this tutorial.
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    // There are actually two more parameters: basePipelineHandle and
    // basePipelineIndex. Vulkan allows you to create a new graphics pipeline by
    // deriving from an existing pipeline. The idea of pipeline derivatives is
    // that it is less expensive to set up pipelines when they have much
    // functionality in common with an existing pipeline and switching between
    // pipelines from the same parent can also be done quicker. You can either
    // specify the handle of an existing pipeline with basePipelineHandle or
    // reference another pipeline that is about to be created by index with
    // basePipelineIndex. Right now there is only a single pipeline, so we'll
    // simply specify a null handle and an invalid index. These values are only
    // used if the VK_PIPELINE_CREATE_DERIVATIVE_BIT flag is also specified in
    // the flags field of VkGraphicsPipelineCreateInfo.
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;  // Optional
    pipelineInfo.basePipelineIndex = -1;               // Optional

    // The vkCreateGraphicsPipelines function actually has more parameters than
    // the usual object creation functions in Vulkan. It is designed to take
    // multiple VkGraphicsPipelineCreateInfo objects and create multiple
    // VkPipeline objects in a single call.
    //
    // The second parameter, for which we've passed the VK_NULL_HANDLE argument,
    // references an optional VkPipelineCache object. A pipeline cache can be
    // used to store and reuse data relevant to pipeline creation across
    // multiple calls to vkCreateGraphicsPipelines and even across program
    // executions if the cache is stored to a file. This makes it possible to
    // significantly speed up pipeline creation at a later time. We'll get into
    // this in the pipeline cache chapter.
    //
    // The graphics pipeline is required for all common drawing operations, so
    // it should also only be destroyed at the end of the program:
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    LOG("Vulkan Graphics Pipeline Creation Successful");
  }

  void createFramebuffers() {
    LOG("Vulkan Frame Buffer Creation Started");

    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      VkImageView attachments[] = {swapChainImageViews[i]};
      VkFramebufferCreateInfo framebufferInfo = {};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = renderPass;
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = attachments;
      framebufferInfo.width = swapChainExtent.width;
      framebufferInfo.height = swapChainExtent.height;
      framebufferInfo.layers = 1;

      if (vkCreateFramebuffer(device, &framebufferInfo, nullptr,
                              &swapChainFramebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }

    LOG("Vulkan Frame Buffer Creation Successful");
  }

  void createCommandPool() {
    LOG("Vulkan Command Pool Creation Started");

    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    graphicsCommandPool.queue = graphicsQueue;

    /*
    There are two possible flags for command pools:
    • VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Hint that command buffers
      are rerecorded with new commands very often (may change memory allocation
      behavior)
    • VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: Allow command
      buffers to be rerecorded individually, without this flag they all have
      to be reset together
    */
    poolInfo.flags = 0;  // Optional

    if (vkCreateCommandPool(device, &poolInfo, nullptr,
                            &graphicsCommandPool.commandPool) != VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool!");
    }

    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.transferFamily.value();
    transferCommandPool.queue = transferQueue;

    /*
    There are two possible flags for command pools:
    • VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Hint that command buffers
      are rerecorded with new commands very often (may change memory allocation
      behavior)
    • VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: Allow command
      buffers to be rerecorded individually, without this flag they all have
      to be reset together
    */
    poolInfo.flags = 0;  // Optional

    if (vkCreateCommandPool(device, &poolInfo, nullptr,
                            &transferCommandPool.commandPool) != VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool!");
    }

    LOG("Vulkan Command Pool Creation Successful");
  }

  void createCommandBuffers() {
    LOG("Vulkan Command Buffer Creation Started");

    commandBuffers.resize(swapChainFramebuffers.size());

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = graphicsCommandPool.commandPool;
    /*
    The level parameter specifies if the allocated command buffers are primary
    or secondary command buffers.
    * VK_COMMAND_BUFFER_LEVEL_PRIMARY: Can be submitted to a queue for
    execution, but cannot be called from other command buffers.
    * VK_COMMAND_BUFFER_LEVEL_SECONDARY: Cannot be submitted directly, but can
    be called from primary command buffers.
    */
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffers!");
    }

    for (size_t i = 0; i < commandBuffers.size(); i++) {
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      /*
      The flags parameter specifies how we’re going to use the command buffer.
      The following values are available: •
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command buffer will be
      rerecorded right after executing it once. •
      VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: This is a secondary
      command buffer that will be entirely within a single render pass. •
      VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: The command buffer can be
      resubmitted while it is also already pending execution.
      */
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
      // We have used the last flag because we may already be scheduling the
      // drawing commands for the next frame while the last frame is not
      // finished yet. The pInheritanceInfo parameter is only relevant for
      // secondary command buffers. It specifies which state to inherit from the
      // calling primary command buffers.
      beginInfo.pInheritanceInfo = nullptr;  // Optional

      if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
      }

      VkRenderPassBeginInfo renderPassInfo = {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = renderPass;
      renderPassInfo.framebuffer = swapChainFramebuffers[i];

      renderPassInfo.renderArea.offset = {0, 0};
      renderPassInfo.renderArea.extent = swapChainExtent;

// -Wmissing-braces is not working properly on clang
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
      VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
#pragma clang diagnostic pop
      renderPassInfo.clearValueCount = 1;
      renderPassInfo.pClearValues = &clearColor;

      vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphicsPipeline);

      // Fixed triangle draw
      // The actual vkCmdDraw function is a bit anticlimactic, but it’s so
      // simple because of all the information we specified in advance. It has
      // the following parameters, aside from the command buffer: • vertexCount:
      // Even though we don’t have a vertex buffer, we technically still have 3
      // vertices to draw. • instanceCount: Used for instanced rendering, use 1
      // if you’re not doing that. • firstVertex: Used as an offset into the
      // vertex buffer, defines the lowest value of gl_VertexIndex.
      // firstInstance: Used as an offset for instanced rendering, defines the
      // lowest value of gl_InstanceIndex.
      // vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);

      // Bind Vertex Buffers
      VkBuffer vertexBuffers[] = {vertexBuffer};
      VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

      vkCmdBindIndexBuffer(
          commandBuffers[i], indexBuffer, 0,
          VK_INDEX_TYPE_UINT16);  // VK_INDEX_TYPE_UINT16 because we are using
                                  // uint_16

      vkCmdBindDescriptorSets(commandBuffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                              0, 1, &descriptorSets[i], 0, nullptr);
      vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()),
                       1, 0, 0, 0);

      vkCmdEndRenderPass(commandBuffers[i]);

      if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }

    LOG("Vulkan Command Buffer Creation Successful");
  }

  void createSyncObjects() {
    LOG("Vulkan Sync Objects Creation Started");

    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                            &imageAvailableSemaphores[i]) != VK_SUCCESS ||
          vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                            &renderFinishedSemaphores[i]) != VK_SUCCESS ||
          vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) !=
              VK_SUCCESS) {
        throw std::runtime_error("failed to create semaphores for a frame!");
      }
    }

    LOG("Vulkan Sync Objects Creation Successful");
  }

  // Get suitable memory type from physical device
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    /*
    However, we're not just interested in a memory type that is suitable for the
    vertex buffer. We also need to be able to write our vertex data to that
    memory. The memoryTypes array consists of VkMemoryType structs that specify
    the heap and properties of each type of memory. The properties define
    special features of the memory, like being able to map it so we can write to
    it from the CPU. This property is indicated with
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, but we also need to use the
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT property. We'll see why when we map the
    memory.
    */
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }

  void createImage(uint32_t width, uint32_t height, VkFormat format,
                   VkImageTiling tiling, VkImageUsageFlags usage,
                   VkMemoryPropertyFlags properties, VkImage &image,
                   VkDeviceMemory &imageMemory) {
    LOG("Vulkan Image Creation Started");
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                     indices.transferFamily.value()};

    if (indices.graphicsFamily != indices.transferFamily) {
      imageInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
      imageInfo.queueFamilyIndexCount = 2;
      imageInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      imageInfo.queueFamilyIndexCount = 0;      // Optional
      imageInfo.pQueueFamilyIndices = nullptr;  // Optional
    }

    // imageInfo.sharingMode =
    //    VK_SHARING_MODE_EXCLUSIVE;  // TODO: Check if this works with the
    // transfer queue

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
      throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
    LOG("Vulkan Image Creation Successful");
  }

  void transitionImageLayout(VkImage image, VkFormat format,
                             VkImageLayout oldLayout, VkImageLayout newLayout) {
    CommandBufferFamily commandBufferFamily;

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;

    // If you are using the barrier to transfer queue family ownership, then
    // these two fields should be the indices of the queue families. They must
    // be set to VK_QUEUE_FAMILY_IGNORED if you don't want to do this (not the
    // default value!).
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      commandBufferFamily = beginSingleTimeCommands(transferCommandPool);

      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

      sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      commandBufferFamily = beginSingleTimeCommands(graphicsCommandPool);

      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

      VkMemoryBarrier memoryBarrier = {};
      memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memoryBarrier.srcAccessMask = barrier.srcAccessMask;
      memoryBarrier.dstAccessMask = barrier.dstAccessMask;

      vkCmdPipelineBarrier(commandBufferFamily.commandBuffer,
                           sourceStage,       // srcStageMask
                           destinationStage,  // dstStageMask
                           0,                 // dependency flags
                           1,                 // memoryBarrierCount
                           &memoryBarrier,    // pMemoryBarriers
                           0,                 // Buffer Barrier Count
                           nullptr,           // Buffer Barrier
                           0,                 // Image Barrier Count
                           nullptr            // Image Barrier
      );

    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(commandBufferFamily.commandBuffer, sourceStage,
                         destinationStage, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);

    endSingleTimeCommands(commandBufferFamily);
  }

  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
                         uint32_t height) {
    CommandBufferFamily commandBufferFamily =
        beginSingleTimeCommands(transferCommandPool);

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(commandBufferFamily.commandBuffer, buffer, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBufferFamily);
  }

  void createTextureImage() {
    LOG("Vulkan Texture Loading Started");
    int texWidth, texHeight, texChannels;
    stbi_uc *pixels =
        stbi_load(utils::fixRelativeAssetPath("textures/texture.jpg").c_str(),
                  &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBufferMemory);

    stbi_image_free(pixels);

    createImage(
        texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, textureImage,
                      static_cast<uint32_t>(texWidth),
                      static_cast<uint32_t>(texHeight));
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    LOG("Vulkan Texture Loading Successful");
  }

  void createTextureImageView() {
    LOG("Vulkan Texture Image View Creation Started");
    textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM);
    LOG("Vulkan Texture Image View Creation Successful");
  }

  void createTextureSampler() {
    LOG("Vulkan Texture Sampler Creation Started");
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // The magFilter and minFilter fields specify how to interpolate texels that
    // are magnified or minified. Magnification concerns the oversampling
    // problem describes above, and minification concerns undersampling. The
    // choices are VK_FILTER_NEAREST and VK_FILTER_LINEAR, corresponding to the
    // modes demonstrated in the images above.
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;

    /* The addressing mode can be specified per axis using the addressMode
    fields. The available values are listed below. Most of these are
    demonstrated in the image above. Note that the axes are called U, V and W
    instead of X, Y and Z. This is a convention for texture space coordinates.

    VK_SAMPLER_ADDRESS_MODE_REPEAT: Repeat the texture when going beyond the
    image dimensions. VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: Like repeat, but
    inverts the coordinates to mirror the image when going beyond the
    dimensions. VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: Take the color of the
    edge closest to the coordinate beyond the image dimensions.
    VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: Like clamp to edge, but
    instead uses the edge opposite to the closest edge.
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: Return a solid color when sampling
    beyond the dimensions of the image. */
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);
    if (supportedFeatures.samplerAnisotropy) {
      samplerInfo.anisotropyEnable = VK_TRUE;
      samplerInfo.maxAnisotropy = 16;
    } else {
      samplerInfo.anisotropyEnable = VK_FALSE;
      samplerInfo.maxAnisotropy = 1;
    }

    /* The unnormalizedCoordinates field specifies which coordinate system you
     * want to use to address texels in an image. If this field is VK_TRUE, then
     * you can simply use coordinates within the [0, texWidth) and [0,
     * texHeight) range. If it is VK_FALSE, then the texels are addressed using
     * the [0, 1) range on all axes. Real-world applications almost always use
     * normalized coordinates, because then it's possible to use textures of
     * varying resolutions with the exact same coordinates. */
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

    /* If a comparison function is enabled, then texels will first be compared
     * to a value, and the result of that comparison is used in filtering
     * operations. This is mainly used for percentage-closer filtering on shadow
     * maps. */
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create texture sampler!");
    }

    LOG("Vulkan Texture Sampler Creation Successful");
  }

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                     indices.transferFamily.value()};

    if (indices.graphicsFamily != indices.transferFamily) {
      bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
      bufferInfo.queueFamilyIndexCount = 2;
      bufferInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      bufferInfo.queueFamilyIndexCount = 0;      // Optional
      bufferInfo.pQueueFamilyIndices = nullptr;  // Optional
    }

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
  }

  CommandBufferFamily beginSingleTimeCommands(
      CommandPoolFamily &commandPoolFamily) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    // allocInfo.commandPool = graphicsCommandPool.commandPool;
    // allocInfo.commandPool = transferCommandPool.commandPool;
    allocInfo.commandPool = commandPoolFamily.commandPool;
    allocInfo.commandBufferCount = 1;

    CommandBufferFamily commandBufferFamily;
    commandBufferFamily.commandPoolFamily = &commandPoolFamily;
    vkAllocateCommandBuffers(device, &allocInfo,
                             &commandBufferFamily.commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBufferFamily.commandBuffer, &beginInfo);

    return commandBufferFamily;
  }

  void endSingleTimeCommands(CommandBufferFamily &commandBufferFamily) {
    vkEndCommandBuffer(commandBufferFamily.commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBufferFamily.commandBuffer;

    vkQueueSubmit(commandBufferFamily.commandPoolFamily->queue, 1, &submitInfo,
                  VK_NULL_HANDLE);
    vkQueueWaitIdle(commandBufferFamily.commandPoolFamily->queue);

    vkFreeCommandBuffers(device,
                         commandBufferFamily.commandPoolFamily->commandPool, 1,
                         &commandBufferFamily.commandBuffer);
  }

  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    CommandBufferFamily commandBufferFamily =
        beginSingleTimeCommands(transferCommandPool);

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;  // Optional
    copyRegion.dstOffset = 0;  // Optional
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBufferFamily.commandBuffer, srcBuffer, dstBuffer, 1,
                    &copyRegion);

    endSingleTimeCommands(commandBufferFamily);
  }

  void createVertexBuffer() {
    LOG("Vulkan Vertex Buffer Creation Started");
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    LOG("Vulkan Vertex Buffer Creation Successful");
  }

  void createIndexBuffer() {
    LOG("Vulkan Index Buffer Creation Started");
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
    LOG("Vulkan Index Buffer Creation Successful");
  }

  void createUniformBuffers() {
    LOG("Vulkan Uniform Buffer Creation Started");

    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(swapChainImages.size());
    uniformBuffersMemory.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   uniformBuffers[i], uniformBuffersMemory[i]);
    }

    LOG("Vulkan Uniform Buffer Creation Successful");
  }

  void createDescriptorPool() {
    LOG("Vulkan Description Pool Creation Started");

    std::array<VkDescriptorPoolSize, 2> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount =
        static_cast<uint32_t>(swapChainImages.size());
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount =
        static_cast<uint32_t>(swapChainImages.size());

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor pool!");
    }

    LOG("Vulkan Description Pool Creation Successful");
  }

  void createDescriptorSets() {
    LOG("Vulkan Uniform Description Set Creation Started");

    std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(),
                                               descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount =
        static_cast<uint32_t>(swapChainImages.size());
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(swapChainImages.size());
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      VkDescriptorBufferInfo bufferInfo = {};
      bufferInfo.buffer = uniformBuffers[i];
      bufferInfo.offset = 0;
      bufferInfo.range = sizeof(UniformBufferObject);

      VkDescriptorImageInfo imageInfo = {};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.imageView = textureImageView;
      imageInfo.sampler = textureSampler;

      std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

      descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[0].dstSet = descriptorSets[i];
      descriptorWrites[0].dstBinding = 0;
      descriptorWrites[0].dstArrayElement = 0;
      descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptorWrites[0].descriptorCount = 1;
      descriptorWrites[0].pBufferInfo = &bufferInfo;

      descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[1].dstSet = descriptorSets[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].dstArrayElement = 0;
      descriptorWrites[1].descriptorType =
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrites[1].descriptorCount = 1;
      descriptorWrites[1].pImageInfo = &imageInfo;

      vkUpdateDescriptorSets(device,
                             static_cast<uint32_t>(descriptorWrites.size()),
                             descriptorWrites.data(), 0, nullptr);
    }

    LOG("Vulkan Uniform Description Set Creation Successful");
  }

  void cleanupSwapChain() {
    LOG("Vulkan SwapChain Cleanup Started");
    for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
      vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
    }

    vkFreeCommandBuffers(device, graphicsCommandPool.commandPool,
                         static_cast<uint32_t>(commandBuffers.size()),
                         commandBuffers.data());

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      vkDestroyImageView(device, swapChainImageViews[i], nullptr);
    }

    vkDestroySwapchainKHR(device, swapChain, nullptr);

    LOG("Vulkan SwapChain Cleanup Successful");
  }

  void recreateSwapChain() {
    LOG("Vulkan SwapChain Recreation Started");

    // Handle Window Minimization by busy wait until we are out of the
    // background again
    int width = 0, height = 0;
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }

    vkDeviceWaitIdle(device);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandBuffers();

    LOG("Vulkan SwapChain Recreation Successful");
  }

  void initVulkan() {
    LOG("Vulkan Init Started");

    createInstance();
    setupDebugCallback();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();

    LOG("Vulkan Init Successful");
  }

  void mainLoop() {
    LOG("Main Loop Started");
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }
    vkDeviceWaitIdle(device);
    LOG("Main Loop Finished");
  }

  void drawFrame() {
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                    std::numeric_limits<uint64_t>::max());

    uint32_t imageIndex;

    // Obtain the array of presentable images associated with a swapchain
    VkResult result = vkAcquireNextImageKHR(
        device, swapChain, std::numeric_limits<uint64_t>::max(),
        imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    // Recreate the swapchain if necessary i.e Window Resize
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      framebufferResized = false;
      recreateSwapChain();
      return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      // Check
      throw std::runtime_error("failed to acquire swap chain image!");
    }

    updateUniformBuffer(imageIndex);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                      inFlightFences[currentFrame]) != VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    presentInfo.pResults = nullptr;  // Optional

    result = vkQueuePresentKHR(presentQueue, &presentInfo);

    // Recreate the swapchain if necessary i.e Window Resize or Suboptimal
    // Swapchain
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      recreateSwapChain();
    } else if (result != VK_SUCCESS) {
      throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();

    UniformBufferObject ubo = {};
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(
        glm::radians(45.0f),
        swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;  // In Vulkan the Y axis is upside down compared to
                           // OpenGL, so We Fix it

    void *data;
    vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0,
                &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
  }

  void cleanup() {
    LOG("Cleanup Started");

    cleanupSwapChain();

    vkDestroySampler(device, textureSampler, nullptr);

    vkDestroyImageView(device, textureImageView, nullptr);

    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      vkDestroyBuffer(device, uniformBuffers[i], nullptr);
      vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }

    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);

    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
      vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
      vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, graphicsCommandPool.commandPool, nullptr);
    vkDestroyCommandPool(device, transferCommandPool.commandPool, nullptr);

    vkDestroyDevice(device, nullptr);

    if (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
    }

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();

    LOG("Cleanup Successful");
  }
};

int main(int argc, char *argv[]) {
  LONGLOG("NDEBUG MACRO NOT DEFINED");

  LONGLOG("LONG "
          << "LOG Test");
  LOG("LOG"
      << " test");

  if (argc < 1) {
    throw std::runtime_error("failed get runtime location!");
  }
  utils::setRuntimeFolder(argv[0]);

  FORLOG(int i = 0; i < 2; i++, "\t " << i);
  FORLONGLOG(int i = 0; i < 2; i++, "\t " << i);

  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}