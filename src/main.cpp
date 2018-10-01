#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_set>

#include <utils/debug/log.hpp>
#include <utils/files/binary_loader.hpp>
#include <utils/files/file_path.hpp>

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
#else  // !NDEBUG
const bool enableValidationLayers = true;
#endif // NDEBUG

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  bool isComplete() {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
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

  VkSwapchainKHR swapChain;
  std::vector<VkImage> swapChainImages;
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;

  std::vector<VkImageView> swapChainImageViews;

  VkRenderPass renderPass;

  VkPipelineLayout pipelineLayout;
  VkPipeline graphicsPipeline;

  std::vector<VkFramebuffer> swapChainFramebuffers;

  void initWindow() {
    LOG("Window Init Started");
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    LOG("Window Init Successful");
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
    if (!enableValidationLayers)
      return;

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
    createInfo.pUserData = nullptr; // Optional

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
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueCount > 0 &&
          queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphicsFamily = i;
      }

      presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

      if (queueFamily.queueCount > 0 && presentSupport) {
        indices.presentFamily = i;
      }

      if (indices.isComplete()) {
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
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      score += 1000;
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
        indices.graphicsFamily.value(), indices.presentFamily.value()};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo = {};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    // Required Device Features (I.e geomtry shader)
    VkPhysicalDeviceFeatures deviceFeatures = {};

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
      VkExtent2D actualExtent = {WIDTH, HEIGHT};

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
      createInfo.queueFamilyIndexCount = 0;     // Optional
      createInfo.pQueueFamilyIndices = nullptr; // Optional
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

  void createImageViews() {
    LOG("Vulkan Image View Creation Started");
    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      VkImageViewCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = swapChainImages[i];

      // The viewType and format fields specify how the image data should be
      // interpreted. The viewType parameter allows you to treat images as 1D
      // textures, 2D textures, 3D textures and cube maps.
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = swapChainImageFormat;

      // The components field allows you to swizzle the color channels around.
      // For example, you can map all of the channels to the red channel for a
      // monochrome texture. You can also map constant values of 0 and 1 to a
      // channel. In our case we'll stick to the default mapping.
      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

      // The subresourceRange field describes what the image's purpose is and
      // which part of the image should be accessed. Our images will be used as
      // color targets without any mipmapping levels or multiple layers.
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;

      if (vkCreateImageView(device, &createInfo, nullptr,
                            &swapChainImageViews[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image views!");
      }
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

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }

    LOG("Vulkan Render Pass Successful");
  }

  void createGraphicsPipeline() {
    LOG("Vulkan Graphics Pipeline Creation Started");
    auto vertShaderCode = utils::readFile("/shaders/triangle.vert.glsl.spv");
    auto fragShaderCode = utils::readFile("/shaders/triangle.frag.glsl.spv");

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
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

    // Vertex Data Description
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable =
        VK_FALSE; // Same vertex used by two triangles for example

    // Viewport size (normally all the window but it could be different)
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f; // Framebuffer depthbuffer
    viewport.maxDepth = 1.0f; // Framebuffer depthbuffer

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
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    // Depth Bias (Sometimes used with ShadowMapping)
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f;          // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f;    // Optional

    // MSAA (disabled for now)
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;          // Optional
    multisampling.pSampleMask = nullptr;            // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE;      // Optional

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
    colorBlendAttachment.blendEnable = VK_FALSE; // No blending
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;             // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;             // Optional

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
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional
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
    pipelineLayoutInfo.setLayoutCount = 0;            // Optional
    pipelineLayoutInfo.pSetLayouts = nullptr;         // Optional
    pipelineLayoutInfo.pushConstantRangeCount = 0;    // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

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
    pipelineInfo.pDepthStencilState = nullptr; // Optional
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr; // Optional

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
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1;              // Optional

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
    createGraphicsPipeline();
    createFramebuffers();

    LOG("Vulkan Init Successful");
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
    LOG("Cleanup Started");

    for (auto &framebuffer : swapChainFramebuffers) {
      vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    vkDestroyPipeline(device, graphicsPipeline, nullptr);

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

    vkDestroyRenderPass(device, renderPass, nullptr);

    for (auto &imageView : swapChainImageViews) {
      vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(device, swapChain, nullptr);

    vkDestroyDevice(device, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);

    if (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
    }

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