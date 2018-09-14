#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>

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
  void initVulkan() {
#ifdef _DEBUG
    std::cout << "Vulkan Init Started" << std::endl;
#endif  // _DEBUG
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

    // GLFW required Extensions
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;

    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

#ifdef _DEBUG
    std::cout << "GLFW Required Vulkan extensions:" << std::endl;

    for (int i = 0; i < glfwExtensionCount; i++) {
      std::cout << "\t" << glfwExtensions[i] << std::endl;
    }

    std::cout << "Checking GLFW Required Vulkan extensions:" << std::endl;
#endif  // _DEBUG

    std::vector<bool> glfwExtensionCheck(glfwExtensionCount, false);

    int extIter = 0, maxiter = 0;
    for (const auto &extension : extensions) {
      for (extIter = 0; extIter < glfwExtensionCount; extIter++) {
        if (!glfwExtensionCheck[extIter] &&
            strcmp(extension.extensionName, glfwExtensions[extIter]) == 0) {
          glfwExtensionCheck[extIter] = true;
          maxiter++;
#ifdef _DEBUG
          std::cout << "\tFound: " << extension.extensionName << std::endl;
#endif  // _DEBUG
          if (maxiter == glfwExtensionCount) {
#ifdef _DEBUG
            std::cout << "All GLFW Vulkan Extensions Found" << std::endl;
#endif  // _DEBUG
            break;
          }
        }
      }
    }

    if (maxiter != glfwExtensionCount) {
      throw std::runtime_error(
          "failed to get the required Vulkan extensions for GLFW!");
    }

    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;

    createInfo.enabledLayerCount = 0;
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
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}