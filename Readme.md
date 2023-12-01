VkBuilder 便捷的vulkan对象构造器
================================


我们在使用vulkan时，往往被繁琐的概念和大量的参数设置困扰，这个库的主要目的是创建一套能快速构建Vulkan对象的构造器，使用这些builder可以快速创建常用的，带有良好设计默认参数的，带有常用配置方案的对象。从而可以快速创建可用的渲染流水线。

我们对Instance，Device等对象都进行了包装，将其和其他常用对象打包，使得很多API调用时可以减少参数的使用，尤其是减少参数的传入错误


设计思路：
1. 先提供常用的参数，允许默认构建性能一般但可用的对象，提供接口对构造时进行特化指定
2. 大量使用流式API
3. 通过打包其他对象，尽量减少接口调用时的参数
4. 明确哪些对象和其他对象的约束关系，在接口中体现这种约束关系，尽量使得不符合约束关系的调用无法被编译
5. 使用模板减少代码量


## 使用方法


使用VKBuilder可以快速创建实例，设备，渲染界面，交换链等等Vulkan对象。

```cpp
#define VKB_IMPL
#include "vkbuilder.hpp"

class Render {
public:
  void init(void *window) {

    // 首先创建构造器，选择API版本，需要的layer，配置debug messenger
    vkb::InstanceBuilder builder;
    inst = builder.require_api_version(1, 0)
                  .request_validation_layers()   // validate correctness
                  .use_default_debug_messenger() // use DebugUtilsMessage
                  .build();

    // 创建surface用于渲染
    auto surface = (VkSurfaceKHR)create_surface_glfw(inst.instance, window);

    // 创建一个PhysicalDeviceSelector用来选择渲染设备
    vkb::PhysicalDeviceSelector selector{inst};

    auto phys = selector.set_surface(surface)
            .set_minimum_version(1, 0) // require a vulkan 1.0 capable device
            .select();

    vkb::DeviceBuilder device_builder{phys};

    // 构造设备
    // automatically propagate needed data from instance & physical device
    device = device_builder.build();

    create_swapchain();
    create_pipeline();
  }

  void create_swapchain() {
    // 创建默认交换链
    vkb::SwapchainBuilder swapchain_builder{device};
    auto swap_ret = swapchain_builder.set_old_swapchain(swapchain).build();
    swapchain.destroy();
    swapchain = swap_ret;
  }

  void create_pipeline() {
    // 创建渲染Pass
    vkb::RenderPassBuilder renderpass_builder{device};
    renderpass = renderpass_builder
            .addPresentAttachment(swapchain.image_format, vk::AttachmentLoadOp::eClear)
            .addSubpass(vkb::SubpassBuilder()
              .addAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal))
            .addDependency(VK_SUBPASS_EXTERNAL, 0)
            .build();

    auto vert_code = readFile("vert.spv");
    auto frag_code = readFile("frag.spv");

    // 创建渲染流水线
    vkb::PipelineBuilder pipeline_builder{device, swapchain};
    pipeline = pipeline_builder
        .useClassicPipeline(vert_code, frag_code)
        .setVertexInputState(
          vkb::VertexInputStateBuilder()  // VertexInputStateBuilder 可以根据Vertex上的函数进行创建
            .addInputBinding<Vertex>()
            .addAttributeDescription<Vertex>()
        )
        .build(renderpass);

    vkb::PresentBuilder present_builder{device, swapchain};
    present = present_builder.build(renderpass);

    initVertex();
  }

  vkb::Instance inst;
  vkb::Device device;
  vkb::Swapchain swapchain;

  vk::RenderPass renderpass;
  vk::Pipeline pipeline;

  vkb::Present present;

  std::vector<Vertex> v;
  vkb::HostVertexBuffer buffer;
};

Render render;

void onRender() {
  render.render();
}

void initVulkan(void *window) {
  render.init(window);
}

int main() {
  GLFW_Window window;
  window.createWindow();
  initVulkan(window.getWindow());
  window.mainLoop();
  return 0;
}
```

对于顶点数据结构，只需要创建`getBindingDescription`函数与`getAttributeDescription`函数，就可以将该结构体注册到我们的系统中来。
```cpp

  struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    
    static vk::VertexInputBindingDescription 
    getBindingDescription(uint32_t binding) {
      vk::VertexInputBindingDescription bindingDescription{};
      bindingDescription.binding = binding;
      bindingDescription.stride = sizeof(Vertex);
      bindingDescription.inputRate = vk::VertexInputRate::eVertex;
      return bindingDescription;
    }
    static std::vector<vk::VertexInputAttributeDescription>
    getAttributeDescription(uint32_t binding) {
      std::vector<vk::VertexInputAttributeDescription> attrs = {
        vk::VertexInputAttributeDescription(0, binding, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)),
        vk::VertexInputAttributeDescription(1, binding, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color))
      };
      return attrs;
    }
  };

  void initVertex() {
    v.resize(3);
    v[0].pos = {0, 0.5}; v[0].color = {1,0,0};
    v[1].pos = {-0.5, -0.5}; v[1].color = {0,1,0};
    v[2].pos = {0.5, -0.5}; v[2].color = {0,0,1};

    // create Vertex buffer
    buffer.allocate<Vertex>(device, v);
  }
```

渲染控制也非常轻松：
```cpp
  void render() {
    present.begin();
    present.beginRenderPass(renderpass);

    vk::DeviceSize offset(0);
    auto& cb = present.getCurrentCommandBuffer();
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    cb.bindVertexBuffers(0, 1, buffer, &offset);
    cb.draw(3, 1, 0, 0);

    present.endRenderPass();
    present.end();
    present.drawFrame();
  }
```


