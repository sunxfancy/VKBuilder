#define VKB_IMPL
#include <cstdio>
#include <stdexcept>
#include "vkbuilder.hpp"

// Real-device regression: no window, no engine, and no exception-to-skip conversion.
// On the iOS simulator maxFramebufferLayers == 1, even though sampling arrays works.
int main() {
    try {
        vkb::InstanceBuilder builder;
        auto                 system = vkb::SystemInfo::query();
        if (system.is_extension_available(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            builder.enable_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            builder.add_flags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
        }
        auto instance = builder.set_headless().require_api_version(1, 1).build();
        auto physical = vkb::PhysicalDeviceSelector(instance)
                            .defer_surface_initialization()
                            .set_minimum_version(1, 1)
                            .add_desired_extension("VK_KHR_portability_subset")
                            .select();
        auto device = vkb::DeviceBuilder(physical).build();
        {
            vkb::DepthArrayImage depth(device, 16, 16, 3);
            if (!depth.arrayView() || depth.layerCount() != 3)
                throw std::runtime_error("missing depth array sampling view");
            vk::AttachmentDescription attachment{};
            attachment.format        = vk::Format::eD32Sfloat;
            attachment.samples       = vk::SampleCountFlagBits::e1;
            attachment.loadOp        = vk::AttachmentLoadOp::eClear;
            attachment.storeOp       = vk::AttachmentStoreOp::eStore;
            attachment.initialLayout = vk::ImageLayout::eUndefined;
            attachment.finalLayout   = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
            vk::AttachmentReference reference(0, vk::ImageLayout::eDepthStencilAttachmentOptimal);
            vk::SubpassDescription  subpass{};
            subpass.pipelineBindPoint       = vk::PipelineBindPoint::eGraphics;
            subpass.pDepthStencilAttachment = &reference;
            vk::RenderPassCreateInfo passInfo{};
            passInfo.attachmentCount = 1;
            passInfo.pAttachments    = &attachment;
            passInfo.subpassCount    = 1;
            passInfo.pSubpasses      = &subpass;
            auto pass                = device->createRenderPassUnique(passInfo);
            for (uint32_t layer = 0; layer < 3; ++layer) {
                auto                      view = depth.layerView(layer);
                vk::FramebufferCreateInfo framebuffer{};
                framebuffer.renderPass      = *pass;
                framebuffer.attachmentCount = 1;
                framebuffer.pAttachments    = &view;
                framebuffer.width = framebuffer.height = 16;
                framebuffer.layers                     = 1;
                auto target                            = device->createFramebufferUnique(framebuffer);
                if (!target) throw std::runtime_error("missing per-layer attachment framebuffer");
            }
            std::printf("depth array: sampling view + 3 single-layer framebuffers OK (maxFramebufferLayers=%u)\n",
                        physical.instance.getProperties().limits.maxFramebufferLayers);
        }
        device.destroy();
        instance.instance.destroy();
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "image view regression FAILED: %s\n", error.what());
        return 1;
    }
}
