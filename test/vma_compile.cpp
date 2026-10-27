#define VKB_ENABLE_VMA
#define VKB_VMA_HEADER "vk_mem_alloc.h"
#include <vkbuilder.hpp>

#include <type_traits>

static_assert(std::is_same_v<decltype(vkb::Device{}.vma_allocator),
                             VmaAllocator>);

void configureVma(vkb::Device &device, VmaAllocator allocator) {
  device.attachVmaAllocator(allocator);
}
