#pragma once
#include "vkbuilder.hpp"
#include <limits>
#include <map>
#include <sstream>

namespace vkb {

/** @brief One actionable shader/host binding mismatch. */
struct BindingDiagnostic {
  uint32_t set = 0, binding = 0;
  vk::ShaderStageFlags stages{};
  std::string expected, actual;
};

/** @brief Owning validation result; no GPU state is changed on failure. */
struct BindingValidation {
  std::vector<BindingDiagnostic> diagnostics;
  explicit operator bool() const { return diagnostics.empty(); }
  void fail(uint32_t set, uint32_t binding, vk::ShaderStageFlags stages,
            std::string expected, std::string actual) {
    diagnostics.push_back({set, binding, stages, std::move(expected), std::move(actual)});
  }
};

enum class ShaderNumericType { Float, Sint, Uint, Unknown };
inline const char* shaderNumericName(ShaderNumericType type) {
  switch (type) {
    case ShaderNumericType::Float: return "float";
    case ShaderNumericType::Sint: return "signed integer";
    case ShaderNumericType::Uint: return "unsigned integer";
    default: return "unknown";
  }
}

/** @brief Reflected descriptor ABI. blockLayout describes types, offsets and strides,
 * not source member names. Empty means the descriptor is not a buffer. */
struct ShaderBinding {
  uint32_t set = 0, binding = 0, count = 1;
  vk::DescriptorType type = vk::DescriptorType::eUniformBuffer;
  vk::ShaderStageFlags stages{};
  vk::DeviceSize minimumSize = 0;
  std::string blockLayout;
  vk::ImageViewType viewType = vk::ImageViewType::e2D;
  ShaderNumericType numeric = ShaderNumericType::Float;
  bool multisampled = false, depthComparison = false;
};

/** @brief Shader interface cached once per program. Merge is transactional. */
class ShaderBindingContract {
public:
  const std::vector<ShaderBinding>& bindings() const { return bindings_; }
  BindingValidation merge(const std::vector<ShaderBinding>& incoming) {
    BindingValidation result;
    auto next = bindings_;
    for (const auto& item : incoming) {
      auto old = std::find_if(next.begin(), next.end(), [&](const ShaderBinding& b) {
        return b.set == item.set && b.binding == item.binding;
      });
      if (item.count == 0 || !item.stages) {
        result.fail(item.set, item.binding, item.stages, "fixed descriptor count and stage", "empty count/stage");
      } else if (old == next.end()) {
        next.push_back(item);
      } else if (!compatible(*old, item)) {
        result.fail(item.set, item.binding, old->stages | item.stages,
                    describe(*old), describe(item));
      } else {
        old->stages |= item.stages;
      }
    }
    if (result) bindings_ = std::move(next);
    return result;
  }

  /** @brief Check host layout against all used descriptors in one set.
   * Extra unused host bindings are permitted by Vulkan. */
  BindingValidation validateLayout(uint32_t set,
      const std::vector<vk::DescriptorSetLayoutBinding>& layout) const {
    BindingValidation result;
    std::map<uint32_t, bool> seen;
    for (const auto& host : layout) {
      if (seen.count(host.binding) || host.descriptorCount == 0)
        result.fail(set, host.binding, host.stageFlags, "unique nonempty layout binding", "duplicate/empty binding");
      seen[host.binding] = true;
    }
    for (const auto& b : bindings_) {
      if (b.set != set) continue;
      auto h = std::find_if(layout.begin(), layout.end(), [&](const auto& x) { return x.binding == b.binding; });
      if (h == layout.end()) {
        result.fail(set, b.binding, b.stages, "layout binding", "missing");
      } else if (h->descriptorType != b.type || h->descriptorCount < b.count ||
                 (h->stageFlags & b.stages) != b.stages) {
        result.fail(set, b.binding, b.stages, vk::to_string(b.type) + " count >= " + std::to_string(b.count),
                    vk::to_string(h->descriptorType) + " count=" + std::to_string(h->descriptorCount) +
                    " stages=" + vk::to_string(h->stageFlags));
      }
    }
    return result;
  }
private:
  static std::string describe(const ShaderBinding& b) {
    return vk::to_string(b.type) + " count=" + std::to_string(b.count) +
      " minimumBytes=" + std::to_string(b.minimumSize) + " layout=" + b.blockLayout +
      " view=" + vk::to_string(b.viewType) + " numeric=" + shaderNumericName(b.numeric) +
      " MS=" + std::to_string(b.multisampled) + " compare=" + std::to_string(b.depthComparison);
  }
  static bool compatible(const ShaderBinding& a, const ShaderBinding& b) {
    return a.type == b.type && a.count == b.count && a.minimumSize == b.minimumSize &&
      a.blockLayout == b.blockLayout && a.viewType == b.viewType && a.numeric == b.numeric &&
      a.multisampled == b.multisampled && a.depthComparison == b.depthComparison;
  }
  std::vector<ShaderBinding> bindings_;
};

/** @brief Creation metadata for an externally owned view and sampler.
 * Copy from the actual image/view/sampler creation records, never from the shader.
 * Vulkan handles alone cannot reveal these properties. */
struct BindingImageInfo {
  vk::ImageViewType viewType = vk::ImageViewType::e2D;
  vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
  vk::ImageUsageFlags usage{};
  vk::Format format = vk::Format::eUndefined;
  bool compareEnable = false;
  vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;

  /** @brief Derive metadata from the actual creation records (view format wins). */
  static BindingImageInfo from(const vk::ImageCreateInfo& image,
      const vk::ImageViewCreateInfo& view, const vk::SamplerCreateInfo& sampler) {
    BindingImageInfo result;
    result.viewType = view.viewType; result.samples = image.samples;
    result.usage = image.usage; result.format = view.format;
    result.compareEnable = sampler.compareEnable != 0; result.aspect = view.subresourceRange.aspectMask;
    return result;
  }
  ShaderNumericType numericType() const {
    if (aspect == vk::ImageAspectFlagBits::eStencil) return ShaderNumericType::Uint;
    const auto name = vk::to_string(format);
    if (name.find("Uint") != std::string::npos) return ShaderNumericType::Uint;
    if (name.find("Sint") != std::string::npos) return ShaderNumericType::Sint;
    if (name.find("Unorm") != std::string::npos || name.find("Snorm") != std::string::npos ||
        name.find("Sfloat") != std::string::npos || name.find("Ufloat") != std::string::npos ||
        name.find("Srgb") != std::string::npos) return ShaderNumericType::Float;
    return ShaderNumericType::Unknown;
  }

};

/** @brief Checked, complete descriptor-set writes. Owns metadata and borrows handles
 * until update(); caller retains resources through GPU use and synchronizes writes.
 * Supports fixed arrays of ordinary UBO/SSBO and combined sampled images. Dynamic
 * descriptors, copies, partial updates, and descriptor indexing are intentionally rejected. */
class CheckedDescriptorSetBuilder {
public:
  CheckedDescriptorSetBuilder(const ShaderBindingContract& contract, uint32_t set,
      const UnboundSet& destination, const vk::PhysicalDeviceLimits& limits)
      : contract_(contract), set_(set), destination_(destination.handle), limits_(limits) {}

  /** @brief Bind an actual VKBuilder buffer. Metadata is copied, no pointer is retained.
   * blockLayout is optional host layout evidence; empty validates range only. */
  CheckedDescriptorSetBuilder& buffer(uint32_t binding, const GenericBuffer& buffer,
      vk::DeviceSize offset = 0, vk::DeviceSize range = VK_WHOLE_SIZE,
      uint32_t element = 0, std::string blockLayout = {}) {
    Entry e;
    e.binding = binding; e.element = element; e.buffer = true;
    e.bufferInfo = vk::DescriptorBufferInfo{buffer.buffer, offset, range};
    e.size = buffer.size; e.usage = buffer.usage_flags; e.blockLayout = std::move(blockLayout);
    entries_.push_back(std::move(e));
    return *this;
  }

  /** @brief Bind a combined sampler with metadata from resource creation. */
  CheckedDescriptorSetBuilder& image(uint32_t binding, const SampledImage& image,
      const BindingImageInfo& info, uint32_t element = 0) {
    Entry e;
    e.binding = binding; e.element = element; e.imageInfo = vk::DescriptorImageInfo{image.sampler, image.view, image.layout};
    e.image = info;
    entries_.push_back(std::move(e));
    return *this;
  }

  /** @brief CPU-only validation, available in Debug and Release. */
  BindingValidation validate() const {
    BindingValidation result;
    if (!destination_) result.fail(set_, 0, {}, "non-null descriptor set", "null");
    std::map<std::pair<uint32_t, uint32_t>, bool> seen;
    for (const auto& e : entries_) {
      const auto key = std::make_pair(e.binding, e.element);
      if (seen.count(key)) result.fail(set_, e.binding, {}, "one write per array element", "duplicate");
      seen[key] = true;
      const auto* b = find(e.binding);
      if (!b) { result.fail(set_, e.binding, {}, "shader binding", "undeclared binding"); continue; }
      if (e.element >= b->count) result.fail(set_, e.binding, b->stages, "array element < " + std::to_string(b->count), std::to_string(e.element));
      if (e.buffer) validateBuffer(*b, e, result);
      else validateImage(*b, e, result);
    }
    for (const auto& b : contract_.bindings()) {
      if (b.set != set_) continue;
      // Count unique in-range writes instead of iterating potentially huge arrays.
      uint32_t present = 0;
      for (const auto& item : seen)
        if (item.first.first == b.binding && item.first.second < b.count) ++present;
      if (present != b.count) result.fail(set_, b.binding, b.stages,
          std::to_string(b.count) + " array elements", std::to_string(present) + " supplied");
    }
    return result;
  }

  /** @brief Validate all writes before touching the device; failures preserve the set.
   * Destination must have a layout accepted by contract.validateLayout(). */
  BindingValidation update(const vk::Device& device) const {
    auto result = validate();
    if (!result) return result;
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(entries_.size());
    for (const auto& e : entries_) {
      vk::WriteDescriptorSet write;
      write.dstSet = destination_; write.dstBinding = e.binding; write.dstArrayElement = e.element;
      write.descriptorCount = 1; write.descriptorType = find(e.binding)->type;
      if (e.buffer) write.pBufferInfo = &e.bufferInfo;
      else write.pImageInfo = &e.imageInfo;
      writes.push_back(write);
    }
    device.updateDescriptorSets(writes, {});
    return result;
  }
private:
  struct Entry {
    uint32_t binding = 0, element = 0;
    bool buffer = false;
    vk::DescriptorBufferInfo bufferInfo{};
    vk::DescriptorImageInfo imageInfo{};
    vk::DeviceSize size = 0;
    vk::BufferUsageFlags usage{};
    std::string blockLayout;
    BindingImageInfo image;
  };
  const ShaderBinding* find(uint32_t binding) const {
    for (const auto& b : contract_.bindings()) if (b.set == set_ && b.binding == binding) return &b;
    return nullptr;
  }
  void validateBuffer(const ShaderBinding& b, const Entry& e, BindingValidation& out) const {
    const bool uniform = b.type == vk::DescriptorType::eUniformBuffer;
    if (!uniform && b.type != vk::DescriptorType::eStorageBuffer) {
      out.fail(set_, b.binding, b.stages, vk::to_string(b.type), "buffer"); return;
    }
    const auto usage = uniform ? vk::BufferUsageFlagBits::eUniformBuffer : vk::BufferUsageFlagBits::eStorageBuffer;
    const auto alignment = uniform ? limits_.minUniformBufferOffsetAlignment : limits_.minStorageBufferOffsetAlignment;
    const auto maximum = uniform ? limits_.maxUniformBufferRange : limits_.maxStorageBufferRange;
    const auto offset = e.bufferInfo.offset;
    if (!e.bufferInfo.buffer || !(e.usage & usage) || offset >= e.size ||
        (alignment && offset % alignment)) {
      out.fail(set_, b.binding, b.stages, "live buffer with correct usage, aligned in-bounds offset", "invalid handle/usage/offset"); return;
    }
    const auto range = e.bufferInfo.range == VK_WHOLE_SIZE ? e.size - offset : e.bufferInfo.range;
    if (range == 0 || range > e.size - offset || range < b.minimumSize || range > maximum)
      out.fail(set_, b.binding, b.stages, "buffer range >= " + std::to_string(b.minimumSize) + " within allocation/device limit", std::to_string(range));
    if (!e.blockLayout.empty() && e.blockLayout != b.blockLayout)
      out.fail(set_, b.binding, b.stages, b.blockLayout, e.blockLayout);
  }
  void validateImage(const ShaderBinding& b, const Entry& e, BindingValidation& out) const {
    if (b.type != vk::DescriptorType::eCombinedImageSampler) {
      out.fail(set_, b.binding, b.stages, vk::to_string(b.type), "combined image sampler"); return;
    }
    if (e.image.format == vk::Format::eUndefined || !e.image.aspect ||
        (e.image.aspect != vk::ImageAspectFlagBits::eColor && e.image.aspect != vk::ImageAspectFlagBits::eDepth &&
         e.image.aspect != vk::ImageAspectFlagBits::eStencil) ||
        !e.imageInfo.imageView || !e.imageInfo.sampler || !(e.image.usage & vk::ImageUsageFlagBits::eSampled) ||
        (e.imageInfo.imageLayout != vk::ImageLayout::eShaderReadOnlyOptimal && e.imageInfo.imageLayout != vk::ImageLayout::eGeneral &&
         e.imageInfo.imageLayout != vk::ImageLayout::eDepthStencilReadOnlyOptimal))
      out.fail(set_, b.binding, b.stages, "sampled view/sampler and readable layout", "invalid handle/usage/layout");
    if (e.image.viewType != b.viewType || e.image.numericType() != b.numeric ||
        (e.image.samples != vk::SampleCountFlagBits::e1) != b.multisampled ||
        e.image.compareEnable != b.depthComparison ||
        (b.depthComparison && e.image.aspect != vk::ImageAspectFlagBits::eDepth))
      out.fail(set_, b.binding, b.stages,
          vk::to_string(b.viewType) + " numeric=" + shaderNumericName(b.numeric) +
          " MS=" + std::to_string(b.multisampled) + " compare=" + std::to_string(b.depthComparison),
          vk::to_string(e.image.viewType) + " numeric=" + shaderNumericName(e.image.numericType()) +
          " MS=" + std::to_string(e.image.samples != vk::SampleCountFlagBits::e1) + " compare=" + std::to_string(e.image.compareEnable));
  }
  ShaderBindingContract contract_;
  uint32_t set_;
  vk::DescriptorSet destination_;
  vk::PhysicalDeviceLimits limits_;
  std::vector<Entry> entries_;
};

} // namespace vkb
