#pragma once
#include "binding_validation.hpp"
#include <spirv_cross/spirv_cross.hpp>

namespace vkb {
namespace binding_detail {
// SPIRV-Cross 1.3.239 (Ubuntu 24.04) predates this resource category.
template <class Resources>
inline auto hasPlainUniforms(const Resources& resources, int)
    -> decltype(resources.gl_plain_uniforms.empty(), bool()) {
  return !resources.gl_plain_uniforms.empty();
}
template <class Resources>
inline bool hasPlainUniforms(const Resources&, long) { return false; }

inline vk::ShaderStageFlagBits stage(spv::ExecutionModel model) {
  switch (model) {
    case spv::ExecutionModelVertex: return vk::ShaderStageFlagBits::eVertex;
    case spv::ExecutionModelFragment: return vk::ShaderStageFlagBits::eFragment;
    case spv::ExecutionModelGLCompute: return vk::ShaderStageFlagBits::eCompute;
    case spv::ExecutionModelGeometry: return vk::ShaderStageFlagBits::eGeometry;
    case spv::ExecutionModelTessellationControl: return vk::ShaderStageFlagBits::eTessellationControl;
    case spv::ExecutionModelTessellationEvaluation: return vk::ShaderStageFlagBits::eTessellationEvaluation;
    default: throw std::runtime_error("Unsupported shader execution model");
  }
}

// Structural serialization includes the actual SPIR-V offsets/strides. It is
// deliberately independent of debug names and SPIR-V result IDs.
inline std::string blockType(spirv_cross::Compiler& compiler, const spirv_cross::SPIRType& type) {
  std::ostringstream text;
  text << int(type.basetype) << ':' << type.width << ':' << type.vecsize << ':' << type.columns;
  for (size_t i = 0; i < type.array.size(); ++i) {
    if (!type.array_size_literal[i]) throw std::runtime_error("Specialized array sizes require a resolved shader variant");
    text << '[' << type.array[i] << ']';
  }
  if (type.basetype == spirv_cross::SPIRType::Struct) {
    text << '{';
    for (uint32_t i = 0; i < type.member_types.size(); ++i) {
      const auto& member = compiler.get_type(type.member_types[i]);
      text << compiler.type_struct_member_offset(type, i) << '=' << blockType(compiler, member);
      // ArrayStride decorates each array type, including nested arrays.
      auto id = type.member_types[i];
      for (;;) {
        const auto& arrayType = compiler.get_type(id);
        if (arrayType.array.empty()) break;
        if (!compiler.has_decoration(id, spv::DecorationArrayStride))
          throw std::runtime_error("Buffer array missing ArrayStride");
        text << "a" << compiler.get_decoration(id, spv::DecorationArrayStride);
        if (!arrayType.parent_type) break;
        id = arrayType.parent_type;
      }
      if (member.columns > 1)
        text << "m" << compiler.type_struct_member_matrix_stride(type, i) << "r"
             << compiler.has_member_decoration(type.self, i, spv::DecorationRowMajor);
      text << ';';
    }
    text << '}';
  }
  return text.str();
}

inline uint32_t count(const spirv_cross::SPIRType& type) {
  uint32_t count = 1;
  for (size_t i = 0; i < type.array.size(); ++i) {
    if (!type.array_size_literal[i] || type.array[i] == 0 ||
        type.array[i] > std::numeric_limits<uint32_t>::max() / count)
      throw std::runtime_error("Runtime/specialized/overflowing descriptor arrays are unsupported");
    count *= type.array[i];
  }
  return count;
}

inline vk::ImageViewType view(const spirv_cross::SPIRType& type) {
  switch (type.image.dim) {
    case spv::Dim1D: return type.image.arrayed ? vk::ImageViewType::e1DArray : vk::ImageViewType::e1D;
    case spv::Dim2D: return type.image.arrayed ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D;
    case spv::Dim3D:
      if (!type.image.arrayed) return vk::ImageViewType::e3D;
      break;
    case spv::DimCube: return type.image.arrayed ? vk::ImageViewType::eCubeArray : vk::ImageViewType::eCube;
    default: break;
  }
  throw std::runtime_error("Unsupported image dimension");
}
} // namespace binding_detail

/** @brief Reflect descriptors for one named entry point from the actual SPIR-V.
 * Requires SPIRV-Cross core. No Vulkan calls. On any unsupported declaration or
 * parse failure, the destination contract is unchanged. Does not validate SPIR-V
 * instructions, vertex input, push constants, or resource contents. */
inline BindingValidation reflectShaderBindings(ShaderBindingContract& destination,
    const std::vector<uint32_t>& words, vk::ShaderStageFlagBits stage,
    const std::string& entryPoint = "main") {
  BindingValidation result;
  uint32_t currentSet = 0, currentBinding = 0;
  try {
    if (words.size() < 5 || words[0] != 0x07230203u || words[4] != 0)
      throw std::runtime_error("Invalid SPIR-V header");
    spirv_cross::Compiler compiler(words);
    bool found = false;
    for (const auto& entry : compiler.get_entry_points_and_stages()) {
      if (entry.name == entryPoint && binding_detail::stage(entry.execution_model) == stage) {
        compiler.set_entry_point(entry.name, entry.execution_model); found = true; break;
      }
    }
    if (!found) throw std::runtime_error("Requested entry point/stage does not exist");
    const auto resources = compiler.get_shader_resources(compiler.get_active_interface_variables());
    if (!resources.storage_images.empty() || !resources.separate_images.empty() ||
        !resources.separate_samplers.empty() || !resources.subpass_inputs.empty() ||
        !resources.atomic_counters.empty() || !resources.acceleration_structures.empty() ||
        !resources.shader_record_buffers.empty() || binding_detail::hasPlainUniforms(resources, 0))
      throw std::runtime_error("Only UBO, SSBO and combined sampled-image descriptors are supported");
    std::vector<ShaderBinding> bindings;
    auto add = [&](const auto& list, vk::DescriptorType descriptorType) {
      for (const auto& resource : list) {
        if (!compiler.has_decoration(resource.id, spv::DecorationDescriptorSet) ||
            !compiler.has_decoration(resource.id, spv::DecorationBinding))
          throw std::runtime_error("Descriptor is missing set/binding decoration");
        ShaderBinding b;
        b.set = currentSet = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
        b.binding = currentBinding = compiler.get_decoration(resource.id, spv::DecorationBinding);
        b.stages = stage; b.type = descriptorType;
        const auto& type = compiler.get_type(resource.type_id);
        b.count = binding_detail::count(type);
        if (descriptorType == vk::DescriptorType::eCombinedImageSampler) {
          b.viewType = binding_detail::view(type);
          b.multisampled = type.image.ms; b.depthComparison = type.image.depth;
          const auto& scalar = compiler.get_type(type.image.type);
          if (scalar.width != 32) throw std::runtime_error("Only 32-bit sampled image scalars are supported");
          if (scalar.basetype == spirv_cross::SPIRType::Float) b.numeric = ShaderNumericType::Float;
          else if (scalar.basetype == spirv_cross::SPIRType::Int) b.numeric = ShaderNumericType::Sint;
          else if (scalar.basetype == spirv_cross::SPIRType::UInt) b.numeric = ShaderNumericType::Uint;
          else throw std::runtime_error("Unsupported sampled image scalar type");
        } else {
          const auto& block = compiler.get_type(resource.base_type_id);
          b.minimumSize = compiler.get_declared_struct_size(block);
          b.blockLayout = binding_detail::blockType(compiler, block);
        }
        bindings.push_back(std::move(b));
      }
    };
    add(resources.uniform_buffers, vk::DescriptorType::eUniformBuffer);
    add(resources.storage_buffers, vk::DescriptorType::eStorageBuffer);
    add(resources.sampled_images, vk::DescriptorType::eCombinedImageSampler);
    return destination.merge(bindings);
  } catch (const std::exception& error) {
    result.fail(currentSet, currentBinding, stage, "supported SPIR-V descriptor interface", error.what());
    return result;
  }
}
} // namespace vkb
