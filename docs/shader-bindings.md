# Shader resource binding validation

VKBuilder has an opt-in, C++17 runtime validation path:

- `vkbuilder/binding_validation.hpp`: cached descriptor contracts, layout checks,
  and complete descriptor-set writes. Depends only on Vulkan-Hpp and VKBuilder.
- `vkbuilder/shader_reflection.hpp`: populate the same contract from the **actual
  SPIR-V loaded by the application**, using SPIRV-Cross core. Include this only
  in translation units that need reflection, and link `spirv-cross-core`.

Validation returns `BindingValidation`; inspect `diagnostics` on failure. Each
item has `set`, `binding`, `stages`, `expected`, and `actual`. It stays enabled
in Release builds and does not depend on assertions or Vulkan validation layers.
The existing unchecked builders remain available for applications managing their
own contracts. Merely including the extension does not change existing calls.

## Reflect once, check layout, check actual resources

```cpp
#include <vkbuilder/shader_reflection.hpp>

vkb::ShaderBindingContract contract;
auto status = vkb::reflectShaderBindings(
    contract, vertexWords, vk::ShaderStageFlagBits::eVertex);
if (!status) return report(status.diagnostics);
status = vkb::reflectShaderBindings(
    contract, fragmentWords, vk::ShaderStageFlagBits::eFragment);
if (!status) return report(status.diagnostics);

// This is the exact vector subsequently passed to Vulkan to create the layout.
std::vector<vk::DescriptorSetLayoutBinding> layoutBindings = {
    {0, vk::DescriptorType::eUniformBuffer, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
    {1, vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment}
};
status = contract.validateLayout(1, layoutBindings);
if (!status) return report(status.diagnostics);
// Create the layout from layoutBindings, allocate the set, then wrap it:
vkb::UnboundSet destination{allocatedSet};

// These are records of actual creation, including the view's format/aspect.
// Do not construct this description from what the shader expects.
auto imageInfo = vkb::BindingImageInfo::from(
    imageCreateInfo, viewCreateInfo, samplerCreateInfo);
vkb::CheckedDescriptorSetBuilder bindings(
    contract, 1, destination, physicalDeviceProperties.limits);
bindings.buffer(0, actualUniformBuffer, 0, uniformByteCount)
        .image(1, sampledImage, imageInfo);
status = bindings.update(device);
if (!status) return report(status.diagnostics);
auto bound = std::move(destination).publish();
```

`reflectShaderBindings` selects the named entry point (default `main`) and stage.
It reflects statically active descriptor resources, merges stages, and refuses
incompatible declarations at the same set/binding. A failed reflection or merge
leaves the previous contract intact. Shader names and debug IDs are not binding
identities. The application must use the same binaries and entry points to create
its pipeline; an opaque `vk::ShaderModule` cannot be reflected after creation.

`validateLayout` checks descriptor kinds, counts, and stage visibility. Unused
extra host bindings are legal. Use the checked layouts when creating the pipeline
layout and descriptor sets. Opaque Vulkan layout/set handles do not expose their
creation records, so the library cannot detect a caller substituting another set.

`CheckedDescriptorSetBuilder` copies the contract and resource metadata. It keeps
no pointers to the original builders, buffers, image descriptions, or vectors.
It validates the complete set before issuing one Vulkan descriptor update:

- Required descriptors and every fixed array element must be supplied exactly
  once. Unknown bindings, duplicates and out-of-range elements are errors.
- UBO and SSBO bindings check handles, actual VKBuilder buffer usage and size,
  offset alignment, overflow-safe ranges, shader minimum size and device limits.
  `VK_WHOLE_SIZE` is resolved relative to the binding offset for these checks.
- Combined sampled images check view dimension, arrayed/cube shape, numeric
  category derived from view format/aspect, multisampling, sampler comparison,
  readable descriptor layout, usage, and non-null handles.
- Validation failure issues no Vulkan call. Validating itself never touches a
  GPU, so a contract can be tested before a device is available.

The caller owns resources through GPU use and must obey the existing UnboundSet /
BoundSet synchronization contract. A successful update does not submit work or
publish the set. Vulkan errors from a valid device call follow Vulkan-Hpp's normal
error handling. `update()` cannot inspect resource lifetime or command-buffer
barriers; a recorded descriptor layout is not proof of a completed transition.

## Buffer layout evidence

Cross-stage merging compares buffer member types, offsets, array strides, matrix
strides and row/column major decoration. Equal byte sizes are insufficient: an
`ivec4` and a `vec4` block of the same size conflict.

An actual raw buffer does not encode its C++ member types. By default, binding a
buffer validates its accessible byte range, **not its contents or host struct
layout**. The optional `blockLayout` argument to `buffer()` compares an independently
derived host schema with the reflected structural signature. Copying the expected
signature into that argument proves nothing about the uploaded bytes. The signature
is an in-process SPIRV-Cross representation, not a stable serialization format.

A future generator can produce host packing code and typed resource setters from
this same contract. This change does not generate C++ structs or claim that size
checks alone catch same-size host layout errors. Runtime SSBO tails likewise do
not prove that arbitrary shader indices fit the supplied buffer.

## Supported boundary

This first runtime implementation supports fixed descriptor arrays of ordinary
UBOs, SSBOs, and combined sampled images. Runtime/specialized descriptor arrays,
separate images/samplers, storage images, texel buffers, subpass inputs,
acceleration structures and unsupported execution models are rejected during
reflection. Checked writes support complete initialization rather than descriptor
copies or partial updates. Dynamic descriptors and descriptor-indexing policies
need additional lifetime and offset contracts and are not accepted as equivalent
ordinary buffer layouts.

This extension checks descriptor interfaces. Push constants, vertex attributes,
inter-stage varyings, image filter support, feature enabling and SPIR-V instruction
validity remain separate concerns. Run SPIRV-Tools and Vulkan validation layers as
appropriate. Normalized/sRGB/float sampled formats share the shader float category;
`sampler2D` does not prescribe an exact `VkFormat`.

For hot reload, reflect new binaries into a candidate contract, check existing
layouts and resource descriptions, and replace the program only after all checks
pass. Cache contracts per shader variant instead of reflecting or doing string
lookups per draw.

## Tests

```sh
cmake -S . -B build -DVKB_BUILD_WINDOW_TESTS=OFF
cmake --build build
ctest --test-dir build --output-on-failure

# Additional real-GLSL reflection fixtures (requires glslc and SPIRV-Cross core):
cmake -S . -B build-reflect -DVKB_BUILD_WINDOW_TESTS=OFF \
  -DVKB_BUILD_REFLECTION_TESTS=ON
cmake --build build-reflect
ctest --test-dir build-reflect --output-on-failure
```

The core validation tests require no SPIRV-Cross installation. Reflection tests
are explicit: requesting them with a missing dependency is a configure error.
CI builds both configurations. Tests record the Vulkan descriptor update dispatch
without needing a GPU, including proof that invalid writes never reach it.
