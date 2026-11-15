#define VKB_IMPL
#include "vkbuilder/binding_validation.hpp"
#ifdef VKB_TEST_REFLECTION
#include "vkbuilder/shader_reflection.hpp"
#include <fstream>
#endif
#include <cstdio>

static int failures = 0, checks = 0;
#define CHECK(...) do { ++checks; if (!(__VA_ARGS__)) { \
  std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #__VA_ARGS__); ++failures; } } while (0)

static int updates = 0;
static VKAPI_ATTR void VKAPI_CALL recordUpdate(VkDevice, uint32_t count,
    const VkWriteDescriptorSet* writes, uint32_t copies, const VkCopyDescriptorSet*) {
  ++updates;
  CHECK(count == 1); CHECK(copies == 0);
  CHECK(writes[0].dstBinding == 0);
  CHECK(writes[0].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  CHECK(writes[0].pBufferInfo->range == 80);
}

static vkb::ShaderBinding uniform() {
  vkb::ShaderBinding b;
  b.set = 1; b.stages = vk::ShaderStageFlagBits::eFragment;
  b.minimumSize = 80; b.blockLayout = "vec4@0;mat4@16:stride16:column";
  return b;
}
static vk::PhysicalDeviceLimits limits() {
  vk::PhysicalDeviceLimits l;
  l.minUniformBufferOffsetAlignment = 16; l.minStorageBufferOffsetAlignment = 16;
  l.maxUniformBufferRange = 65536; l.maxStorageBufferRange = 1u << 20;
  return l;
}
static vkb::GenericBuffer buffer(vk::DeviceSize size = 256) {
  vkb::GenericBuffer b;
  b.buffer = vk::Buffer{(VkBuffer)(uintptr_t)1}; b.size = size;
  b.usage_flags = vk::BufferUsageFlagBits::eUniformBuffer;
  return b;
}
static vkb::UnboundSet set() { return vkb::UnboundSet{vk::DescriptorSet{(VkDescriptorSet)(uintptr_t)1}}; }

static void validationTests() {
  vkb::ShaderBindingContract contract;
  auto b = uniform();
  CHECK(contract.merge({b}));
  std::vector<vk::DescriptorSetLayoutBinding> layout = {
    {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment}};
  CHECK(contract.validateLayout(1, layout));
  CHECK(!contract.validateLayout(1, {}));
  layout[0].stageFlags = vk::ShaderStageFlagBits::eVertex;
  CHECK(!contract.validateLayout(1, layout));
  layout[0].stageFlags = vk::ShaderStageFlagBits::eFragment;
  layout.push_back(layout.front());
  CHECK(!contract.validateLayout(1, layout));
  auto bad = b; bad.blockLayout = "ivec4@0;mat4@16:stride16:column";
  bad.stages = vk::ShaderStageFlagBits::eVertex;
  CHECK(!contract.merge({bad}));
  CHECK(contract.bindings().size() == 1);
  CHECK(contract.bindings()[0].stages == vk::ShaderStageFlagBits::eFragment);
  bad = b; bad.stages = vk::ShaderStageFlagBits::eVertex;
  CHECK(contract.merge({bad}));
  CHECK(contract.bindings()[0].stages == (b.stages | bad.stages));
  auto u = buffer(); auto dst = set(); auto l = limits();
  using Builder = vkb::CheckedDescriptorSetBuilder;
  CHECK(!Builder(contract, 1, dst, l).validate());
  CHECK(Builder(contract, 1, dst, l).buffer(0, u).validate());
  CHECK(Builder(contract, 1, dst, l).buffer(0, u, 16, 80).validate());
  VULKAN_HPP_DEFAULT_DISPATCHER.vkUpdateDescriptorSets = recordUpdate;
  CHECK(Builder(contract, 1, dst, l).buffer(0, u, 0, 80).update(vk::Device{(VkDevice)(uintptr_t)1}));
  CHECK(updates == 1);
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u, 0, 64).update(vk::Device{(VkDevice)(uintptr_t)1}));
  CHECK(updates == 1);
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u, 1, 80).validate());
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u, 0, 64).validate());
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u, 240, 80).validate());
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u, 256).validate());
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u, 0, 0).validate());
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u, 0, std::numeric_limits<uint64_t>::max()-1).validate());
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u).buffer(0, u).validate());
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u).buffer(7, u).validate());
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u, 0, 80, 1).validate());
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u, 0, 80, 0, "different layout").validate());
  CHECK(Builder(contract, 1, dst, l).buffer(0, u, 0, 80, 0, b.blockLayout).validate());
  l.maxUniformBufferRange = 128;
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u).validate());
  l = limits(); u.usage_flags = vk::BufferUsageFlagBits::eStorageBuffer;
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u).validate());
  // Failed update must not invoke any Vulkan dispatch; null device is intentional.
  CHECK(!Builder(contract, 1, dst, l).buffer(0, u).update(vk::Device{}));
  vkb::UnboundSet nullSet;
  CHECK(!Builder(contract, 1, nullSet, l).buffer(0, u).validate());

  vkb::ShaderBinding image;
  image.set = 1; image.binding = 2; image.count = 2;
  image.type = vk::DescriptorType::eCombinedImageSampler;
  image.stages = vk::ShaderStageFlagBits::eFragment; image.viewType = vk::ImageViewType::eCube;
  vkb::ShaderBindingContract images;
  CHECK(images.merge({image}));
  vkb::SampledImage sampled{vk::Sampler{(VkSampler)(uintptr_t)1}, vk::ImageView{(VkImageView)(uintptr_t)1}};
  vkb::BindingImageInfo info;
  info.format = vk::Format::eR8G8B8A8Unorm; info.viewType = vk::ImageViewType::eCube; info.usage = vk::ImageUsageFlagBits::eSampled;
  CHECK(Builder(images, 1, dst, l).image(2, sampled, info, 0).image(2, sampled, info, 1).validate());
  CHECK(!Builder(images, 1, dst, l).image(2, sampled, info, 0).validate());
  auto wrong = info; wrong.viewType = vk::ImageViewType::e2D;
  auto mismatch = Builder(images, 1, dst, l).image(2, sampled, wrong, 0).image(2, sampled, info, 1).validate();
  CHECK(!mismatch); CHECK(mismatch.diagnostics[0].set == 1); CHECK(mismatch.diagnostics[0].binding == 2);
  wrong = info; wrong.format = vk::Format::eR32Uint;
  CHECK(!Builder(images, 1, dst, l).image(2, sampled, wrong, 0).image(2, sampled, info, 1).validate());
  wrong = info; wrong.format = vk::Format::eUndefined;
  CHECK(!Builder(images, 1, dst, l).image(2, sampled, wrong, 0).image(2, sampled, info, 1).validate());
  wrong = info; wrong.samples = vk::SampleCountFlagBits::e4;
  CHECK(!Builder(images, 1, dst, l).image(2, sampled, wrong, 0).image(2, sampled, info, 1).validate());
  wrong = info; wrong.compareEnable = true;
  CHECK(!Builder(images, 1, dst, l).image(2, sampled, wrong, 0).image(2, sampled, info, 1).validate());
  wrong = info; wrong.usage = vk::ImageUsageFlagBits::eStorage;
  CHECK(!Builder(images, 1, dst, l).image(2, sampled, wrong, 0).image(2, sampled, info, 1).validate());
  CHECK(!Builder(images, 1, dst, l).image(2, {}, info, 0).image(2, sampled, info, 1).validate());
  sampled.layout = vk::ImageLayout::eTransferDstOptimal;
  CHECK(!Builder(images, 1, dst, l).image(2, sampled, info, 0).image(2, sampled, info, 1).validate());
}

#ifdef VKB_TEST_REFLECTION
static std::vector<uint32_t> load(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("Missing fixture: " + path);
  const auto bytes = input.tellg();
  std::vector<uint32_t> words(size_t(bytes) / 4);
  input.seekg(0); input.read(reinterpret_cast<char*>(words.data()), bytes);
  return words;
}
static void reflectionTests(const std::string& directory) {
  vkb::ShaderBindingContract contract;
  CHECK(vkb::reflectShaderBindings(contract, load(directory + "/resources.frag.spv"), vk::ShaderStageFlagBits::eFragment));
  CHECK(contract.bindings().size() == 4);
  CHECK(contract.bindings()[0].minimumSize == 80);
  CHECK(!contract.bindings()[0].blockLayout.empty());
  CHECK(vkb::reflectShaderBindings(contract, load(directory + "/compatible.vert.spv"), vk::ShaderStageFlagBits::eVertex));
  const auto before = contract.bindings()[0].stages;
  CHECK(!vkb::reflectShaderBindings(contract, load(directory + "/conflict.vert.spv"), vk::ShaderStageFlagBits::eVertex));
  CHECK(contract.bindings().size() == 4); CHECK(contract.bindings()[0].stages == before);
  CHECK(!vkb::reflectShaderBindings(contract, load(directory + "/row-major.vert.spv"), vk::ShaderStageFlagBits::eVertex));
  CHECK(!vkb::reflectShaderBindings(contract, load(directory + "/runtime.comp.spv"), vk::ShaderStageFlagBits::eCompute));
  vkb::ShaderBindingContract nested;
  CHECK(vkb::reflectShaderBindings(nested, load(directory + "/nested.comp.spv"), vk::ShaderStageFlagBits::eCompute));
  CHECK(nested.bindings().size() == 1);
  CHECK(nested.bindings()[0].minimumSize == 224);
  CHECK(nested.bindings()[0].blockLayout.find("a64") != std::string::npos);
  CHECK(nested.bindings()[0].blockLayout.find("a48") != std::string::npos);
  CHECK(nested.bindings()[0].blockLayout.find("a16") != std::string::npos);
  CHECK(!vkb::reflectShaderBindings(contract, load(directory + "/unsupported.comp.spv"), vk::ShaderStageFlagBits::eCompute));
  CHECK(!vkb::reflectShaderBindings(contract, load(directory + "/resources.frag.spv"), vk::ShaderStageFlagBits::eVertex));
  CHECK(!vkb::reflectShaderBindings(contract, load(directory + "/resources.frag.spv"), vk::ShaderStageFlagBits::eFragment, "missing"));
  CHECK(!vkb::reflectShaderBindings(contract, {}, vk::ShaderStageFlagBits::eFragment));
  CHECK(contract.bindings().size() == 4);
  // Use the reflected contract to validate a complete set of resource bindings.
  auto dst = set(); auto u = buffer(80); auto storage = buffer(64);
  storage.usage_flags = vk::BufferUsageFlagBits::eStorageBuffer;
  vkb::SampledImage sampled{vk::Sampler{(VkSampler)(uintptr_t)1}, vk::ImageView{(VkImageView)(uintptr_t)1}};
  vkb::BindingImageInfo cube, image;
  cube.format = image.format = vk::Format::eR8G8B8A8Unorm;
  cube.viewType = vk::ImageViewType::eCube; cube.usage = image.usage = vk::ImageUsageFlagBits::eSampled;
  vkb::CheckedDescriptorSetBuilder builder(contract, 1, dst, limits());
  builder.buffer(0, u).buffer(3, storage).image(1, sampled, cube)
      .image(2, sampled, image, 0).image(2, sampled, image, 1);
  CHECK(builder.validate());
}
#endif

int main(int argc, char** argv) {
  validationTests();
#ifdef VKB_TEST_REFLECTION
  if (argc != 2) return 2;
  reflectionTests(argv[1]);
#else
  (void)argc; (void)argv;
#endif
  std::printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
