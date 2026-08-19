#pragma once
// vkbuilder/framegraph.hpp
//
// A dependency-graph renderer for vkb::Device / vkb::Swapchain.
//
// Unlike the classic Present-based recording (one command buffer per frame,
// implicit begin*/end*SampledLayout layout bookkeeping), a FrameGraph:
//
//   1. makes every pass declare its resource accesses explicitly
//      (read / write / colorAttachment / depthAttachment);
//   2. derives pass dependencies from those declarations instead of manual
//      ordering;
//   3. plans image layout transitions and memory barriers automatically,
//      folding attachment transitions into render-pass initial/final layouts
//      and merging everything else into one prologue barrier per pass;
//   4. groups independent passes into layers so their command buffers can be
//      recorded concurrently on worker threads;
//   5. caches VkRenderPass / VkFramebuffer / command buffers keyed by their
//      signatures, so rebuilding the same graph every frame is cheap and
//      culled passes simply disappear.
//
// The graph is deliberately rebuilt each frame (draw lists change). Resource
// handles stay stable across frames; physical Vulkan objects are created once
// and reused until their description changes.
//
// Threading contract:
//   * build/compile happen on the main thread;
//   * record() may run pass callbacks concurrently (one callback per pass);
//     callbacks must only touch their own command buffer and read-only data;
//   * descriptor-set updates must happen before record() (vkUpdateDescriptorSets
//     is not thread-safe by default).
//
// Example:
//
//   vkb::FrameGraph graph{&device, 2};
//   auto color = graph.createTexture("sceneColor", vkb::TextureDesc{...});
//   graph.addPass("gbuffer")
//       .colorAttachment(color, vkb::AttachmentOp::clear(clearColor))
//       .record([&](vkb::FrameGraphPassContext &ctx) {
//         ctx.commandBuffer().bindPipeline(...);
//         ...
//       });
//   graph.addPass("compose")
//       .sample(color)
//       .record([&](vkb::FrameGraphPassContext &ctx) { ... });
//   graph.compile();
//   graph.execute();   // acquire (if swapchain), record, submit, present

#include "vkbuilder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace vkb {

// ---------------------------------------------------------------------------
// Opaque resource handles
// ---------------------------------------------------------------------------

struct TextureHandle {
  uint32_t id = 0xFFFFFFFFu;
  bool valid() const { return id != 0xFFFFFFFFu; }
};

struct BufferHandle {
  uint32_t id = 0xFFFFFFFFu;
  bool valid() const { return id != 0xFFFFFFFFu; }
};

inline bool operator==(const TextureHandle &a, const TextureHandle &b) {
  return a.id == b.id;
}
inline bool operator!=(const TextureHandle &a, const TextureHandle &b) {
  return !(a == b);
}
inline bool operator==(const BufferHandle &a, const BufferHandle &b) {
  return a.id == b.id;
}
inline bool operator!=(const BufferHandle &a, const BufferHandle &b) {
  return !(a == b);
}

// ---------------------------------------------------------------------------
// Access descriptions
// ---------------------------------------------------------------------------

/// How a pass touches a texture: layout plus the stage/access pair used to
/// build memory dependencies.
struct TextureUsage {
  vk::ImageLayout layout = vk::ImageLayout::eUndefined;
  vk::PipelineStageFlags stages{};
  vk::AccessFlags access{};

  bool operator==(const TextureUsage &o) const {
    return layout == o.layout && stages == o.stages && access == o.access;
  }
  bool operator!=(const TextureUsage &o) const { return !(*this == o); }
};

/// Buffer access has no layout.
struct BufferUsage {
  vk::PipelineStageFlags stages{};
  vk::AccessFlags access{};

  bool operator==(const BufferUsage &o) const {
    return stages == o.stages && access == o.access;
  }
  bool operator!=(const BufferUsage &o) const { return !(*this == o); }
};

/// Preset access patterns. `sample()` is the common "read as texture" case.
namespace access {

inline TextureUsage sample() {
  return {vk::ImageLayout::eShaderReadOnlyOptimal,
          vk::PipelineStageFlagBits::eVertexShader |
              vk::PipelineStageFlagBits::eFragmentShader |
              vk::PipelineStageFlagBits::eComputeShader,
          vk::AccessFlagBits::eShaderRead};
}

inline TextureUsage sampleFragment() {
  return {vk::ImageLayout::eShaderReadOnlyOptimal,
          vk::PipelineStageFlagBits::eFragmentShader,
          vk::AccessFlagBits::eShaderRead};
}

inline TextureUsage sampleCompute() {
  return {vk::ImageLayout::eShaderReadOnlyOptimal,
          vk::PipelineStageFlagBits::eComputeShader,
          vk::AccessFlagBits::eShaderRead};
}

/// Depth read-only sampling (PCF comparison).
inline TextureUsage depthRead() {
  return {vk::ImageLayout::eDepthStencilReadOnlyOptimal,
          vk::PipelineStageFlagBits::eEarlyFragmentTests |
              vk::PipelineStageFlagBits::eLateFragmentTests,
          vk::AccessFlagBits::eDepthStencilAttachmentRead};
}

inline TextureUsage colorAttachment() {
  return {vk::ImageLayout::eColorAttachmentOptimal,
          vk::PipelineStageFlagBits::eColorAttachmentOutput,
          vk::AccessFlagBits::eColorAttachmentRead |
              vk::AccessFlagBits::eColorAttachmentWrite};
}

inline TextureUsage depthAttachment() {
  return {vk::ImageLayout::eDepthStencilAttachmentOptimal,
          vk::PipelineStageFlagBits::eEarlyFragmentTests |
              vk::PipelineStageFlagBits::eLateFragmentTests,
          vk::AccessFlagBits::eDepthStencilAttachmentRead |
              vk::AccessFlagBits::eDepthStencilAttachmentWrite};
}

inline TextureUsage transferWrite() {
  return {vk::ImageLayout::eTransferDstOptimal,
          vk::PipelineStageFlagBits::eTransfer,
          vk::AccessFlagBits::eTransferWrite};
}

inline TextureUsage transferRead() {
  return {vk::ImageLayout::eTransferSrcOptimal,
          vk::PipelineStageFlagBits::eTransfer,
          vk::AccessFlagBits::eTransferRead};
}

inline TextureUsage storageWrite() {
  return {vk::ImageLayout::eGeneral,
          vk::PipelineStageFlagBits::eComputeShader,
          vk::AccessFlagBits::eShaderWrite};
}

inline TextureUsage storageRead() {
  return {vk::ImageLayout::eGeneral,
          vk::PipelineStageFlagBits::eComputeShader,
          vk::AccessFlagBits::eShaderRead};
}

inline TextureUsage present() {
  return {vk::ImageLayout::ePresentSrcKHR,
          vk::PipelineStageFlagBits::eBottomOfPipe, {}};
}

inline BufferUsage bufferRead() {
  return {vk::PipelineStageFlagBits::eVertexShader |
              vk::PipelineStageFlagBits::eFragmentShader |
              vk::PipelineStageFlagBits::eComputeShader,
          vk::AccessFlagBits::eShaderRead};
}

inline BufferUsage bufferWrite() {
  return {vk::PipelineStageFlagBits::eComputeShader,
          vk::AccessFlagBits::eShaderWrite};
}

inline BufferUsage transferBufferWrite() {
  return {vk::PipelineStageFlagBits::eTransfer,
          vk::AccessFlagBits::eTransferWrite};
}

inline BufferUsage transferBufferRead() {
  return {vk::PipelineStageFlagBits::eTransfer,
          vk::AccessFlagBits::eTransferRead};
}

} // namespace access

// ---------------------------------------------------------------------------
// Resource descriptions
// ---------------------------------------------------------------------------

struct TextureDesc {
  vk::Format format = vk::Format::eR8G8B8A8Unorm;
  vk::Extent3D extent{1, 1, 1};
  uint32_t mipLevels = 1;
  uint32_t arrayLayers = 1;
  vk::ImageType imageType = vk::ImageType::e2D;
  vk::ImageViewType viewType = vk::ImageViewType::e2D;
  vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
  vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
  vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eColorAttachment;
  vk::ImageCreateFlags createFlags{};
  /// Number of physical copies kept in a ring (2 = double buffered).
  uint32_t framesInFlight = 1;
  /// Layout the resource should be left in after this frame when no later pass
  /// in the same frame needs it. eUndefined = auto-decide:
  /// shader-read-only when the resource is sampled anywhere this frame,
  /// otherwise the attachment-optimal layout.
  vk::ImageLayout afterLayout = vk::ImageLayout::eUndefined;
};

struct BufferDesc {
  vk::DeviceSize size = 0;
  vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer;
  vk::MemoryPropertyFlags memory = vk::MemoryPropertyFlagBits::eDeviceLocal;
  uint32_t framesInFlight = 1;
};

/// Load/store behaviour for a render-pass attachment.
struct AttachmentOp {
  vk::AttachmentLoadOp loadOp = vk::AttachmentLoadOp::eClear;
  vk::AttachmentStoreOp storeOp = vk::AttachmentStoreOp::eStore;
  vk::ClearValue clearValue{};

  static AttachmentOp clear(const vk::ClearValue &c) {
    AttachmentOp op;
    op.loadOp = vk::AttachmentLoadOp::eClear;
    op.storeOp = vk::AttachmentStoreOp::eStore;
    op.clearValue = c;
    return op;
  }
  static AttachmentOp clearColor(float r, float g, float b, float a = 1.0f) {
    return clear(vk::ClearValue(vk::ClearColorValue(
        std::array<float, 4>{r, g, b, a})));
  }
  static AttachmentOp load() {
    AttachmentOp op;
    op.loadOp = vk::AttachmentLoadOp::eLoad;
    op.storeOp = vk::AttachmentStoreOp::eStore;
    return op;
  }
  static AttachmentOp dontCare() {
    AttachmentOp op;
    op.loadOp = vk::AttachmentLoadOp::eDontCare;
    op.storeOp = vk::AttachmentStoreOp::eDontCare;
    return op;
  }
};

enum class PassType { Graphics, Compute, Transfer };

struct FrameGraphPassContext;
using PassRecordCallback = std::function<void(FrameGraphPassContext &)>;
using PassRecordExecutor = std::function<void(
    const std::vector<uint32_t> &layerPasses,
    const std::function<void(uint32_t passOrder)> &recordOne)>;

namespace fg {

// --- plan-time structures (device-free) -----------------------------------

struct ImageTransition {
  uint32_t resource = 0;
  vk::ImageLayout oldLayout = vk::ImageLayout::eUndefined;
  vk::ImageLayout newLayout = vk::ImageLayout::eUndefined;
  vk::PipelineStageFlags srcStage{};
  vk::PipelineStageFlags dstStage{};
  vk::AccessFlags srcAccess{};
  vk::AccessFlags dstAccess{};
  vk::ImageSubresourceRange range;
};

struct BufferTransition {
  uint32_t resource = 0;
  vk::PipelineStageFlags srcStage{};
  vk::PipelineStageFlags dstStage{};
  vk::AccessFlags srcAccess{};
  vk::AccessFlags dstAccess{};
};

/// One merged prologue barrier: all transitions a pass needs before it starts.
struct PassBarrier {
  std::vector<ImageTransition> images;
  std::vector<BufferTransition> buffers;
  bool empty() const { return images.empty() && buffers.empty(); }
};

struct AccessRef {
  uint32_t resource = 0;
  bool isTexture = true;
  bool isWrite = false;
  bool isAttachment = false;
  TextureUsage tex;
  BufferUsage buf;
  uint32_t colorIndex = 0;
  bool isDepth = false;
};

struct AttachmentRef {
  uint32_t resource = 0;
  bool isDepth = false;
  AttachmentOp op;
};

struct PassData {
  std::string name;
  PassType type = PassType::Graphics;
  std::vector<AccessRef> accesses;
  std::vector<AttachmentRef> attachments;
  PassRecordCallback record;
};

struct ResourceData {
  std::string name;
  bool isTexture = true;
  TextureDesc tex;
  BufferDesc buf;
  bool imported = false;
  bool isSwapchain = false;
  bool output = false;
};

/// Physical runtime objects for one resource (owned or imported).
struct ResourceRuntime {
  struct TextureRT {
    bool own = false;
    vk::UniqueImage image;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory memory;
    vk::Image externalImage{};
    vk::ImageView externalView{};
  };
  struct BufferRT {
    bool own = false;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::Buffer externalBuffer{};
  };
  std::vector<TextureRT> textures;
  std::vector<BufferRT> buffers;
  uint32_t framesInFlight = 1;
};

struct AttachmentBinding {
  uint32_t resource = 0;
  bool isDepth = false;
  AttachmentOp op;
  vk::ImageLayout initialLayout = vk::ImageLayout::eUndefined;
  vk::ImageLayout finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
  vk::PipelineStageFlags srcStages{};
  vk::PipelineStageFlags dstStages{};
  vk::AccessFlags srcAccess{};
  vk::AccessFlags dstAccess{};
};

struct PassInstance {
  std::string name;
  PassType type = PassType::Graphics;
  uint32_t passId = 0;
  uint32_t order = 0;
  uint32_t layer = 0;
  PassBarrier barrier;
  bool hasRenderPass = false;
  std::string rpSignature;
  vk::RenderPass renderPass{};
  std::vector<AttachmentBinding> attachments;
  std::vector<vk::ClearValue> clears;
  std::vector<vk::Framebuffer> framebuffers;
  vk::Extent2D extent{};
  uint32_t framebufferCount = 1;
};

struct CompiledFrameGraph {
  std::vector<PassInstance> passes;    // active passes in topological order
  std::vector<uint32_t> passOrderById; // graph pass id -> index into passes
  std::vector<std::vector<uint32_t>> layers;

  std::string dump() const {
    std::ostringstream os;
    for (size_t l = 0; l < layers.size(); ++l) {
      os << "layer " << l << ":";
      for (uint32_t idx : layers[l])
        os << " [" << passes[idx].name << "]";
      os << "\n";
    }
    for (const auto &p : passes) {
      os << "pass " << p.order << " " << p.name << " layer=" << p.layer;
      if (p.hasRenderPass) {
        os << " rp=" << p.rpSignature;
        os << " attachments=";
        for (const auto &a : p.attachments)
          os << "res" << a.resource << "("
             << vk::to_string(a.initialLayout) << "->"
             << vk::to_string(a.finalLayout) << ") ";
      }
      if (!p.barrier.empty()) {
        os << " barriers=";
        for (const auto &t : p.barrier.images)
          os << "img" << t.resource << ":" << vk::to_string(t.oldLayout) << "->"
             << vk::to_string(t.newLayout) << " ";
        for (const auto &t : p.barrier.buffers)
          os << "buf" << t.resource << " ";
      }
      os << "\n";
    }
    return os.str();
  }
};

} // namespace fg

// ---------------------------------------------------------------------------
// FrameGraph
// ---------------------------------------------------------------------------

class FrameGraphPassBuilder;

class FrameGraph {
public:
  /// device may be null for pure planning/debugging; runtime calls then throw.
  explicit FrameGraph(vkb::Device *device = nullptr,
                      uint32_t framesInFlight = 2)
      : device_(device), framesInFlight_(framesInFlight) {}

  FrameGraph(const FrameGraph &) = delete;
  FrameGraph &operator=(const FrameGraph &) = delete;

  ~FrameGraph() { destroy(); }

  // -- resources -----------------------------------------------------------

  TextureHandle createTexture(const std::string &name, const TextureDesc &desc) {
    return addTexture(name, desc, false, false);
  }

  /// Import an externally-owned texture (e.g. a depth image managed by the
  /// engine). The graph never destroys the image/view.
  TextureHandle importTexture(const std::string &name, vk::Image image,
                              vk::ImageView view, const TextureDesc &desc) {
    TextureHandle h = addTexture(name, desc, true, false);
    resources_[h.id].imported = true;
    auto &rt = resourcesRT_[h.id];
    rt.framesInFlight = 1;
    rt.textures.resize(1);
    rt.textures[0].own = false;
    rt.textures[0].externalImage = image;
    rt.textures[0].externalView = view;
    return h;
  }

  /// Import the swapchain. Calling this again after a swapchain recreate
  /// rebinds the same logical handle to the new images.
  TextureHandle importSwapchain(vkb::Swapchain &swapchain) {
    swapchain_ = &swapchain;
    // Reuse an existing handle with the same name if present.
    for (size_t i = 0; i < resources_.size(); ++i) {
      if (resources_[i].name == "swapchain" && resources_[i].isSwapchain) {
        // Swapchain was recreated: rebind the same logical handle to the new
        // images/views.
        ensureSwapchainRuntime(uint32_t(i));
        return TextureHandle{uint32_t(i)};
      }
    }
    TextureDesc desc;
    desc.format = swapchain.image_format;
    desc.extent =
        vk::Extent3D(swapchain.extent.width, swapchain.extent.height, 1);
    desc.viewType = vk::ImageViewType::e2D;
    desc.usage = vk::ImageUsageFlagBits::eColorAttachment;
    desc.framesInFlight = swapchain.image_count;
    desc.afterLayout = vk::ImageLayout::ePresentSrcKHR;
    TextureHandle h = addTexture("swapchain", desc, true, true);
    resources_[h.id].output = true;
    ensureSwapchainRuntime(h.id);
    return h;
  }

  BufferHandle createBuffer(const std::string &name, const BufferDesc &desc) {
    return addBuffer(name, desc, false);
  }

  BufferHandle importBuffer(const std::string &name, vk::Buffer buffer,
                            const BufferDesc &desc) {
    BufferHandle h = addBuffer(name, desc, true);
    auto &rt = resourcesRT_[h.id];
    rt.framesInFlight = 1;
    rt.buffers.resize(1);
    rt.buffers[0].own = false;
    rt.buffers[0].externalBuffer = buffer;
    return h;
  }

  /// Mark a resource as an output: culling keeps every pass on a path from an
  /// output. Swapchain images are outputs automatically.
  void markOutput(TextureHandle h) {
    if (h.valid()) resources_[h.id].output = true;
  }
  void markOutput(BufferHandle h) {
    if (h.valid()) resources_[h.id].output = true;
  }

  // -- passes --------------------------------------------------------------

  FrameGraphPassBuilder addPass(const std::string &name,
                                PassType type = PassType::Graphics);

  // -- frame lifecycle -----------------------------------------------------

  /// Re-plan the graph and (re)create device objects that are missing.
  /// Safe to call every frame; caches make it cheap.
  void compile();

  /// Acquire the swapchain image (no-op without a swapchain) and select the
  /// frame slot. Returns false when acquisition failed (out-of-date surface).
  bool beginFrame();

  /// Record every pass command buffer. Passes in the same layer are recorded
  /// independently and may run concurrently through the supplied executor.
  /// The default executor records serially.
  void record(const PassRecordExecutor &executor = {});

  /// Submit all recorded command buffers in topological order.
  void submit();

  /// Present the swapchain image (no-op without a swapchain).
  void present();

  /// Convenience: beginFrame + record + submit + present.
  bool execute(const PassRecordExecutor &executor = {}) {
    if (!compiled_) compile();
    if (!beginFrame()) return false;
    // Re-plan after the frame slot is known so ring-buffered resources use the
    // right physical image's last-known layout.
    compile();
    record(executor);
    submit();
    present();
    return true;
  }

  /// Release all Vulkan objects. Safe to call more than once.
  void destroy();

  // -- introspection -------------------------------------------------------

  vkb::Device *device() const { return device_; }
  uint32_t framesInFlight() const { return framesInFlight_; }
  FrameSlot frameSlot() const { return FrameSlot{frameSlot_}; }
  uint32_t acquiredImageIndex() const {
    return hasAcquired_ ? acquiredImageIndex_ : 0u;
  }
  bool needsRecreate() const { return needsRecreate_; }

  const fg::CompiledFrameGraph &compiled() const {
    if (!compiled_) throw std::runtime_error("FrameGraph::compile() not called");
    return *compiled_;
  }

  /// Render pass handle of a compiled pass (for pipeline creation).
  vk::RenderPass renderPassOf(const std::string &passName) const {
    if (!compiled_) return {};
    for (const auto &p : compiled_->passes)
      if (p.name == passName) return p.renderPass;
    return {};
  }

  std::string dump() const {
    return compiled_ ? compiled_->dump() : std::string("(not compiled)");
  }

  // -- advanced: resolution helpers (used by FrameGraphPassContext) --------

  /// Physical image for a handle in the current frame.
  vk::Image resolveImage(TextureHandle h) const;
  vk::ImageView resolveImageView(TextureHandle h) const;
  vk::Buffer resolveBuffer(BufferHandle h) const;
  vk::DeviceSize bufferSize(BufferHandle h) const;

private:
  friend class FrameGraphPassBuilder;

  TextureHandle addTexture(const std::string &name, const TextureDesc &desc,
                           bool imported, bool isSwapchain);
  BufferHandle addBuffer(const std::string &name, const BufferDesc &desc,
                         bool imported);

  void ensureResourceRuntime(uint32_t resId);
  void ensureSwapchainRuntime(uint32_t resId);

  /// Planning (device-free): edges, topo order, culling, layers, barriers.
  std::shared_ptr<fg::CompiledFrameGraph> compilePlan();
  /// Create VkRenderPass / framebuffers / command buffers / sync objects.
  void materializeDeviceObjects(const fg::CompiledFrameGraph &plan);

  /// Canonical stage/access pair for a layout (cross-frame transition table).
  static void canonicalTransitionFrom(vk::ImageLayout layout,
                                      vk::PipelineStageFlags &stages,
                                      vk::AccessFlags &access);

  uint32_t physicalIndexFor(uint32_t resId) const;

  vk::CommandBuffer commandBufferFor(uint32_t passOrder);
  vk::Framebuffer framebufferFor(const fg::PassInstance &pass,
                                 uint32_t variant);

  vkb::Device *device_ = nullptr;
  uint32_t framesInFlight_ = 2;
  vkb::Swapchain *swapchain_ = nullptr;

  std::vector<fg::ResourceData> resources_;
  std::vector<fg::ResourceRuntime> resourcesRT_;
  std::vector<fg::PassData> passes_;

  std::shared_ptr<fg::CompiledFrameGraph> compiled_;

  // Device-object caches (stable across frames / recompiles).
  std::map<std::string, vk::UniqueRenderPass> renderPassCache_;
  std::map<std::string, vk::UniqueFramebuffer> framebufferCache_;
  std::vector<vk::CommandPool> pools_;
  std::vector<std::vector<vk::CommandBuffer>> cbs_; // [slot][passOrder]
  std::vector<vk::Fence> fences_;
  std::vector<vk::Semaphore> acquireSems_;
  std::vector<vk::Semaphore> finishedSems_;
  std::vector<vk::Fence> imageInFlight_;

  // Layout each physical image is left in after the last executed frame.
  std::vector<std::vector<vk::ImageLayout>> resourceLastLayout_;

  uint32_t frameSlot_ = 0;
  uint32_t frameIndex_ = 0;
  uint32_t acquiredImageIndex_ = 0;
  bool hasAcquired_ = false;
  bool needsRecreate_ = false;
  bool destroyed_ = false;
};

class FrameGraphPassBuilder {
public:
  FrameGraphPassBuilder(FrameGraph &graph, uint32_t passId)
      : graph_(graph), passId_(passId) {}

  FrameGraphPassBuilder &read(TextureHandle h, const TextureUsage &usage) {
    return addAccess(h.id, true, false, usage);
  }
  FrameGraphPassBuilder &read(BufferHandle h, const BufferUsage &usage) {
    return addAccess(h.id, false, false, usage);
  }
  FrameGraphPassBuilder &write(TextureHandle h, const TextureUsage &usage) {
    return addAccess(h.id, true, true, usage);
  }
  FrameGraphPassBuilder &write(BufferHandle h, const BufferUsage &usage) {
    return addAccess(h.id, false, true, usage);
  }

  /// Sugar for read(tex, access::sample()).
  FrameGraphPassBuilder &sample(TextureHandle h) {
    return read(h, access::sample());
  }

  FrameGraphPassBuilder &colorAttachment(TextureHandle h,
                                         const AttachmentOp &op = AttachmentOp{}) {
    auto &pass = graph_.passes_[passId_];
    pass.accesses.push_back(fg::AccessRef{
        h.id, true, true, true, access::colorAttachment(), {},
        uint32_t(pass.attachments.size()), false});
    pass.attachments.push_back(fg::AttachmentRef{h.id, false, op});
    return *this;
  }

  FrameGraphPassBuilder &depthAttachment(TextureHandle h,
                                         const AttachmentOp &op = AttachmentOp{}) {
    auto &pass = graph_.passes_[passId_];
    pass.accesses.push_back(fg::AccessRef{
        h.id, true, true, true, access::depthAttachment(), {}, 0, true});
    pass.attachments.push_back(fg::AttachmentRef{h.id, true, op});
    return *this;
  }

  /// Store the per-pass record callback. Ends the builder chain.
  FrameGraphPassBuilder &record(PassRecordCallback cb) {
    graph_.passes_[passId_].record = std::move(cb);
    return *this;
  }

private:
  template <typename Usage>
  FrameGraphPassBuilder &addAccess(uint32_t resId, bool isTexture,
                                   bool isWrite, const Usage &usage);

  FrameGraph &graph_;
  uint32_t passId_ = 0;
};

// ---------------------------------------------------------------------------
// Recording context handed to pass callbacks
// ---------------------------------------------------------------------------

struct FrameGraphPassContext {
  FrameSlot slot{};
  uint32_t passOrder = 0;
  FrameGraph *graph = nullptr;
  vk::CommandBuffer cmd{};

  vk::CommandBuffer &commandBuffer() { return cmd; }
  FrameSlot frameSlot() const { return slot; }
  uint32_t passIndex() const { return passOrder; }

  vk::Image image(TextureHandle h) const { return graph->resolveImage(h); }
  vk::ImageView view(TextureHandle h) const {
    return graph->resolveImageView(h);
  }
  vk::Buffer buffer(BufferHandle h) const { return graph->resolveBuffer(h); }
  vk::DeviceSize bufferSize(BufferHandle h) const {
    return graph->bufferSize(h);
  }

  /// Extent of the pass's first attachment (or 1x1 for attachment-less passes).
  vk::Extent2D extent() const {
    const auto &pass = graph->compiled().passes[passOrder];
    return pass.extent;
  }

  vk::RenderPass renderPass() const {
    const auto &pass = graph->compiled().passes[passOrder];
    return pass.renderPass;
  }
};

// ---------------------------------------------------------------------------
// Inline implementation
// ---------------------------------------------------------------------------

inline TextureHandle FrameGraph::addTexture(const std::string &name,
                                            const TextureDesc &desc,
                                            bool imported, bool isSwapchain) {
  fg::ResourceData rd;
  rd.name = name;
  rd.isTexture = true;
  rd.tex = desc;
  rd.imported = imported;
  rd.isSwapchain = isSwapchain;
  uint32_t id = uint32_t(resources_.size());
  resources_.push_back(std::move(rd));
  resourcesRT_.emplace_back();
  resourceLastLayout_.emplace_back();
  return TextureHandle{id};
}

inline BufferHandle FrameGraph::addBuffer(const std::string &name,
                                          const BufferDesc &desc,
                                          bool imported) {
  fg::ResourceData rd;
  rd.name = name;
  rd.isTexture = false;
  rd.buf = desc;
  rd.imported = imported;
  uint32_t id = uint32_t(resources_.size());
  resources_.push_back(std::move(rd));
  resourcesRT_.emplace_back();
  resourceLastLayout_.emplace_back();
  return BufferHandle{id};
}

inline FrameGraphPassBuilder FrameGraph::addPass(const std::string &name,
                                                 PassType type) {
  fg::PassData pd;
  pd.name = name;
  pd.type = type;
  passes_.push_back(std::move(pd));
  return FrameGraphPassBuilder(*this, uint32_t(passes_.size() - 1));
}

template <typename Usage>
inline FrameGraphPassBuilder &FrameGraphPassBuilder::addAccess(
    uint32_t resId, bool isTexture, bool isWrite, const Usage &usage) {
  fg::AccessRef ref;
  ref.resource = resId;
  ref.isTexture = isTexture;
  ref.isWrite = isWrite;
  if constexpr (std::is_same_v<Usage, TextureUsage>)
    ref.tex = usage;
  else
    ref.buf = usage;
  graph_.passes_[passId_].accesses.push_back(std::move(ref));
  return *this;
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

inline void FrameGraph::canonicalTransitionFrom(vk::ImageLayout layout,
                                                vk::PipelineStageFlags &stages,
                                                vk::AccessFlags &access) {
  using il = vk::ImageLayout;
  using afb = vk::AccessFlagBits;
  switch (layout) {
  case il::eColorAttachmentOptimal:
    stages = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    access = afb::eColorAttachmentWrite;
    break;
  case il::eDepthStencilAttachmentOptimal:
    stages = vk::PipelineStageFlagBits::eEarlyFragmentTests |
             vk::PipelineStageFlagBits::eLateFragmentTests;
    access = afb::eDepthStencilAttachmentWrite;
    break;
  case il::eDepthStencilReadOnlyOptimal:
    stages = vk::PipelineStageFlagBits::eEarlyFragmentTests |
             vk::PipelineStageFlagBits::eLateFragmentTests;
    access = afb::eDepthStencilAttachmentRead;
    break;
  case il::eShaderReadOnlyOptimal:
    stages = vk::PipelineStageFlagBits::eVertexShader |
             vk::PipelineStageFlagBits::eFragmentShader;
    access = afb::eShaderRead;
    break;
  case il::eTransferSrcOptimal:
    stages = vk::PipelineStageFlagBits::eTransfer;
    access = afb::eTransferRead;
    break;
  case il::eTransferDstOptimal:
    stages = vk::PipelineStageFlagBits::eTransfer;
    access = afb::eTransferWrite;
    break;
  case il::eGeneral:
    stages = vk::PipelineStageFlagBits::eComputeShader;
    access = afb::eShaderRead | afb::eShaderWrite;
    break;
  case il::ePresentSrcKHR:
    stages = vk::PipelineStageFlagBits::eBottomOfPipe;
    access = {};
    break;
  default:
    stages = vk::PipelineStageFlagBits::eTopOfPipe;
    access = {};
    break;
  }
}

namespace fg {

/// Compact byte serialization used for render-pass / framebuffer cache keys.
inline void appendBytes(std::string &s, const void *data, size_t n) {
  const char *p = static_cast<const char *>(data);
  s.append(p, n);
}

template <typename T>
inline void appendPOD(std::string &s, const T &v) {
  appendBytes(s, &v, sizeof(T));
}

inline std::string attachmentSignature(const AttachmentBinding &a) {
  std::string s;
  appendPOD(s, a.initialLayout);
  appendPOD(s, a.finalLayout);
  appendPOD(s, a.op.loadOp);
  appendPOD(s, a.op.storeOp);
  appendPOD(s, a.isDepth);
  appendPOD(s, a.srcStages);
  appendPOD(s, a.dstStages);
  appendPOD(s, a.srcAccess);
  appendPOD(s, a.dstAccess);
  return s;
}

} // namespace fg

inline std::shared_ptr<fg::CompiledFrameGraph> FrameGraph::compilePlan() {
  auto plan = std::make_shared<fg::CompiledFrameGraph>();
  const uint32_t passCount = uint32_t(passes_.size());
  const uint32_t resCount = uint32_t(resources_.size());

  // 1. Per-resource access lists, in pass registration order.
  struct ResAccess {
    uint32_t passId;
    uint32_t accessIndex;
    bool isWrite;
  };
  std::vector<std::vector<ResAccess>> resAccesses(resCount);
  for (uint32_t p = 0; p < passCount; ++p) {
    const auto &pd = passes_[p];
    for (size_t ai = 0; ai < pd.accesses.size(); ++ai) {
      const auto &a = pd.accesses[ai];
      if (a.resource < resCount)
        resAccesses[a.resource].push_back(
            ResAccess{p, uint32_t(ai), a.isWrite});
    }
  }

  // 2. Edges: connect each access to the last previous access from another
  //    pass that conflicts (either side writes, or the texture layout changes).
  //    Non-consecutive conflicts matter too (W, R1, R2 must order W before R2).
  std::vector<std::vector<uint32_t>> adj(passCount);
  for (uint32_t r = 0; r < resCount; ++r) {
    const auto &list = resAccesses[r];
    for (size_t i = 0; i < list.size(); ++i) {
      for (size_t j = i; j > 0; --j) {
        const auto &a = list[j - 1];
        const auto &b = list[i];
        if (a.passId == b.passId) continue; // same pass: user handles in-pass sync
        const auto &accA = passes_[a.passId].accesses[a.accessIndex];
        const auto &accB = passes_[b.passId].accesses[b.accessIndex];
        const bool layoutChanged =
            accA.isTexture && accB.isTexture &&
            accA.tex.layout != accB.tex.layout;
        if (a.isWrite || b.isWrite || layoutChanged) {
          adj[a.passId].push_back(b.passId);
          break;
        }
      }
    }
  }

  // 3. Topological sort (Kahn, deterministic by lowest pass id).
  std::vector<uint32_t> indeg(passCount, 0);
  for (uint32_t p = 0; p < passCount; ++p) {
    std::sort(adj[p].begin(), adj[p].end());
    adj[p].erase(std::unique(adj[p].begin(), adj[p].end()), adj[p].end());
    for (uint32_t q : adj[p]) ++indeg[q];
  }
  std::vector<uint32_t> ready;
  for (uint32_t p = 0; p < passCount; ++p)
    if (indeg[p] == 0) ready.push_back(p);
  std::vector<uint32_t> topo;
  topo.reserve(passCount);
  while (!ready.empty()) {
    // pick smallest pass id for determinism
    auto it = std::min_element(ready.begin(), ready.end());
    uint32_t p = *it;
    ready.erase(it);
    topo.push_back(p);
    for (uint32_t q : adj[p])
      if (--indeg[q] == 0) ready.push_back(q);
  }
  if (topo.size() != passCount) {
    std::ostringstream os;
    os << "FrameGraph cycle detected among passes:";
    for (uint32_t p = 0; p < passCount; ++p)
      if (indeg[p] != 0) os << " " << passes_[p].name;
    throw std::runtime_error(os.str());
  }

  // 4. Culling: keep passes that (transitively) feed an output resource.
  std::vector<char> needed(resCount, 0);
  bool anyOutput = false;
  for (uint32_t r = 0; r < resCount; ++r)
    if (resources_[r].output) {
      needed[r] = 1;
      anyOutput = true;
    }
  std::vector<char> active(passCount, 0);
  if (!anyOutput) {
    std::fill(active.begin(), active.end(), 1);
  } else {
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      const auto &pd = passes_[*it];
      bool isActive = false;
      for (const auto &a : pd.accesses)
        if (needed[a.resource]) isActive = true;
      if (isActive) {
        active[*it] = 1;
        for (const auto &a : pd.accesses) needed[a.resource] = 1;
      }
    }
  }

  // 5. Layers = longest path from sources.
  std::vector<uint32_t> layer(passCount, 0);
  for (uint32_t p : topo) {
    for (uint32_t q : adj[p]) layer[q] = std::max(layer[q], layer[p] + 1);
  }

  // 6. Barriers / attachment layouts.
  // Per-resource state AFTER each access, in active-pass topo order.
  struct ResState {
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    vk::PipelineStageFlags stages{};
    vk::AccessFlags access{};
    bool isWrite = false;
    uint32_t passId = 0xFFFFFFFFu;
    bool known = false;
  };
  auto hasWriteBits = [](vk::AccessFlags a) {
    return (a & (vk::AccessFlagBits::eTransferWrite |
                 vk::AccessFlagBits::eShaderWrite |
                 vk::AccessFlagBits::eColorAttachmentWrite |
                 vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                 vk::AccessFlagBits::eHostWrite |
                 vk::AccessFlagBits::eMemoryWrite)) != vk::AccessFlags{};
  };
  // In-frame per-resource usage sequences (active passes only, topo order).
  struct UsageSeq {
    std::vector<uint32_t> passIds; // in topo order
    std::vector<size_t> accessIdx;
    std::vector<bool> isAttachment;
  };
  std::vector<UsageSeq> usageSeq(resCount);

  std::vector<uint32_t> activePassList; // pass ids in topo order
  for (uint32_t p : topo)
    if (active[p]) activePassList.push_back(p);
  for (uint32_t p : activePassList) {
    const auto &pd = passes_[p];
    for (size_t ai = 0; ai < pd.accesses.size(); ++ai) {
      const auto &a = pd.accesses[ai];
      auto &sq = usageSeq[a.resource];
      sq.passIds.push_back(p);
      sq.accessIdx.push_back(ai);
      sq.isAttachment.push_back(a.isAttachment);
    }
  }

  // Build active pass instances.
  plan->passOrderById.assign(passCount, 0xFFFFFFFFu);
  plan->passes.reserve(activePassList.size());
  std::vector<std::vector<ResState>> stateSeq(resCount);

  // Seed the in-frame usage state from the physical image's last-known layout
  // (the plan runs after beginFrame(), so the frame slot is fixed). The seed
  // is stored as a synthetic state with passId == ~0 so the conflict scan sees
  // it as a real predecessor.
  auto seedState = [&](uint32_t r) -> ResState {
    ResState st;
    if (r >= resources_.size() || !resources_[r].isTexture) return st;
    if (r >= resourceLastLayout_.size()) return st;
    uint32_t phys = physicalIndexFor(r);
    if (phys >= resourceLastLayout_[r].size()) return st;
    vk::ImageLayout last = resourceLastLayout_[r][phys];
    if (last == vk::ImageLayout::eUndefined) return st;
    st.layout = last;
    canonicalTransitionFrom(last, st.stages, st.access);
    st.isWrite = hasWriteBits(st.access);
    st.known = true;
    return st;
  };
  auto ensureSeed = [&](uint32_t r) {
    if (stateSeq[r].empty() && seedState(r).known)
      stateSeq[r].push_back(seedState(r));
  };
  auto prevState = [&](uint32_t r) -> const ResState * {
    if (stateSeq[r].empty()) return nullptr;
    return &stateSeq[r].back();
  };

  for (uint32_t order = 0; order < activePassList.size(); ++order) {
    const uint32_t p = activePassList[order];
    const auto &pd = passes_[p];
    fg::PassInstance inst;
    inst.name = pd.name;
    inst.type = pd.type;
    inst.passId = p;
    inst.order = order;
    inst.layer = layer[p];
    plan->passOrderById[p] = order;

    // ---- attachments first (they define the render pass) ----
    if (!pd.attachments.empty()) {
      inst.hasRenderPass = true;
      vk::Extent2D extent(1, 1);
      bool firstExtent = true;
      for (const auto &att : pd.attachments) {
        const auto &rd = resources_[att.resource];
        if (!rd.isTexture)
          throw std::runtime_error("attachment must be a texture resource");
        fg::AttachmentBinding ab;
        ab.resource = att.resource;
        ab.isDepth = att.isDepth;
        ab.op = att.op;
        ab.finalLayout = att.isDepth
                             ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                             : vk::ImageLayout::eColorAttachmentOptimal;
        if (firstExtent) {
          extent = vk::Extent2D(rd.tex.extent.width, rd.tex.extent.height);
          firstExtent = false;
        }
        inst.attachments.push_back(ab);
      }
      inst.extent = extent;
      for (auto &ab : inst.attachments) {
        const auto &rd = resources_[ab.resource];
        const uint32_t physFrames =
            ab.resource < resources_.size() && resources_[ab.resource].isSwapchain
                ? (swapchain_ ? swapchain_->image_count : 1)
                : std::max(1u, rd.tex.framesInFlight);
        inst.framebufferCount = std::max(inst.framebufferCount, physFrames);
      }
    }

    // ---- resolve per-access transitions ----
    for (size_t ai = 0; ai < pd.accesses.size(); ++ai) {
      const auto &acc = pd.accesses[ai];
      if (acc.isAttachment) continue; // handled via render pass layouts
      const uint32_t r = acc.resource;
      const auto &rd = resources_[r];
      ensureSeed(r);

      // Last conflicting predecessor from another pass: either side writes,
      // or the texture layout changes.
      const ResState *src = nullptr;
      for (auto it = stateSeq[r].rbegin(); it != stateSeq[r].rend(); ++it) {
        if (it->passId == p) continue; // in-pass: user handles ordering
        const bool conflict =
            hasWriteBits(it->access) ||
            hasWriteBits(acc.isTexture ? acc.tex.access : acc.buf.access) ||
            (acc.isTexture && it->layout != acc.tex.layout);
        if (conflict) {
          src = &*it;
          break;
        }
      }

      if (acc.isTexture) {
        if (src) {
          fg::ImageTransition t;
          t.resource = r;
          t.oldLayout = src->layout;
          t.newLayout = acc.tex.layout;
          t.srcStage = src->stages;
          t.srcAccess = src->access;
          t.dstStage = acc.tex.stages;
          t.dstAccess = acc.tex.access;
          t.range = vk::ImageSubresourceRange(
              rd.tex.aspect, 0, rd.tex.mipLevels, 0, rd.tex.arrayLayers);
          inst.barrier.images.push_back(std::move(t));
        } else if (stateSeq[r].empty() &&
                   acc.tex.layout != vk::ImageLayout::eUndefined &&
                   acc.tex.layout != vk::ImageLayout::ePreinitialized) {
          // First touch: transition from undefined.
          fg::ImageTransition t;
          t.resource = r;
          t.oldLayout = vk::ImageLayout::eUndefined;
          t.newLayout = acc.tex.layout;
          t.srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
          t.dstStage = acc.tex.stages;
          t.dstAccess = acc.tex.access;
          t.range = vk::ImageSubresourceRange(
              rd.tex.aspect, 0, rd.tex.mipLevels, 0, rd.tex.arrayLayers);
          inst.barrier.images.push_back(std::move(t));
        }
        ResState st;
        st.layout = acc.tex.layout;
        st.stages = acc.tex.stages;
        st.access = acc.tex.access;
        st.isWrite = acc.isWrite;
        st.passId = p;
        st.known = true;
        stateSeq[r].push_back(st);
      } else {
        if (src) {
          fg::BufferTransition t;
          t.resource = r;
          t.srcStage = src->stages;
          t.srcAccess = src->access;
          t.dstStage = acc.buf.stages;
          t.dstAccess = acc.buf.access;
          inst.barrier.buffers.push_back(std::move(t));
        }
        ResState st;
        st.stages = acc.buf.stages;
        st.access = acc.buf.access;
        st.isWrite = acc.isWrite;
        st.passId = p;
        st.known = true;
        stateSeq[r].push_back(st);
      }
    }

    // ---- attachment layouts: initial from last use, final from next use ----
    {
      uint32_t ring = 0;
      for (const auto &ab : inst.attachments) {
        const auto &rd = resources_[ab.resource];
        if (rd.isSwapchain) continue;
        if (rd.tex.framesInFlight > 1) {
          if (ring != 0 && ring != rd.tex.framesInFlight)
            throw std::runtime_error(
                "pass '" + inst.name +
                "' mixes attachments with different framesInFlight");
          ring = rd.tex.framesInFlight;
        }
      }
    }
    for (auto &ab : inst.attachments) {
      const uint32_t r = ab.resource;
      const auto &rd = resources_[r];
      const auto &seq = usageSeq[r];
      ensureSeed(r);
      const ResState *prev = prevState(r);
      // Find this pass in the usage sequence.
      size_t idx = seq.passIds.size();
      for (size_t i = 0; i < seq.passIds.size(); ++i) {
        if (seq.passIds[i] == p && seq.isAttachment[i]) {
          idx = i;
          break;
        }
      }
      const bool preserve =
          ab.op.loadOp == vk::AttachmentLoadOp::eLoad && prev && prev->known;
      if (preserve) {
        ab.initialLayout = prev->layout;
        ab.srcStages = prev->stages;
        ab.srcAccess = prev->access;
      } else {
        ab.initialLayout = vk::ImageLayout::eUndefined;
      }

      // Next use after this pass.
      bool foundNext = false;
      for (size_t i = idx + 1; i < seq.passIds.size(); ++i) {
        const auto &nacc = passes_[seq.passIds[i]].accesses[seq.accessIdx[i]];
        if (seq.isAttachment[i]) {
          ab.finalLayout =
              nacc.isDepth ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                           : vk::ImageLayout::eColorAttachmentOptimal;
          ab.dstStages = nacc.tex.stages;
          ab.dstAccess = nacc.tex.access;
        } else if (nacc.isTexture) {
          ab.finalLayout = nacc.tex.layout;
          ab.dstStages = nacc.tex.stages;
          ab.dstAccess = nacc.tex.access;
        }
        foundNext = true;
        break;
      }
      if (!foundNext) {
        // Decide the after-frame layout.
        if (rd.tex.afterLayout != vk::ImageLayout::eUndefined) {
          ab.finalLayout = rd.tex.afterLayout;
          canonicalTransitionFrom(ab.finalLayout, ab.dstStages, ab.dstAccess);
        } else if (rd.isSwapchain) {
          ab.finalLayout = vk::ImageLayout::ePresentSrcKHR;
          ab.dstStages = vk::PipelineStageFlagBits::eBottomOfPipe;
          ab.dstAccess = {};
        } else {
          // Sampled anywhere this frame -> hand off shader-readable.
          bool sampledAnywhere = false;
          for (const auto &e : seq.passIds) {
            const auto &pd2 = passes_[e];
            for (const auto &a2 : pd2.accesses) {
              if (a2.resource == r && a2.isTexture &&
                  !a2.isAttachment &&
                  a2.tex.layout == vk::ImageLayout::eShaderReadOnlyOptimal)
                sampledAnywhere = true;
            }
          }
          if (sampledAnywhere) {
            ab.finalLayout =
                ab.isDepth ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                           : vk::ImageLayout::eShaderReadOnlyOptimal;
            canonicalTransitionFrom(ab.finalLayout, ab.dstStages, ab.dstAccess);
          }
        }
      }

      // Update the resource's in-frame state for the next access.
      ResState st;
      st.layout = ab.finalLayout;
      st.stages =
          ab.dstStages != vk::PipelineStageFlags{}
              ? ab.dstStages
              : (ab.isDepth ? vk::PipelineStageFlagBits::eEarlyFragmentTests
                            : vk::PipelineStageFlagBits::
                                  eColorAttachmentOutput);
      st.access =
          ab.dstAccess != vk::AccessFlags{}
              ? ab.dstAccess
              : (ab.isDepth ? vk::AccessFlagBits::eDepthStencilAttachmentWrite
                            : vk::AccessFlagBits::eColorAttachmentWrite);
      st.isWrite = true;
      st.passId = p;
      st.known = true;
      stateSeq[r].push_back(st);
    }

    inst.clears.clear();
    for (const auto &ab : inst.attachments)
      inst.clears.push_back(ab.op.clearValue);
    plan->passes.push_back(std::move(inst));
  }

  // Build layers from the active passes.
  {
    uint32_t maxLayer = 0;
    for (const auto &p : plan->passes) maxLayer = std::max(maxLayer, p.layer);
    plan->layers.resize(maxLayer + 1);
    for (uint32_t i = 0; i < plan->passes.size(); ++i)
      plan->layers[plan->passes[i].layer].push_back(i);
  }

  // Render-pass signatures.
  for (auto &p : plan->passes) {
    if (!p.hasRenderPass) continue;
    std::string sig;
    for (const auto &ab : p.attachments) {
      const auto &rd = resources_[ab.resource];
      sig += "att:";
      fg::appendPOD(sig, rd.tex.format);
      fg::appendPOD(sig, rd.tex.samples);
      sig += fg::attachmentSignature(ab);
    }
    sig += "ext:";
    {
      vk::PipelineStageFlags srcStages{};
      vk::PipelineStageFlags dstStages{};
      vk::AccessFlags srcAccess{};
      vk::AccessFlags dstAccess{};
      for (const auto &ab : p.attachments) {
        srcStages |= ab.srcStages;
        dstStages |= ab.dstStages;
        srcAccess |= ab.srcAccess;
        dstAccess |= ab.dstAccess;
      }
      fg::appendPOD(sig, srcStages);
      fg::appendPOD(sig, dstStages);
      fg::appendPOD(sig, srcAccess);
      fg::appendPOD(sig, dstAccess);
    }
    p.rpSignature = sig;
  }

  return plan;
}

inline void FrameGraph::compile() {
  if (destroyed_) return;
  compiled_ = compilePlan();
  if (device_) materializeDeviceObjects(*compiled_);
}

// ---------------------------------------------------------------------------
// Device-object materialization
// ---------------------------------------------------------------------------

namespace fg {

inline std::string framebufferSignature(vk::RenderPass rp,
                                        const std::vector<vk::ImageView> &views,
                                        const vk::Extent2D &extent,
                                        uint32_t layers) {
  std::string s;
  appendPOD(s, rp);
  for (auto v : views) appendPOD(s, v);
  appendPOD(s, extent);
  appendPOD(s, layers);
  return s;
}

} // namespace fg

inline void FrameGraph::ensureResourceRuntime(uint32_t resId) {
  if (!device_) return;
  auto &rd = resources_[resId];
  auto &rt = resourcesRT_[resId];
  const uint32_t frames = std::max(1u, rd.isTexture ? rd.tex.framesInFlight
                                                    : rd.buf.framesInFlight);
  rt.framesInFlight = frames;

  if (rd.isTexture) {
    if (rt.textures.size() != frames) rt.textures.resize(frames);
    for (uint32_t i = 0; i < frames; ++i) {
      auto &t = rt.textures[i];
      if (rd.imported) continue;
      if (t.own && t.image) continue;
      if (t.own) {
        t.image.reset();
        t.view.reset();
        t.memory.reset();
      }
      const auto &desc = rd.tex;
      vk::ImageCreateInfo ci;
      ci.flags = desc.createFlags;
      ci.imageType = desc.imageType;
      ci.format = desc.format;
      ci.extent = desc.extent;
      ci.mipLevels = desc.mipLevels;
      ci.arrayLayers = desc.arrayLayers;
      ci.samples = desc.samples;
      ci.tiling = vk::ImageTiling::eOptimal;
      ci.usage = desc.usage;
      ci.sharingMode = vk::SharingMode::eExclusive;
      ci.initialLayout = vk::ImageLayout::eUndefined;
      t.image = (*device_)->createImageUnique(ci);

      auto memreq = (*device_)->getImageMemoryRequirements(*t.image);
      vk::MemoryAllocateInfo mai;
      mai.allocationSize = memreq.size;
      mai.memoryTypeIndex = device_->physical_device.findMemoryTypeIndex(
          memreq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
      t.memory = (*device_)->allocateMemoryUnique(mai);
      (*device_)->bindImageMemory(*t.image, *t.memory, 0);

      vk::ImageViewCreateInfo vi;
      vi.image = *t.image;
      vi.viewType = desc.viewType;
      vi.format = desc.format;
      vi.components = {vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                       vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA};
      vi.subresourceRange =
          vk::ImageSubresourceRange(desc.aspect, 0, desc.mipLevels, 0,
                                    desc.arrayLayers);
      t.view = (*device_)->createImageViewUnique(vi);
      t.own = true;
    }
  } else {
    if (rt.buffers.size() != frames) rt.buffers.resize(frames);
    for (uint32_t i = 0; i < frames; ++i) {
      auto &b = rt.buffers[i];
      if (rd.imported) continue;
      if (b.own && b.buffer) continue;
      if (b.own) {
        b.buffer.reset();
        b.memory.reset();
      }
      const auto &desc = rd.buf;
      vk::BufferCreateInfo ci;
      ci.size = desc.size;
      ci.usage = desc.usage;
      ci.sharingMode = vk::SharingMode::eExclusive;
      b.buffer = (*device_)->createBufferUnique(ci);
      auto memreq = (*device_)->getBufferMemoryRequirements(*b.buffer);
      vk::MemoryAllocateInfo mai;
      mai.allocationSize = memreq.size;
      mai.memoryTypeIndex = device_->physical_device.findMemoryTypeIndex(
          memreq.memoryTypeBits, desc.memory);
      b.memory = (*device_)->allocateMemoryUnique(mai);
      (*device_)->bindBufferMemory(*b.buffer, *b.memory, 0);
      b.own = true;
    }
  }
}

inline void FrameGraph::ensureSwapchainRuntime(uint32_t resId) {
  if (!device_ || !swapchain_) return;
  auto &rd = resources_[resId];
  auto &rt = resourcesRT_[resId];
  auto &views = swapchain_->get_image_views();
  auto &images = swapchain_->get_images();
  rt.framesInFlight = uint32_t(images.size());
  rt.textures.resize(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    rt.textures[i].own = false;
    rt.textures[i].externalImage = images[i];
    rt.textures[i].externalView = views[i];
  }
  rd.tex.extent =
      vk::Extent3D(swapchain_->extent.width, swapchain_->extent.height, 1);
  rd.tex.format = swapchain_->image_format;
}

inline void FrameGraph::materializeDeviceObjects(
    const fg::CompiledFrameGraph &plan) {
  // Physical resources used by active passes.
  for (const auto &p : plan.passes)
    for (const auto &a : p.attachments) ensureResourceRuntime(a.resource);
  for (const auto &p : plan.passes)
    for (const auto &t : p.barrier.images) ensureResourceRuntime(t.resource);
  for (const auto &p : plan.passes)
    for (const auto &t : p.barrier.buffers) ensureResourceRuntime(t.resource);

  // Render passes + framebuffers.
  for (auto &p : const_cast<fg::CompiledFrameGraph &>(plan).passes) {
    if (!p.hasRenderPass) continue;
    auto it = renderPassCache_.find(p.rpSignature);
    if (it == renderPassCache_.end()) {
      std::vector<vk::AttachmentDescription> atts;
      for (const auto &ab : p.attachments) {
        const auto &rd = resources_[ab.resource];
        vk::AttachmentDescription ad;
        ad.flags = {};
        ad.format = rd.tex.format;
        ad.samples = rd.tex.samples;
        ad.loadOp = ab.op.loadOp;
        ad.storeOp = ab.op.storeOp;
        ad.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        ad.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        ad.initialLayout = ab.initialLayout;
        ad.finalLayout = ab.finalLayout;
        atts.push_back(ad);
      }
      std::vector<vk::AttachmentReference> colorRefs;
      vk::AttachmentReference depthRef;
      bool hasDepth = false;
      for (const auto &ab : p.attachments) {
        if (ab.isDepth) {
          depthRef = vk::AttachmentReference(
              uint32_t(&ab - p.attachments.data()),
              vk::ImageLayout::eDepthStencilAttachmentOptimal);
          hasDepth = true;
        } else {
          colorRefs.push_back(vk::AttachmentReference(
              uint32_t(&ab - p.attachments.data()),
              vk::ImageLayout::eColorAttachmentOptimal));
        }
      }
      vk::SubpassDescription subpass;
      subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
      subpass.colorAttachmentCount = uint32_t(colorRefs.size());
      subpass.pColorAttachments = colorRefs.data();
      subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;

      std::vector<vk::SubpassDependency> deps;
      vk::PipelineStageFlags srcStages{};
      vk::PipelineStageFlags dstStages{};
      vk::AccessFlags srcAccess{};
      vk::AccessFlags dstAccess{};
      for (const auto &ab : p.attachments) {
        srcStages |= ab.srcStages;
        dstStages |= ab.dstStages;
        srcAccess |= ab.srcAccess;
        dstAccess |= ab.dstAccess;
      }
      if (srcStages != vk::PipelineStageFlags{})
        deps.push_back(vk::SubpassDependency(
            VK_SUBPASS_EXTERNAL, 0, srcStages,
            vk::PipelineStageFlagBits::eColorAttachmentOutput |
                vk::PipelineStageFlagBits::eEarlyFragmentTests |
                vk::PipelineStageFlagBits::eLateFragmentTests,
            srcAccess,
            vk::AccessFlagBits::eColorAttachmentWrite |
                vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            {}));
      if (dstStages != vk::PipelineStageFlags{})
        deps.push_back(vk::SubpassDependency(
            0, VK_SUBPASS_EXTERNAL,
            vk::PipelineStageFlagBits::eColorAttachmentOutput |
                vk::PipelineStageFlagBits::eEarlyFragmentTests |
                vk::PipelineStageFlagBits::eLateFragmentTests,
            dstStages,
            vk::AccessFlagBits::eColorAttachmentWrite |
                vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            dstAccess, {}));

      vk::RenderPassCreateInfo rpci;
      rpci.attachmentCount = uint32_t(atts.size());
      rpci.pAttachments = atts.data();
      rpci.subpassCount = 1;
      rpci.pSubpasses = &subpass;
      rpci.dependencyCount = uint32_t(deps.size());
      rpci.pDependencies = deps.data();
      auto rp = (*device_)->createRenderPassUnique(
          rpci, device_->allocation_callbacks);
      p.renderPass = *rp;
      renderPassCache_.emplace(p.rpSignature, std::move(rp));
    } else {
      p.renderPass = *it->second;
    }

    // Framebuffers per physical variant.
    p.framebuffers.resize(p.framebufferCount);
    for (uint32_t v = 0; v < p.framebufferCount; ++v) {
      std::vector<vk::ImageView> views;
      views.reserve(p.attachments.size());
      for (const auto &ab : p.attachments) {
        const auto &rt = resourcesRT_[ab.resource];
        if (ab.resource < resources_.size() &&
            resources_[ab.resource].isSwapchain) {
          views.push_back(rt.textures[v].externalView);
        } else {
          uint32_t phys = rt.framesInFlight > 1 ? v % rt.framesInFlight : 0;
          views.push_back(rt.textures[phys].own
                              ? *rt.textures[phys].view
                              : rt.textures[phys].externalView);
        }
      }
      std::string sig = fg::framebufferSignature(
          p.renderPass, views, p.extent, 1);
      auto fit = framebufferCache_.find(sig);
      if (fit == framebufferCache_.end()) {
        vk::FramebufferCreateInfo fci;
        fci.renderPass = p.renderPass;
        fci.attachmentCount = uint32_t(views.size());
        fci.pAttachments = views.data();
        fci.width = p.extent.width;
        fci.height = p.extent.height;
        fci.layers = 1;
        auto fb = (*device_)->createFramebufferUnique(
            fci, device_->allocation_callbacks);
        p.framebuffers[v] = *fb;
        framebufferCache_.emplace(sig, std::move(fb));
      } else {
        p.framebuffers[v] = *fit->second;
      }
    }
  }

  // Command pools / buffers per frame slot.
  const uint32_t slotCount =
      swapchain_ ? std::min(framesInFlight_, std::max(1u, swapchain_->image_count))
                 : std::max(1u, framesInFlight_);
  pools_.resize(slotCount);
  cbs_.resize(slotCount);
  for (uint32_t s = 0; s < slotCount; ++s) {
    if (!pools_[s])
      pools_[s] = device_->createCommandPool();
    if (cbs_[s].size() < plan.passes.size()) {
      auto extra = device_->createCommandBuffers(
          pools_[s], uint32_t(plan.passes.size() - cbs_[s].size()));
      cbs_[s].insert(cbs_[s].end(), extra.begin(), extra.end());
    }
  }

  // Sync objects (only needed when presenting through a swapchain).
  if (swapchain_) {
    fences_.resize(slotCount);
    acquireSems_.resize(slotCount);
    finishedSems_.resize(slotCount);
    for (uint32_t s = 0; s < slotCount; ++s) {
      if (!fences_[s]) fences_[s] = device_->createFence();
      if (!acquireSems_[s]) acquireSems_[s] = device_->createSemaphore();
      if (!finishedSems_[s]) finishedSems_[s] = device_->createSemaphore();
    }
    imageInFlight_.assign(swapchain_->image_count, vk::Fence{});
  }
}

inline uint32_t FrameGraph::physicalIndexFor(uint32_t resId) const {
  if (resId >= resources_.size()) return 0;
  const auto &rd = resources_[resId];
  if (rd.isSwapchain) {
    return hasAcquired_ ? acquiredImageIndex_ : 0;
  }
  const uint32_t frames = rd.isTexture ? rd.tex.framesInFlight
                                       : rd.buf.framesInFlight;
  if (frames <= 1) return 0;
  return frameSlot_ % frames;
}

inline vk::Image FrameGraph::resolveImage(TextureHandle h) const {
  if (!h.valid() || h.id >= resourcesRT_.size()) return {};
  const auto &rt = resourcesRT_[h.id];
  const uint32_t phys = physicalIndexFor(h.id);
  if (phys >= rt.textures.size()) return {};
  const auto &t = rt.textures[phys];
  return t.own ? *t.image : t.externalImage;
}

inline vk::ImageView FrameGraph::resolveImageView(TextureHandle h) const {
  if (!h.valid() || h.id >= resourcesRT_.size()) return {};
  const auto &rt = resourcesRT_[h.id];
  const uint32_t phys = physicalIndexFor(h.id);
  if (phys >= rt.textures.size()) return {};
  const auto &t = rt.textures[phys];
  return t.own ? *t.view : t.externalView;
}

inline vk::Buffer FrameGraph::resolveBuffer(BufferHandle h) const {
  if (!h.valid() || h.id >= resourcesRT_.size()) return {};
  const auto &rt = resourcesRT_[h.id];
  const uint32_t phys = physicalIndexFor(h.id);
  if (phys >= rt.buffers.size()) return {};
  const auto &b = rt.buffers[phys];
  return b.own ? *b.buffer : b.externalBuffer;
}

inline vk::DeviceSize FrameGraph::bufferSize(BufferHandle h) const {
  if (!h.valid() || h.id >= resources_.size()) return 0;
  return resources_[h.id].buf.size;
}

// ---------------------------------------------------------------------------
// Frame execution
// ---------------------------------------------------------------------------

inline bool FrameGraph::beginFrame() {
  if (destroyed_ || !compiled_) return false;
  needsRecreate_ = false;

  if (!swapchain_) {
    frameSlot_ = frameIndex_ % std::max(1u, framesInFlight_);
    return true;
  }

  const uint32_t slotCount =
      std::min(framesInFlight_, std::max(1u, swapchain_->image_count));
  frameSlot_ = swapchain_->current_frame % slotCount;
  if (fences_.empty() || frameSlot_ >= fences_.size()) return false;

  (void)(*device_)->waitForFences(1, &fences_[frameSlot_], VK_TRUE, UINT64_MAX);
  auto result = (*device_)->acquireNextImageKHR(
      *swapchain_, UINT64_MAX, acquireSems_[frameSlot_], vk::Fence{},
      &acquiredImageIndex_);
  if (result == vk::Result::eErrorOutOfDateKHR) {
    needsRecreate_ = true;
    hasAcquired_ = false;
    return false;
  }
  if (result != vk::Result::eSuccess &&
      result != vk::Result::eSuboptimalKHR)
    throw std::runtime_error("failed to acquire swapchain image");
  hasAcquired_ = true;

  // Rebind per-image fence aliases like vkb::Present does.
  const vk::Fence slotFence = fences_[frameSlot_];
  for (auto &f : imageInFlight_)
    if (f == slotFence) f = vk::Fence{};
  imageInFlight_[acquiredImageIndex_] = slotFence;
  (void)(*device_)->resetFences(1, &fences_[frameSlot_]);
  return true;
}

inline vk::CommandBuffer FrameGraph::commandBufferFor(uint32_t passOrder) {
  if (frameSlot_ >= cbs_.size()) return {};
  auto &slotCbs = cbs_[frameSlot_];
  if (passOrder >= slotCbs.size()) return {};
  return slotCbs[passOrder];
}

inline vk::Framebuffer FrameGraph::framebufferFor(
    const fg::PassInstance &pass, uint32_t variant) {
  if (!pass.hasRenderPass || variant >= pass.framebuffers.size()) return {};
  return pass.framebuffers[variant];
}

inline void FrameGraph::record(const PassRecordExecutor &executor) {
  if (!device_ || !compiled_)
    throw std::runtime_error("FrameGraph::record requires a device");
  const auto &plan = *compiled_;

  auto recordOne = [&](uint32_t order) {
    const auto &pass = plan.passes[order];
    vk::CommandBuffer cb = commandBufferFor(order);
    if (!cb) throw std::runtime_error("missing command buffer for pass");
    cb.reset({});
    cb.begin(vk::CommandBufferBeginInfo{});

    if (!pass.barrier.empty()) {
      vk::PipelineStageFlags srcStage{};
      vk::PipelineStageFlags dstStage{};
      std::vector<vk::ImageMemoryBarrier> imgBars;
      imgBars.reserve(pass.barrier.images.size());
      for (const auto &t : pass.barrier.images) {
        srcStage |= t.srcStage;
        dstStage |= t.dstStage;
        vk::ImageMemoryBarrier bar;
        bar.srcAccessMask = t.srcAccess;
        bar.dstAccessMask = t.dstAccess;
        bar.oldLayout = t.oldLayout;
        bar.newLayout = t.newLayout;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image = resolveImage(TextureHandle{t.resource});
        bar.subresourceRange = t.range;
        imgBars.push_back(bar);
      }
      std::vector<vk::BufferMemoryBarrier> bufBars;
      bufBars.reserve(pass.barrier.buffers.size());
      for (const auto &t : pass.barrier.buffers) {
        srcStage |= t.srcStage;
        dstStage |= t.dstStage;
        vk::BufferMemoryBarrier bar;
        bar.srcAccessMask = t.srcAccess;
        bar.dstAccessMask = t.dstAccess;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.buffer = resolveBuffer(BufferHandle{t.resource});
        bar.offset = 0;
        bar.size = resources_[t.resource].buf.size;
        bufBars.push_back(bar);
      }
      if (srcStage == vk::PipelineStageFlags{})
        srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
      if (dstStage == vk::PipelineStageFlags{})
        dstStage = vk::PipelineStageFlagBits::eTopOfPipe;
      cb.pipelineBarrier(srcStage, dstStage, {}, nullptr, bufBars, imgBars);
    }

    if (pass.hasRenderPass) {
      uint32_t variant = 0;
      bool targetsSwapchain = false;
      for (const auto &a : pass.attachments)
        if (resources_[a.resource].isSwapchain) targetsSwapchain = true;
      if (targetsSwapchain) {
        // Present-target passes get a framebuffer per swapchain image.
        variant = acquiredImageIndex_;
      } else {
        // Ring-buffered attachments: variant = physical frame index.
        for (const auto &a : pass.attachments) {
          const auto &rd = resources_[a.resource];
          if (rd.isTexture && rd.tex.framesInFlight > 1) {
            variant = frameSlot_ % rd.tex.framesInFlight;
            break;
          }
        }
      }
      vk::Framebuffer fb = framebufferFor(pass, variant);
      vk::RenderPassBeginInfo rpbi;
      rpbi.renderPass = pass.renderPass;
      rpbi.framebuffer = fb;
      rpbi.renderArea = vk::Rect2D(vk::Offset2D(0, 0), pass.extent);
      rpbi.clearValueCount = uint32_t(pass.clears.size());
      rpbi.pClearValues = pass.clears.data();
      cb.beginRenderPass(rpbi, vk::SubpassContents::eInline);
    }

    FrameGraphPassContext ctx{FrameSlot{frameSlot_}, order, this, cb};
    if (passes_[pass.passId].record) passes_[pass.passId].record(ctx);

    if (pass.hasRenderPass) cb.endRenderPass();
    cb.end();
  };

  if (executor) {
    for (const auto &layer : plan.layers) executor(layer, recordOne);
  } else {
    for (uint32_t order = 0; order < plan.passes.size(); ++order)
      recordOne(order);
  }
}

inline void FrameGraph::submit() {
  if (!device_ || !compiled_) return;
  const auto &plan = *compiled_;
  std::vector<vk::CommandBuffer> cbs;
  cbs.reserve(plan.passes.size());
  for (uint32_t order = 0; order < plan.passes.size(); ++order)
    cbs.push_back(commandBufferFor(order));
  if (cbs.empty()) return;

  vk::SubmitInfo si;
  std::vector<vk::Semaphore> waitSems;
  std::vector<vk::PipelineStageFlags> waitStages;
  std::vector<vk::Semaphore> signalSems;
  if (swapchain_ && hasAcquired_) {
    waitSems.push_back(acquireSems_[frameSlot_]);
    waitStages.push_back(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    signalSems.push_back(finishedSems_[frameSlot_]);
  }
  si.waitSemaphoreCount = uint32_t(waitSems.size());
  si.pWaitSemaphores = waitSems.data();
  si.pWaitDstStageMask = waitStages.data();
  si.commandBufferCount = uint32_t(cbs.size());
  si.pCommandBuffers = cbs.data();
  si.signalSemaphoreCount = uint32_t(signalSems.size());
  si.pSignalSemaphores = signalSems.data();

  vk::Fence fence{};
  if (swapchain_ && hasAcquired_) fence = fences_[frameSlot_];
  vk::Queue graphicsQueue = device_->getQueue(QueueType::graphics);
  (void)graphicsQueue.submit(1, &si, fence);
}

inline void FrameGraph::present() {
  if (!device_ || !compiled_ || !swapchain_ || !hasAcquired_) return;
  vk::PresentInfoKHR pi;
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &finishedSems_[frameSlot_];
  vk::SwapchainKHR swapChains[] = {swapchain_->instance};
  pi.swapchainCount = 1;
  pi.pSwapchains = swapChains;
  pi.pImageIndices = &acquiredImageIndex_;
  vk::Queue presentQueue = device_->getQueue(QueueType::present);
  vk::Result result = presentQueue.presentKHR(&pi);
  hasAcquired_ = false;
  if (result == vk::Result::eErrorOutOfDateKHR ||
      result == vk::Result::eSuboptimalKHR) {
    needsRecreate_ = true;
  } else if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to present swapchain image");
  }
  const uint32_t slotCount =
      std::min(framesInFlight_, std::max(1u, swapchain_->image_count));
  swapchain_->current_frame = (swapchain_->current_frame + 1) % slotCount;
  ++frameIndex_;

  // Update per-physical-image layout knowledge for next frame's planning.
  for (const auto &p : compiled_->passes) {
    if (!p.hasRenderPass) continue;
    for (const auto &ab : p.attachments) {
      if (ab.resource >= resourceLastLayout_.size()) continue;
      auto &layouts = resourceLastLayout_[ab.resource];
      const auto &rd = resources_[ab.resource];
      uint32_t phys = rd.isSwapchain ? acquiredImageIndex_
                                     : physicalIndexFor(ab.resource);
      if (phys >= layouts.size()) layouts.resize(phys + 1,
                                                vk::ImageLayout::eUndefined);
      layouts[phys] = ab.finalLayout;
    }
  }
}

inline void FrameGraph::destroy() {
  if (destroyed_) return;
  if (device_) {
    for (auto &cbVec : cbs_) cbVec.clear();
    for (auto &pool : pools_)
      if (pool) (*device_)->destroyCommandPool(pool,
                                               device_->allocation_callbacks);
    for (auto &f : fences_)
      if (f) (*device_)->destroyFence(f);
    for (auto &s : acquireSems_)
      if (s) (*device_)->destroySemaphore(s);
    for (auto &s : finishedSems_)
      if (s) (*device_)->destroySemaphore(s);
  }
  pools_.clear();
  cbs_.clear();
  fences_.clear();
  acquireSems_.clear();
  finishedSems_.clear();
  imageInFlight_.clear();
  renderPassCache_.clear();
  framebufferCache_.clear();
  resourcesRT_.clear();
  compiled_.reset();
  destroyed_ = true;
}

} // namespace vkb
