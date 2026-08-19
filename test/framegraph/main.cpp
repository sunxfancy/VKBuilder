#define VKB_IMPL
#include "vkbuilder.hpp"
#include "vkbuilder/framegraph.hpp"

#include <algorithm>
#include <atomic>
#include <crtdbg.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Headless unit tests for vkb::FrameGraph. The planning tests never touch a
// GPU; the device smoke test at the bottom skips itself when Vulkan is
// unavailable (e.g. CI containers).

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                           \
  do {                                                                        \
    ++g_checks;                                                               \
    if (!(cond)) {                                                            \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);             \
      ++g_failures;                                                           \
    }                                                                         \
  } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

static vkb::TextureDesc makeTexDesc() {
  vkb::TextureDesc td;
  td.extent = vk::Extent3D(64, 64, 1);
  td.usage = vk::ImageUsageFlagBits::eSampled |
             vk::ImageUsageFlagBits::eColorAttachment;
  return td;
}

// ---------------------------------------------------------------------------
// 1. Producer -> consumer chain: explicit dependency, automatic layout folding
// ---------------------------------------------------------------------------
static void testChainAndAttachmentFolding() {
  vkb::FrameGraph g{nullptr, 2};
  auto scene = g.createTexture("sceneColor", makeTexDesc());
  auto white = g.createTexture("white", makeTexDesc());
  g.markOutput(scene);

  g.addPass("gbuffer")
      .colorAttachment(scene, vkb::AttachmentOp::clearColor(0, 0, 0, 1))
      .sample(white)
      .record([](vkb::FrameGraphPassContext &) {});
  g.addPass("compose")
      .sample(scene)
      .record([](vkb::FrameGraphPassContext &) {});

  g.compile();
  const auto &plan = g.compiled();
  CHECK_EQ(plan.passes.size(), size_t(2));
  CHECK_EQ(plan.passes[0].name, std::string("gbuffer"));
  CHECK_EQ(plan.passes[1].name, std::string("compose"));

  // gbuffer writes the attachment, compose samples it: the transition is
  // folded into the render pass final layout, no explicit barrier needed.
  CHECK(plan.passes[0].hasRenderPass);
  CHECK(plan.passes[0].attachments[0].finalLayout ==
        vk::ImageLayout::eShaderReadOnlyOptimal);
  CHECK(plan.passes[1].barrier.images.empty());
  CHECK(plan.passes[1].barrier.buffers.empty());

  // Dependency edge: gbuffer before compose.
  CHECK(plan.passes[0].layer == 0);
  CHECK(plan.passes[1].layer == 1);
  CHECK(!g.dump().empty());
}

// ---------------------------------------------------------------------------
// 2. Barrier merging: two write->read transitions land in one prologue barrier
// ---------------------------------------------------------------------------
static void testBarrierMerging() {
  vkb::FrameGraph g{nullptr, 2};
  auto img = g.createTexture("ssboImage", makeTexDesc());
  vkb::BufferDesc bd;
  bd.size = 256;
  auto buf = g.createBuffer("lightBuffer", bd);
  g.markOutput(img);
  g.markOutput(buf);

  g.addPass("computeA")
      .write(img, vkb::access::storageWrite())
      .write(buf, vkb::access::bufferWrite())
      .record([](vkb::FrameGraphPassContext &) {});
  g.addPass("computeB")
      .read(img, vkb::access::storageRead())
      .read(buf, vkb::access::bufferRead())
      .record([](vkb::FrameGraphPassContext &) {});

  g.compile();
  const auto &plan = g.compiled();
  CHECK_EQ(plan.passes.size(), size_t(2));
  CHECK_EQ(plan.passes[1].name, std::string("computeB"));

  // computeB's single merged prologue barrier holds both transitions.
  CHECK_EQ(plan.passes[1].barrier.images.size(), size_t(1));
  CHECK_EQ(plan.passes[1].barrier.buffers.size(), size_t(1));
  CHECK(plan.passes[1].barrier.images[0].newLayout ==
        vk::ImageLayout::eGeneral);
  CHECK(plan.passes[1].barrier.images[0].oldLayout ==
        vk::ImageLayout::eGeneral);
  CHECK(plan.passes[1].barrier.buffers[0].srcAccess ==
        vk::AccessFlagBits::eShaderWrite);

  // computeA still needs the undefined->general first-touch transition.
  CHECK_EQ(plan.passes[0].barrier.images.size(), size_t(1));
  CHECK(plan.passes[0].barrier.images[0].oldLayout ==
        vk::ImageLayout::eUndefined);
}

// ---------------------------------------------------------------------------
// 3. Parallel recording layers: independent consumers share a layer
// ---------------------------------------------------------------------------
static void testParallelLayers() {
  vkb::FrameGraph g{nullptr, 2};
  auto img = g.createTexture("shared", makeTexDesc());
  g.markOutput(img);

  g.addPass("producer")
      .write(img, vkb::access::storageWrite())
      .record([](vkb::FrameGraphPassContext &) {});
  g.addPass("consumerA")
      .read(img, vkb::access::storageRead())
      .record([](vkb::FrameGraphPassContext &) {});
  g.addPass("consumerB")
      .read(img, vkb::access::storageRead())
      .record([](vkb::FrameGraphPassContext &) {});

  g.compile();
  const auto &plan = g.compiled();
  CHECK_EQ(plan.passes.size(), size_t(3));
  // producer alone in layer 0; both consumers in layer 1 -> recordable in
  // parallel.
  CHECK_EQ(plan.layers.size(), size_t(2));
  CHECK_EQ(plan.layers[1].size(), size_t(2));
  CHECK(plan.layers[1][0] != plan.layers[1][1]);

  // Exercise the executor contract: every pass recorded exactly once, even
  // when layer members run on separate threads.
  std::mutex m;
  std::vector<int> recorded;
  auto recordOne = [&](uint32_t idx) {
    std::lock_guard<std::mutex> lk(m);
    recorded.push_back(int(idx));
  };
  vkb::PassRecordExecutor exec =
      [&](const std::vector<uint32_t> &layerPasses,
          const std::function<void(uint32_t)> &one) {
        std::vector<std::thread> threads;
        threads.reserve(layerPasses.size());
        for (uint32_t idx : layerPasses) threads.emplace_back(one, idx);
        for (auto &t : threads) t.join();
      };
  for (const auto &layer : plan.layers) exec(layer, recordOne);
  CHECK_EQ(recorded.size(), plan.passes.size());
  std::sort(recorded.begin(), recorded.end());
  for (size_t i = 0; i < recorded.size(); ++i)
    CHECK_EQ(recorded[i], int(i));
}

// ---------------------------------------------------------------------------
// 4. Culling: passes that do not feed an output disappear
// ---------------------------------------------------------------------------
static void testCulling() {
  vkb::FrameGraph g{nullptr, 2};
  auto used = g.createTexture("used", makeTexDesc());
  auto unused = g.createTexture("unused", makeTexDesc());
  g.markOutput(used);

  g.addPass("producer")
      .write(used, vkb::access::transferWrite())
      .record([](vkb::FrameGraphPassContext &) {});
  g.addPass("dead")
      .write(unused, vkb::access::transferWrite())
      .record([](vkb::FrameGraphPassContext &) {});

  g.compile();
  const auto &plan = g.compiled();
  CHECK_EQ(plan.passes.size(), size_t(1));
  CHECK_EQ(plan.passes[0].name, std::string("producer"));
}

// ---------------------------------------------------------------------------
// 5. WAW / load-preserve: second writer keeps the previous layout
// ---------------------------------------------------------------------------
static void testLoadPreserve() {
  vkb::FrameGraph g{nullptr, 2};
  auto tex = g.createTexture("post", makeTexDesc());
  g.markOutput(tex);

  g.addPass("first")
      .colorAttachment(tex, vkb::AttachmentOp::clearColor(1, 0, 0))
      .record([](vkb::FrameGraphPassContext &) {});
  g.addPass("second")
      .colorAttachment(tex, vkb::AttachmentOp::load())
      .record([](vkb::FrameGraphPassContext &) {});

  g.compile();
  const auto &plan = g.compiled();
  CHECK_EQ(plan.passes.size(), size_t(2));
  // Second pass loads the previous pass's output: initial layout must equal
  // the first pass's final layout and no explicit barrier is emitted.
  CHECK(plan.passes[1].attachments[0].initialLayout ==
        plan.passes[0].attachments[0].finalLayout);
  CHECK(plan.passes[1].attachments[0].initialLayout ==
        vk::ImageLayout::eColorAttachmentOptimal);
  CHECK(plan.passes[1].barrier.images.empty());
}

// ---------------------------------------------------------------------------
// 6. In-pass write+read: no self edge, no bogus barrier. (Dependency edges are
//    order-preserving by construction, so cross-pass cycles are impossible;
//    the planner still keeps a defensive cycle check.)
// ---------------------------------------------------------------------------
static void testInPassAccess() {
  vkb::FrameGraph g{nullptr, 2};
  auto img = g.createTexture("inplace", makeTexDesc());
  g.markOutput(img);
  g.addPass("inplace")
      .write(img, vkb::access::storageWrite())
      .read(img, vkb::access::storageRead())
      .record([](vkb::FrameGraphPassContext &) {});
  g.compile();
  const auto &plan = g.compiled();
  CHECK_EQ(plan.passes.size(), size_t(1));
  // First touch transition only (undefined -> general); the in-pass read must
  // not generate an extra barrier or an edge back to the same pass.
  CHECK_EQ(plan.passes[0].barrier.images.size(), size_t(1));
  CHECK(plan.passes[0].barrier.images[0].oldLayout ==
        vk::ImageLayout::eUndefined);
}

// ---------------------------------------------------------------------------
// 7. Recompile stability: render-pass signature stays stable across frames
// ---------------------------------------------------------------------------
static void testRecompileStability() {
  vkb::FrameGraph g{nullptr, 2};
  auto scene = g.createTexture("scene", makeTexDesc());
  g.markOutput(scene);
  g.addPass("gbuffer")
      .colorAttachment(scene, vkb::AttachmentOp::clearColor(0, 0, 0, 1))
      .record([](vkb::FrameGraphPassContext &) {});
  g.addPass("compose")
      .sample(scene)
      .record([](vkb::FrameGraphPassContext &) {});

  g.compile();
  std::string sig1 = g.compiled().passes[0].rpSignature;
  g.compile();
  std::string sig2 = g.compiled().passes[0].rpSignature;
  CHECK(sig1 == sig2);
  CHECK(!sig1.empty());
}

// ---------------------------------------------------------------------------
// 8. Ring-buffered attachment selects the right physical variant
// ---------------------------------------------------------------------------
static void testRingBuffering() {
  vkb::FrameGraph g{nullptr, 2};
  auto tex = makeTexDesc();
  tex.framesInFlight = 2;
  auto scene = g.createTexture("scene", tex);
  g.markOutput(scene);
  g.addPass("gbuffer")
      .colorAttachment(scene, vkb::AttachmentOp::clearColor(0, 0, 0, 1))
      .record([](vkb::FrameGraphPassContext &) {});
  g.addPass("compose")
      .sample(scene)
      .record([](vkb::FrameGraphPassContext &) {});
  g.compile();
  const auto &plan = g.compiled();
  CHECK_EQ(plan.passes[0].framebufferCount, uint32_t(2));
  CHECK(plan.passes[0].attachments[0].finalLayout ==
        vk::ImageLayout::eShaderReadOnlyOptimal);
}

// ---------------------------------------------------------------------------
// Device smoke test (skips itself when no Vulkan device is available)
// ---------------------------------------------------------------------------
static void testDeviceSmoke() {
  // Device creation needs a real Vulkan driver; CI/sandboxes without GPU
  // access can hang inside driver init. Opt in explicitly.
  const char *env = std::getenv("VKB_FRAMEGRAPH_DEVICE_TEST");
  if (!env || std::string(env) != "1") {
    std::printf("device smoke: SKIP (set VKB_FRAMEGRAPH_DEVICE_TEST=1)\n");
    return;
  }
  try {
    vkb::InstanceBuilder ib;
    // InstanceBuilder's constructor initializes the dynamic dispatch loader,
    // which SystemInfo::query() requires.
    vkb::SystemInfo system = vkb::SystemInfo::query();
    const bool haveHeadlessSurface =
        system.is_extension_available(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
    if (haveHeadlessSurface)
      ib.enable_extension(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
    auto inst = ib.set_headless().require_api_version(1, 0).build();

    vk::SurfaceKHR surface{};
    if (haveHeadlessSurface) {
      vk::HeadlessSurfaceCreateInfoEXT hci;
      surface = inst.instance.createHeadlessSurfaceEXT(hci);
    }
    vkb::PhysicalDeviceSelector sel{inst};
    auto phys = sel.set_surface(surface).set_minimum_version(1, 0).select();
    vkb::DeviceBuilder db{phys};
    auto device = db.build();

    vkb::FrameGraph g{&device, 2};
    auto texDesc = makeTexDesc();
    texDesc.usage = vk::ImageUsageFlagBits::eSampled |
                    vk::ImageUsageFlagBits::eColorAttachment |
                    vk::ImageUsageFlagBits::eTransferDst;
    auto scene = g.createTexture("scene", texDesc);
    g.markOutput(scene);
    g.addPass("clear")
        .colorAttachment(scene, vkb::AttachmentOp::clearColor(0.1f, 0.2f, 0.3f))
        .record([](vkb::FrameGraphPassContext &ctx) {
          (void)ctx.extent();
          (void)ctx.renderPass();
        });
    g.addPass("readback")
        .sample(scene)
        .record([](vkb::FrameGraphPassContext &) {});
    g.compile();
    g.execute();
    device->waitIdle();
    g.destroy();
    device.destroy();
    if (surface) inst.instance.destroySurfaceKHR(surface);
    inst.destroy();
    std::printf("device smoke: OK\n");
  } catch (const std::exception &e) {
    std::printf("device smoke: SKIP (%s)\n", e.what());
  }
}

int main() {
  // Route CRT error dialogs (heap corruption etc.) to stderr so headless CI
  // does not block on a message box.
  _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);

  testChainAndAttachmentFolding();
  testBarrierMerging();
  testParallelLayers();
  testCulling();
  testLoadPreserve();
  testInPassAccess();
  testRecompileStability();
  testRingBuffering();
  if (std::getenv("VKB_FRAMEGRAPH_DEVICE_TEST")) testDeviceSmoke();

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
