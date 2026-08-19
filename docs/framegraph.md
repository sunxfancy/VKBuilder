# FrameGraph（渲染图）设计说明

> 头文件：`include/vkbuilder/framegraph.hpp`（**可选扩展**，需显式引入；
> 不包含在 `vkbuilder.hpp` 里，避免核心库用户承担额外编译成本）。

## 1. 动机

经典的 `vkb::Present` 录制模型有以下问题：

- **依赖靠手工顺序**：pass 之间的先后关系由调用者硬编码（如 EVEngine 里
  shadow → GBuffer → scene color → present 的顺序写死在 `recordPending*Passes`
  里），改动渲染管线时要手动重排。
- **隐式 layout 管理**：`GenericImage::beginColorAttachment() /
  endSampledLayout()` 只是改写 CPU 侧的 `currentLayout` 单值，真正的 GPU 转换
  靠 render pass 的 `initialLayout/finalLayout` 隐式完成。同一个 image 只能记
  住一种状态，跨帧、跨线程都容易出错。
- **录制完全串行**：`Present` 每帧只有一个 command buffer，所有 pass 在同一
  线程按顺序录进同一个 CB。
- **barrier 不合并**：`setLayout()` 一次只插一条 image barrier，多资源转换没有
  批量合并。

Frostbite / UE5 / Unity SRP 的 render graph 用显式资源依赖换取三类能力：

1. 自动推导 pass 顺序与并行性；
2. 自动生成并合并 barrier（layout 转换 + stage/access 内存依赖）；
3. pass 复用与裁剪（culling），以及稳定的 VkRenderPass/Framebuffer 缓存。

## 2. 设计目标

- 纯头文件、C++17，无新增第三方依赖；
- 沿用 VKB 的流式 builder 风格（`addPass(...).colorAttachment(...).record(...)`）；
- **声明式**：pass 声明它读/写哪些资源、以什么 layout/stage/access 使用；
- 每帧重建图（draw list 会变），但物理资源、VkRenderPass、Framebuffer、
  command buffer 全部按签名缓存复用；
- 规划（planning）不依赖 GPU，可单测；
- 并行录制作为一等公民：同一 layer 的 pass 互相独立，可并发录制。

## 3. 核心概念

### 3.1 资源

| 类型 | 说明 |
|------|------|
| `TextureHandle` / `BufferHandle` | 不透明句柄，跨帧稳定 |
| `TextureDesc` / `BufferDesc` | 格式、尺寸、usage、`framesInFlight`（环形缓冲份数）、`afterLayout` |
| `createTexture` / `createBuffer` | 图内拥有，按需创建物理对象 |
| `importTexture` / `importBuffer` / `importSwapchain` | 外部资源（引擎自己管理的深度图、swapchain 图像），图不销毁 |
| `markOutput` | 声明输出资源，供 culling 使用；swapchain 自动视为输出 |

环形资源（`framesInFlight > 1`）会分配多份物理 image/buffer，按帧槽取模使用，
等价于 EVEngine 现在的 `shadowMaps[2]` / `gbufferSlots[2]` 双缓冲模式。

### 3.2 访问声明

每个 pass 用 `read / write / colorAttachment / depthAttachment / sample`
声明资源访问。`TextureUsage` / `BufferUsage` 描述 layout + stage + access，
`vkb::access::` 提供常用预设（`sample`、`colorAttachment`、`storageWrite`、
`transferWrite`、`present` 等）。

```cpp
vkb::FrameGraph graph{&device, 2};          // framesInFlight = 2
auto scene = graph.createTexture("sceneColor", texDesc);
graph.markOutput(scene);

graph.addPass("gbuffer")
    .colorAttachment(scene, vkb::AttachmentOp::clearColor(0, 0, 0, 1))
    .sample(white)
    .record([](vkb::FrameGraphPassContext &ctx) {
        // 图已经 beginRenderPass；这里只负责 draw
        auto &cb = ctx.commandBuffer();
        cb.bindPipeline(...);
        cb.draw(...);
    });

graph.addPass("compose")
    .sample(scene)
    .record([](vkb::FrameGraphPassContext &ctx) { ... });

graph.compile();
graph.execute();      // acquire -> record -> submit -> present
```

`FrameGraphPassContext` 提供 `commandBuffer()`、`view(handle)`、
`image(handle)`、`buffer(handle)`、`extent()`、`renderPass()`、
`frameSlot()`。声明了 attachment 的 pass，图会自动 `beginRenderPass` /
`endRenderPass`，用户回调里只写 draw 命令。

## 4. 编译期规划（compilePlan，无 GPU 依赖）

`compile()` 每次调用都会重新规划，但设备对象全部走缓存：

1. **访问序列**：按 pass 注册顺序收集每个资源的访问列表。
2. **依赖边**：对每个访问，向前找到最近一个“冲突”的前驱并建边。冲突定义：
   任一方是写，或纹理 layout 不同。非相邻冲突（W、R1、R2 中的 W→R2）也会建边，
   保证 read-after-write 在提交顺序上正确。由于边总是沿注册顺序，跨 pass 环
   在结构上不可能出现（仍保留防御性检测）。
3. **拓扑排序**：Kahn 算法，按 pass id 最小优先保证确定性。
4. **Culling**：从输出资源反向可达性分析；不（间接）喂给任何输出的 pass 被剔除。
5. **Layer**：最长路径分层；同 layer 的 pass 之间没有依赖边，可并行录制。
6. **Barrier 规划**：
   - 附件转换折进 render pass 的 `initialLayout/finalLayout`（如
     attachment → sampled 直接 `finalLayout = ShaderReadOnlyOptimal`，零显式 barrier）；
   - 其余转换按“最近冲突前驱”生成，同一 pass 的所有转换合并成**一条**
     `vkCmdPipelineBarrier`（多个 image/buffer barrier 一起提交）；
   - 跨帧同步：每个物理 image 记录上一帧结束时的 layout（`resourceLastLayout_`），
     帧首访问用它做 seed，load=Load 的附件可以跨帧保内容；
   - 同 layout 的 write→read 只做 access 内存依赖，不做 layout 转换。
7. **RP 签名**：attachment 描述（format/samples/load/store/initial/final）+
   EXTERNAL 依赖的 stage/access 序列化后作为 `VkRenderPass` 缓存键；
   framebuffer 以（RP + views + extent）为键。相同图结构跨帧复用，零重建。

## 5. 并行录制模型

`record()` 接受 `PassRecordExecutor`：

```cpp
using PassRecordExecutor = std::function<void(
    const std::vector<uint32_t> &layerPasses,     // 该 layer 的 pass 下标
    const std::function<void(uint32_t)> &recordOne)>;
```

默认串行执行；传入线程池实现即可让每个 layer 内的 pass 并发录制：

```cpp
graph.record([](const std::vector<uint32_t> &layer,
                const std::function<void(uint32_t)> &one) {
    std::vector<std::thread> ts;
    for (uint32_t i : layer) ts.emplace_back(one, i);
    for (auto &t : ts) t.join();
});
```

正确性依据：

- 每个 pass 独占自己的 command buffer（pool 按帧槽分配，CB 按 pass 下标复用）；
- 同 layer 内没有依赖边，也没有需要跨 pass 的 barrier；
- 所有 barrier 都写在“消费方”pass 的 CB 开头，且提交顺序 = 拓扑顺序，
  因此同一队列上语义正确；
- 一次 `vkQueueSubmit` 按拓扑顺序提交全部 CB。

### 线程契约

- 图构建与 `compile()` 在主线程；
- `record()` 期间 pass 回调可能并发运行，回调**只能**写自己的 CB 和只读数据；
- descriptor set 更新必须在 `record()` 之前完成（`vkUpdateDescriptorSets`
  默认非线程安全），或使用带锁的 pool；
- 帧槽索引通过 `ctx.frameSlot()` 获取，用于索引每帧一份的 UBO / 顶点缓冲。

## 6. 与现有 API 的关系

- 不修改 `vkb::Present` / `GenericImage` 的行为，老代码继续可用；
- `FrameGraph` 内部的 image/buffer 是独立实现（不再依赖
  `GenericImage::currentLayout` 单值状态），因此可以按物理帧份数正确跟踪 layout；
- 需要 pipeline 时用 `graph.renderPassOf("passName")` 取 VkRenderPass，
   与 `PipelineBuilder` 产物一致；
- swapchain 集成：`graph.importSwapchain(swapchain)` 后 `execute()` 内部完成
  acquire/submit/present（复刻 `vkb::Present` 的 fence/semaphore 与
  out-of-date 处理）。不 import swapchain 时只做 record + submit。

## 7. 限制与后续路线

当前版本（v1）：

- 单一 graphics queue；无 async compute / transfer 队列与 timeline semaphore；
- 每个 pass 一个 primary CB；未使用 secondary CB（后续可在同一 layer 内把
  draw 拆分到多个 secondary CB，进一步细分并行度）；
- 无 transient 内存别名（transient 附件可与同帧其他附件复用同一块内存）；
- 无 input attachment 特殊处理（按采样读处理）；
- 无自动后处理链/子图合并（subpass merging）。

建议的后续增量：

1. 内存别名：transient 资源按生命周期区间做 offset 复用（Frostbite 式）；
2. 多队列：按 pass 的 stage 自动分到 graphics/compute/transfer 队列，
   跨队列用 semaphore + queue-family ownership transfer；
3. secondary CB：同 layer 的 pass 内部多线程记录 secondary CB；
4. descriptor 版本化：让每个 frame slot 自动生成描述符副本，允许录制期间
   安全更新；
5. 与 EVEngine 的 `Graphics.cpp` 迁移：shadow/GBuffer/scene color 改成
   声明式 pass，`recordPending*Passes` 全部删除。

## 8. 测试

`test/framegraph/main.cpp` 是纯逻辑单测（不需要 GPU、不需要窗口）：

- 链式依赖与 attachment→sampled 折叠（零显式 barrier）；
- barrier 合并（一条 prologue barrier 同时包含 image + buffer 转换）；
- layer 并行性 + 线程化 executor 契约；
- culling；
- load=Load 的 WAW 保布局；
- 同 pass 内写+读；
- 重编译 RP 签名稳定；
- 环形附件 framebuffer 份数。

设备冒烟测试（真实 Vulkan device 上 record/submit）默认关闭，显式设置
`VKB_FRAMEGRAPH_DEVICE_TEST=1` 才会尝试，失败时自动 SKIP。
