# Phase 设计目标与实现

## phase0

设计目标：

- 保留原始多模态处理链路
- 只增加统一性能打点
- 作为精度和性能基线

实现方案：

- `VLLM_ASCEND_API_OPT_PHASE=0`
- 不启用 phase1/2/3 的行为变更
- 只启用性能 probe patch

链路特点：

- API server 负责本地图像读取
- API server 负责 HF processor 图像预处理
- API server 负责 mm cache / shm copy
- worker 维持原始推理消费方式

## phase1

设计目标：

- 降低本地图像 hash 计算成本
- 避免每次都基于图像内容做重 hash

实现方案：

- `VLLM_ASCEND_API_OPT_PHASE=1`
- 在 `bundle/vllm_phase2_plugin/patches/phase1.py` 中，把本地图像身份切换到稳定 UUID/hash
- UUID 由本地路径和文件修改时间派生

链路特点：

- 除 hash 行为外，其余链路仍基本保持 phase0 语义
- 主要观察 `ProcessorInputs.get_mm_hashes` 和 phase1 repeat-hit 测试结果

## phase2

设计目标：

- 把 `image_url` 对应的本地路径透传到 engine core / worker
- 验证 worker 侧具备按 vit-DP 分工读取图片的能力
- 在不改变 API server 主处理语义的前提下先打通正确性链路

实现方案：

- `VLLM_ASCEND_API_OPT_PHASE=2`
- API server 继续保留原多模态处理主路径
- 附加透传本地图片路径信息到 engine core / worker
- worker 按 vit-DP 负载切片，只读取自己负责的图片子集

链路特点：

- 这个阶段以“链路正确性”和“透传打通”为主
- 不以最终性能最优为目标

## phase3

设计目标：

- 真正把 API server 的图像读取与图像处理下沉到 worker
- 让 API server 尽量只保留请求编排、tokenizer、cache 语义
- 固化 `phase3 direct encode` 作为唯一继续演进的 phase3 主方案

实现方案：

- `VLLM_ASCEND_API_OPT_PHASE=3`
- API server 不再做本地图像主读取
- API server 不再做本地图像 HF 预处理主路径
- worker 负责：
  - 按 vit-DP 切分 multimodal 数据
  - 读取本地图片
  - 执行 HF preprocess
  - direct encode 主路径
  - 对当前不适合 direct encode 的模态执行稳定 fallback

链路特点：

- API 热点应明显下降
- 热点会迁移到 tp worker
- `phase3 direct encode` 当前已经支持：
  - `local_path`
  - `http`
- 当前主要剩余问题是多图精度，不是服务稳定性

## 不再继续维护的旧实验路径

下列历史实验路径不再作为当前目录的推荐方案：

- `phase3-local-only`
- `phase3-local-plus`

原因：

- 它们不再是后续继续开发的主线
- 它们会干扰接手人对当前正确方案的理解
- 当前已经确认 `phase3 direct encode` 才是需要继续打磨的主方案
