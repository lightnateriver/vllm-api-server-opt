# API Server 下沉总体方案设计文档

## 1. 文档目的

本文档用于完整描述当前多模态服务优化方案，重点解释：

- 为什么要做 API server 下沉
- `phase0 ~ phase3` 每个阶段分别在解决什么问题
- `phase3 direct encode` 为什么是当前唯一继续演进的主方案
- 当前代码里具体修改了哪些层、哪些参数和哪些数据流
- 当前已经支持哪些输入形式
- 后续接手任务时应该从哪里继续推进

本文档是 `0428/` 目录下最完整的方案说明，适合作为新任务接手时的首要设计参考。

## 2. 背景

### 2.1 业务背景

目标模型为 `Qwen3.5-4B` 多模态服务，典型输入特征为：

- 文本约 `10k` token
- 图片约 `40` 张
- 部署模式以 `tp4`、`--enforce-eager`、`mm shm cache` 为主

原始 vLLM 多模态链路在处理本地图片输入时，API server 会承担大量 CPU 侧工作：

- 本地文件路径解析
- 图片文件读取
- 图片解码
- HF processor 预处理
- 多模态 hash 计算
- multimodal cache / shm 相关处理

当单请求里图片数较多时，API server 会成为 TTFT 和端到端时延的热点区域。

### 2.2 工程约束

本项目始终遵守以下工程约束：

1. 不修改 site-packages 中原始安装版 `vllm` / `vllm-ascend`
2. 不要求重新打 wheel 或二次安装框架
3. 所有行为变化通过插件注册和运行期 patch 实现
4. phase 行为由环境变量切换，便于回退和逐阶段验证

## 3. 设计目标

### 3.1 总体目标

总体目标是逐步把多模态请求中的重型图像处理逻辑从 API server 下沉到 worker，
使 API server 尽量只保留轻量控制面语义：

- 请求解析
- tokenizer
- 请求调度
- cache 语义

同时让 worker 接管：

- 图像读取
- 图像预处理
- multimodal encode 主路径

### 3.2 当前阶段目标

当前 `0428` 版本的目标已经明确收敛为：

1. 保留 `phase0 ~ phase3` 的渐进式验证链
2. 固化 `phase3 direct encode` 为唯一继续演进的 `phase3` 主方案
3. 明确支持：
   - `local_path`
   - `http`
4. 修复历史上 `phase3` 在视频本地输入下的稳定性问题
5. 将剩余问题收敛为多图精度优化，而非服务稳定性问题

### 3.3 非目标

当前版本不再追求：

- 继续维护 `phase3-local-only`
- 继续维护 `phase3-local-plus`
- 在当前目录中保留历史试验分支作为正式入口
- 在没有稳定精度前贸然继续扩展更多并发/性能变体

## 4. 方案演进概览

### 4.1 演进思想

`phase0 ~ phase3` 的设计思路不是一次性推翻原始链路，而是分阶段验证：

1. 先建立基线
2. 再优化 hash
3. 再透传本地路径
4. 最后真正下沉图像主处理链路

### 4.2 Phase 划分

#### phase0

目标：

- 完全保留原始多模态处理主链路
- 只增加统一性能打点
- 作为精度和性能基线

#### phase1

目标：

- 降低本地图片 hash 计算成本
- 避免每次都基于完整图像内容做重 hash

#### phase2

目标：

- 把本地图片路径透传到 engine core / worker
- 验证 worker 侧按 vit-DP 分工读取图片的能力
- 不改变 API server 主图像处理语义，只先打通数据透传和 worker 感知

#### phase3

目标：

- 将 API server 的本地图像主处理链路下沉到 worker
- 让 worker 接管 direct encode 主路径
- 保持对不适合 direct encode 的模态执行稳定 fallback

当前 `phase3` 的唯一主方案就是：

- `phase3 direct encode`

## 5. 当前主方案：phase3 direct encode

## 5.1 方案定义

`phase3 direct encode` 指的是：

- API server 不再承担本地图像主读取与主预处理
- worker 在接收到透传路径和必要元数据后，执行本 rank 需要的 multimodal 处理
- 对支持 direct encode 的路径，worker 自行完成 encode 并回填 encoder cache
- 对当前不适合 direct encode 的模态，比如当前视频路径，执行稳定 fallback

### 5.2 为什么不继续使用 phase3-local-only

历史上的 `phase3-local-only / local-plus` 方案存在两个问题：

1. 容易让接手人误以为它仍然是推荐主线
2. 会把当前真正应该继续维护的 `phase3 direct encode` 主方案和旧实验路径混在一起

因此在 `0428` 中：

- 代码入口已下线
- 启动脚本已删除
- 文档只保留“历史上存在，但已下线”的说明

## 6. 原始链路与修改后链路

### 6.1 原始 phase0 数据流

```text
Client
  |
  v
OpenAI API Server
  |
  |-- parse request
  |-- read local image/video path
  |-- load file / decode image
  |-- run HF multimodal processor
  |-- compute mm hash
  |-- build mm cache / shm payload
  v
Engine Core
  |
  |-- schedule request
  |-- prepare encoder inputs
  v
TP Workers
  |
  |-- consume prebuilt multimodal inputs
  |-- run model forward
  v
Response
```

### 6.2 phase1 数据流

```text
Client
  |
  v
OpenAI API Server
  |
  |-- same as phase0 main path
  |-- replace local image identity with stable UUID/hash
  v
Engine Core
  |
  v
TP Workers
  |
  v
Response
```

变化点：

- 主数据流基本不变
- 只优化本地图片 hash 计算方式

### 6.3 phase2 数据流

```text
Client
  |
  v
OpenAI API Server
  |
  |-- still loads and preprocesses image in main path
  |-- additionally collect local media paths
  |-- attach local paths into request/extra_args
  v
Engine Core
  |
  |-- keep request scheduling
  |-- preserve local path metadata
  v
TP Workers
  |
  |-- see local paths
  |-- validate local shard mapping
  |-- can load local shard for correctness checks
  v
Response
```

变化点：

- 重点是“透传本地路径”
- 不是最终性能形态

### 6.4 phase3 direct encode 数据流

```text
Client
  |
  v
OpenAI API Server
  |
  |-- parse request
  |-- keep request orchestration / tokenizer / cache semantics
  |-- for local_path and supported http path:
  |     do not perform full image preprocess main path
  |     only keep lightweight metadata needed by downstream
  v
Engine Core
  |
  |-- schedule encoder inputs
  |-- pass request state / multimodal metadata to workers
  v
TP Workers
  |
  |-- compute local shard assignment
  |-- load only local images for this rank
  |-- run HF multimodal preprocess on local shard
  |-- direct encode local shard
  |-- all-gather / rebuild original order if needed
  |-- write encoder outputs into encoder cache
  |-- for unsupported modality:
  |     fallback to stock encoder path
  v
Model Forward
  |
  v
Response
```

### 6.5 当前视频路径的数据流

当前视频请求已经修复，但它不是纯 direct encode 主路径，而是：

```text
Video local_path request
  |
  v
Phase3 worker patch detects unsupported / non-image modality
  |
  v
Explicit fallback to stock encoder execution
  |
  v
Stock encoder fills encoder cache
  |
  v
Normal forward continues without cache miss
```

这正是当前版本修复历史 `Encoder cache miss -> 500` 的关键点。

## 7. 参数与开关

### 7.1 phase 选择

核心 phase 开关：

```bash
export VLLM_ASCEND_API_OPT_PHASE=0|1|2|3
```

其中：

- `0` 对应 phase0
- `1` 对应 phase1
- `2` 对应 phase2
- `3` 对应 phase3 direct encode

### 7.2 常用启动参数

```bash
export MODEL_DIR=/path/to/Qwen3.5-4B
export HOST=127.0.0.1
export PORT=8000
export TENSOR_PARALLEL_SIZE=4
export MM_PROCESSOR_CACHE_TYPE=shm
export MM_PROCESSOR_CACHE_GB=20
export MM_ENCODER_TP_MODE=data
export GPU_MEMORY_UTILIZATION=0.85
export MAX_MODEL_LEN=36864
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/media/root
```

### 7.3 当前推荐 phase3 启动

```bash
export MODEL_DIR=/path/to/Qwen3.5-4B
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/0428/data
export TENSOR_PARALLEL_SIZE=4
export MM_PROCESSOR_CACHE_TYPE=shm
export MM_PROCESSOR_CACHE_GB=20
export MM_ENCODER_TP_MODE=data
export GPU_MEMORY_UTILIZATION=0.85
export MAX_MODEL_LEN=36864
/path/to/0428/bundle/start_phase3_server.sh
```

### 7.4 其它运行时参数

统一启动脚本默认还会设置：

- `HCCL_BUFFSIZE=1024`
- `VLLM_ASCEND_ENABLE_PREFETCH_MLP=1`
- `HCCL_OP_EXPANSION_MODE=AIV`
- `CPU_AFFINITY_CONF=1`
- `VLLM_USE_V1=1`
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
- `ENABLE_ASYNC_SCHEDULING`
- `ENABLE_CPU_BINDING`

## 8. 具体实现落点

## 8.1 启动与注册层

关键文件：

- `bundle/start_phase_server.sh`
- `bundle/start_phase0_server.sh`
- `bundle/start_phase1_server.sh`
- `bundle/start_phase2_server.sh`
- `bundle/start_phase3_server.sh`
- `scripts/launch_registered_server.py`
- `bundle/vllm_phase2_plugin/register.py`

职责：

- 设置运行时环境变量
- 注入 `bundle/` 到 `PYTHONPATH`
- 注册 vLLM 插件
- 将 phase 配置转成实际服务启动参数

## 8.2 phase1 层

关键文件：

- `bundle/vllm_phase2_plugin/patches/phase1.py`

职责：

- 稳定本地文件 UUID/hash
- 收集 multimodal local path 相关基础信息

## 8.3 phase2/phase3 主逻辑层

关键文件：

- `bundle/vllm_phase2_plugin/patches/phase2.py`
- `bundle/vllm_phase2_plugin/common/mm.py`

职责：

- phase2 路径透传
- phase3 renderer / parser / input processor / worker patch
- worker 侧 direct encode 主逻辑
- unsupported modality fallback
- encoder cache 回填

## 8.4 性能与调试层

关键文件：

- `bundle/vllm_phase2_plugin/patches/perf.py`
- `bundle/vllm_phase2_plugin/probes/`

职责：

- API server / engine core / worker 的统一打点
- 调试和定位热点函数

## 9. 当前支持范围

### 9.1 输入形式

当前 `phase3 direct encode` 已支持：

- `local_path`
- `http`

### 9.2 local_path

当前状态：

- 图片：支持
- 视频：支持

说明：

- 视频路径当前通过 patch 中的稳定 fallback 保证功能正确性
- 这已经足够作为当前正式交付能力写入 `0428`

### 9.3 http

当前状态：

- `http` 输入已打通并可作为当前支持能力记录

说明：

- `http` 的性能口径不能直接和 `local_path` 混看
- 当前更推荐把 `local_path` 作为主验证口径

## 10. 当前验证结论

基于本轮 `0427/0428` 最新修复结果，当前可确认：

- `phase3 direct encode + tp4 + local_path`
  - `L0` 可通过
  - 视频 `local_path` 不再触发历史 `500`
- `L0.5` 仍存在多图精度问题
- `MME` 已可作为当前阶段的 `local_path` 精度参考

## 11. 当前已知问题

当前最重要的已知问题是：

- 多图场景下的精度仍然不够理想

当前已不是主要问题的是：

- phase3 视频本地输入导致的 `Encoder cache miss`
- 因视频路径断链导致的 `500 Internal Server Error`

## 12. 后续建议

### 12.1 优先继续做什么

后续默认优先级：

1. `phase3 direct encode` 多图精度优化
2. `local_path` / `http` 回归能力保持
3. worker 热点进一步优化

### 12.2 不建议现在优先做什么

当前不建议优先：

- 恢复 `phase3-local-only` 旧试验路径
- 在未固定精度问题前继续扩散新的 phase3 变体
- 把多个 transport 或多个实验方案混在一起对比

## 13. 新任务接手建议

一个新的任务在读取 `0428/` 后，推荐执行：

1. 先读 `README.md`
2. 再读本文档
3. 再看：
   - `docs/02_phase_details.md`
   - `docs/03_deployment.md`
   - `docs/04_testing.md`
4. 最后再进入代码：
   - `register.py`
   - `phase1.py`
   - `phase2.py`
   - `perf.py`

这样能最快建立完整上下文：

- 为什么做
- 现在做到哪一步
- 哪些是当前主线
- 哪些历史方案已经下线
- 下一步应该继续优化哪里
