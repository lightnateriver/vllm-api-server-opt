# 总体设计

## 交付边界

这套交付的边界是：

- 不修改 site-packages 中的 `vllm` / `vllm-ascend`
- 不要求重新打 wheel 或重新安装框架
- 所有行为变化都来自插件注册和运行期 patch
- 所有 phase 都由环境变量控制，方便逐阶段验证和回退

## 设计分层

### 1. 注册层

利用 vLLM 的 general plugin 入口，让原始
`python -m vllm.entrypoints.openai.api_server`
在启动时自动执行本目录里的插件注册函数。

当前交付通过下面两个文件完成插件发现：

- `bundle/vllm_phase2_plugin-0.0.1.dist-info/entry_points.txt`
- `bundle/vllm_phase2_plugin/register.py`

其中 entry point 名称是 `phase2_local_plugin`，启动脚本会设置：

```bash
export VLLM_PLUGINS=phase2_local_plugin,ascend
```

### 2. patch 层

插件入口会根据 `VLLM_ASCEND_API_OPT_PHASE` 决定是否安装 patch：

- `phase1`
  - 本地图像稳定 UUID/hash
- `phase2`
  - `image_url` 对应的本地路径透传到 engine core / worker
- `phase3`
  - `phase3 direct encode`
  - API server 不再承担本地图像主处理工作
  - worker 承担 multimodal 读取、处理与 encode 主路径
- `perf`
  - API server / engine core / tp worker 的统一打点

当前实现位于：

- `bundle/vllm_phase2_plugin/common/`
- `bundle/vllm_phase2_plugin/patches/`
- `bundle/vllm_phase2_plugin/probes/`

### 3. phase3 direct encode 主方案

当前 `phase3` 的唯一主方案是 `phase3 direct encode`。

目标是把 API server 的重型 multimodal 工作下沉到 worker，使 API 侧尽量只保留：

- 请求解析
- tokenizer
- 调度与 cache 语义

worker 负责：

- 本地路径透传后的 multimodal 数据读取
- HF processor 图像预处理
- direct encode 路径下的本 rank multimodal encode
- 必要时对不适合 direct encode 的模态执行稳定 fallback

### 4. 输入支持边界

当前版本在 `phase3 direct encode` 下明确支持：

- `local_path`
- `http`

说明：

- `local_path` 是当前最主要、最稳定的优化目标
- `http` 已支持，但它的性能口径应和 `local_path` 分开分析
- 视频输入在当前实现下已通过 patch 修复为稳定功能路径

## 启动层

`bundle/start_phase_server.sh` 是统一启动入口，负责：

1. 设置运行时环境变量
2. 把 `bundle/` prepend 到 `PYTHONPATH`
3. 设置 `VLLM_PLUGINS=phase2_local_plugin,ascend`
4. 保持底层服务仍由原始 `python -m vllm.entrypoints.openai.api_server` 启动

phase 专用脚本只负责设置 phase 环境变量后转调统一入口：

- `start_phase0_server.sh`
- `start_phase1_server.sh`
- `start_phase2_server.sh`
- `start_phase3_server.sh`

当前目录不再提供 `phase3-local-only` 或 `phase3-local-plus` 的推荐启动入口。

## 默认运行时参数

启动脚本默认对齐当前验证环境：

- `HCCL_BUFFSIZE=1024`
- `VLLM_ASCEND_ENABLE_PREFETCH_MLP=1`
- `HCCL_OP_EXPANSION_MODE=AIV`
- `CPU_AFFINITY_CONF=1`
- `VLLM_USE_V1=1`
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
- `--async-scheduling`
- `--additional-config '{"enable_cpu_binding": true}'`
- `--enforce-eager`
- `--tensor-parallel-size 4`
- `--mm-processor-cache-type shm`
- `--mm-processor-cache-gb 20`
- `--mm-encoder-tp-mode data`
- `--gpu-memory-utilization 0.85`
- `--max-model-len 36864`

这些都可以通过环境变量覆盖。

## 当前已完成状态

当前已经完成的设计目标：

- `phase1/phase2/phase3` 插件化接入
- `phase3 direct encode` 主链路落地
- `local_path` 图片支持
- `local_path` 视频支持
- `http` 输入支持
- `phase3` 视频请求稳定性修复

## 当前仍待继续工作的方向

后续默认继续推进：

1. `phase3 direct encode` 多图精度优化
2. `local_path` / `http` 统一精度回归
3. worker 热点进一步优化
