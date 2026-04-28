# 0428 Phase0~3 / Phase3 Direct Encode 交付工程

这个目录用于沉淀当前多模态优化工作的最新可交付版本，目标是在不修改原始安装版
`vllm` / `vllm-ascend` 的前提下，只依赖本目录就能完成：

- `phase0 ~ phase3` 的部署与回归
- `phase3 direct encode` 的功能验证与精度测试
- 后续新特性的继续开发

当前版本明确将 `phase3 direct encode` 作为唯一继续演进的 `phase3` 主方案。
历史上的 `phase3-local-only / local-plus` 实验路径已经从接手视角下线，不再作为推荐方案。

## 当前结论

- `phase0` 仍然是稳定的精度与性能基线
- `phase3 direct encode` 已经完成：
  - `local_path` 输入支持
  - `http` 输入支持
  - `tp4` 启动与功能验证
  - 视频请求回退到 stock encoder 的稳定性修复
- 当前 `phase3 direct encode` 的主要剩余问题不是服务稳定性，而是多图精度优化

## 适用场景

- 模型：`Qwen3.5-4B`
- 典型输入：
  - 约 `10k` 文本 token
  - `40` 张图片
- 部署：
  - `--enforce-eager`
  - `tp4`
  - `mm shm cache`
  - `mm encoder tp mode=data`

## 目录说明

- `bundle/`
  - 统一启动脚本
  - vLLM 插件入口
  - phase1/2/3 和性能 patch
- `scripts/`
  - 数据准备、性能测试、精度抓取、phase 对比
- `data/`
  - 本地样例和测试数据根目录
- `docs/`
  - 项目背景、设计目标、phase 说明、部署与测试说明

## 你接手后应该怎么理解这个目录

如果你是一个新任务，默认按下面顺序理解：

1. 看 `docs/00_project_scope.md`
2. 看 `docs/01_design.md`
3. 看 `docs/06_api_server_offload_design.md`
4. 看 `docs/02_phase_details.md`
5. 看 `docs/03_deployment.md`
6. 看 `docs/04_testing.md`

读完之后，你应该知道：

- 我们为什么要做 `phase3 direct encode`
- 当前方案和 `phase0 / phase1 / phase2` 的差异
- 怎么启动服务
- 怎么验证 `local_path` 和 `http` 输入
- 下一阶段应该继续优化什么

## 最常用启动命令

### phase0

```bash
export MODEL_DIR=/path/to/Qwen3.5-4B
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/0428/data
/path/to/0428/bundle/start_phase0_server.sh
```

### phase3 direct encode

```bash
export MODEL_DIR=/path/to/Qwen3.5-4B
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/0428/data
/path/to/0428/bundle/start_phase3_server.sh
```

如果要跑 `tp4 local_path` 精度测试，常用额外参数是：

```bash
export TENSOR_PARALLEL_SIZE=4
export MM_PROCESSOR_CACHE_TYPE=shm
export MM_PROCESSOR_CACHE_GB=20
export MM_ENCODER_TP_MODE=data
export GPU_MEMORY_UTILIZATION=0.85
export MAX_MODEL_LEN=36864
```

## 当前推荐验证项

- Smoke:
  - `curl -s http://127.0.0.1:8000/v1/models`
- `L0 local_path`
- `L0.5 local_path`
- `MME local_path`
- 如需验证 transport 覆盖面，再补 `http`

## 当前支持范围

`phase3 direct encode` 当前明确支持：

- `local_path` 图片输入
- `local_path` 视频输入
- `http` 图片输入

其中：

- 本地图片与视频在 `phase3` 下已经可以稳定运行
- 视频请求会在 worker 侧按当前 patch 逻辑安全回退到 stock encoder 路径
- `http` 输入已支持，但其性能和稳定性要和 `local_path` 分开评估

## 当前重点任务

后续继续开发时，默认把重点放在：

1. `phase3 direct encode` 多图精度提升
2. 保持 `local_path` / `http` 功能正确性
3. 在不破坏当前稳定性的前提下继续优化 worker 侧热点
