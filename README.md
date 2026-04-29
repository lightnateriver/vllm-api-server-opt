# 0428 Phase0~4 / Phase3 Direct Encode + Phase4 HTTP Cache 交付工程

这个目录用于沉淀当前多模态优化工作的最新可交付版本，目标是在不修改原始安装版
`vllm` / `vllm-ascend` 的前提下，只依赖本目录就能完成：

- `phase0 ~ phase4` 的部署与回归
- `phase3 direct encode` 与 `phase4 http cache bridge` 的功能验证
- `local_path / http / base64` 三种 transport 的精度矩阵回归
- 后续新特性的继续开发

当前版本明确将 `phase3 direct encode` 作为唯一继续演进的 `phase3` 主方案。
同时将 `phase4` 固化为 `http` 输入的推荐增强路径。
历史上的 `phase3-local-only / local-plus` 实验路径已经从接手视角下线，不再作为推荐方案。

## 当前结论

- `phase0` 仍然是稳定的精度与性能基线
- `phase3 direct encode` 已经完成：
  - `local_path` 输入支持
  - `http` 输入支持
  - `tp4` 启动与功能验证
  - 视频请求回退到 stock encoder 的稳定性修复
- `phase4` 已经完成：
  - `http/https` 图片先 materialize 到本地缓存池
  - worker 继续复用 `phase3 direct encode`
  - `http` 输入性能相对 `phase0 http` 明显改善
  - `L0 / L0.5 / MME` 三种 transport 回归跑通
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
3. 看 `docs/http输入phase4优化设计.md`
4. 看 `docs/02_phase_details.md`
5. 看 `docs/03_deployment.md`
6. 看 `docs/04_testing.md`

读完之后，你应该知道：

- 我们为什么要做 `phase3 direct encode`
- 为什么又补了 `phase4`
- 当前方案和 `phase0 / phase1 / phase2 / phase4` 的差异
- 怎么启动服务
- 怎么验证 `local_path / http / base64`
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

### phase4 http direct encode

```bash
export MODEL_DIR=/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/0428/data
export VLLM_ASCEND_HTTP_CACHE_DIR=/tmp/vllm_ascend_http_cache
export VLLM_ASCEND_HTTP_CACHE_TTL_S=86400
export VLLM_ASCEND_HTTP_TIMEOUT_S=30
export VLLM_ASCEND_HTTP_MAX_FILE_BYTES=67108864
/path/to/0428/bundle/start_phase4_server.sh
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

## phase4 精度矩阵

当前 `0428` 新增了统一编排脚本：

- `scripts/run_phase4_precision_matrix.py`

它用于在同一 `phase4` 服务上统一回归：

- transport：
  - `local_path`
  - `http`
  - `base64`
- suite：
  - `L0`
  - `L0.5`
  - `MME`

### 使用方式

先启动 `phase4` 服务，然后准备一个本地静态 HTTP 根：

```bash
python3 -m http.server 9000 --directory /path/to/0428/results/phase4_precision/http_root
```

再执行：

```bash
python /path/to/0428/scripts/run_phase4_precision_matrix.py \
  --execute \
  --wait-ready
```

默认产物目录：

- `results/phase4_precision/`

说明：

- 脚本会自动为 `http` 模式准备 `http_root/l0` 与 `http_root/multi_pics/cases`
- `MME` 仍依赖本机可用的 `/tmp/MME.tsv`
- 这些结果默认视为本地运行产物，不作为交付源码的一部分提交

### 2026-04-29 最新回归结论

- `local_path`
  - `L0`: `10/10`
  - `L0.5`: `23/40`，accuracy `57.5%`
  - `MME`: `80.6655%`
- `http`
  - `L0`: `10/10`
  - `L0.5`: `23/40`，accuracy `57.5%`
  - `MME`: `80.7077%`
- `base64`
  - `L0`: `10/10`
  - `L0.5`: `23/40`，accuracy `57.5%`
  - `MME`: `80.6655%`

这轮结果说明：

1. `phase4` 下三种 transport 的基础链路都已跑通
2. 历史上的 `http -> base64/local_path` 串扰式 `404/500` 本轮未复现
3. 当前 `L0.5` 失败 case 在三种 transport 上完全一致，因此剩余问题不是 transport-specific

## 当前推荐验证项

- Smoke:
  - `curl -s http://127.0.0.1:8000/v1/models`
- `L0 local_path`
- `L0.5 local_path`
- `MME local_path`
- `phase4` 下补齐 `local_path / http / base64` 精度矩阵

## 当前支持范围

`phase3 direct encode` 当前明确支持：

- `local_path` 图片输入
- `local_path` 视频输入
- `http` 图片输入

其中：

- 本地图片与视频在 `phase3` 下已经可以稳定运行
- 视频请求会在 worker 侧按当前 patch 逻辑安全回退到 stock encoder 路径
- `http` 输入已支持，但其性能和稳定性要和 `local_path` 分开评估

`phase4` 额外提供：

- `http/https` 图片输入的本地缓存池桥接
- 在保持 transport 身份为 `http` 的前提下复用 `phase3 direct encode`
- 不影响 `local_path` 与 `base64` 的 transport 语义

## 当前重点任务

后续继续开发时，默认把重点放在：

1. `phase3 direct encode` 多图精度提升
2. 保持 `phase4 http` 与 `local_path / base64` 的功能正确性
3. 在不破坏当前稳定性的前提下继续优化 worker 侧热点
