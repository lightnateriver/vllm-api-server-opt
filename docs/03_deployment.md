# 部署方式

## 前置条件

目标机器需要满足：

1. `python -c "import vllm"` 成功
2. `python -c "import vllm_ascend"` 成功
3. `python -m vllm.entrypoints.openai.api_server --help` 成功
4. 模型目录可读
5. 本地媒体目录可读

这份交付默认假设系统里的 `vllm` / `vllm-ascend` 是最初版安装环境，
不要求改 site-packages，也不要求重新打包安装插件。

## 插件如何生效

`0428/bundle/start_phase_server.sh` 会自动完成下面几件事：

1. 把 `0428/bundle` prepend 到 `PYTHONPATH`
2. 设置 `VLLM_PLUGINS=phase2_local_plugin,ascend`
3. 让 vLLM 通过 `bundle/vllm_phase2_plugin-0.0.1.dist-info/entry_points.txt` 找到插件入口
4. 用原始 `python -m vllm.entrypoints.openai.api_server` 起服务

因此，新机器只需要拷贝整个 `0428/` 目录，不需要单独 `pip install` 这个插件。

## 数据与输入准备

如果使用本地输入，确保：

```bash
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/0428/data
```

如果使用其它目录，必须把 `ALLOWED_LOCAL_MEDIA_PATH` 指到能覆盖该目录的上级路径。

当前 `phase3 direct encode` 已支持：

- `local_path`
- `http`

建议优先使用 `local_path` 做功能与精度验证。

## 启动命令

### phase0

```bash
export MODEL_DIR=/path/to/Qwen3.5-4B
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/0428/data
/path/to/0428/bundle/start_phase0_server.sh
```

### phase1

```bash
export MODEL_DIR=/path/to/Qwen3.5-4B
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/0428/data
/path/to/0428/bundle/start_phase1_server.sh
```

### phase2

```bash
export MODEL_DIR=/path/to/Qwen3.5-4B
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/0428/data
/path/to/0428/bundle/start_phase2_server.sh
```

### phase3 direct encode

```bash
export MODEL_DIR=/path/to/Qwen3.5-4B
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/0428/data
/path/to/0428/bundle/start_phase3_server.sh
```

## 推荐的 tp4 启动参数

```bash
export TENSOR_PARALLEL_SIZE=4
export MM_PROCESSOR_CACHE_TYPE=shm
export MM_PROCESSOR_CACHE_GB=20
export MM_ENCODER_TP_MODE=data
export GPU_MEMORY_UTILIZATION=0.85
export MAX_MODEL_LEN=36864
```

## 默认参数

统一启动脚本默认包含：

- `--enforce-eager`
- `--tensor-parallel-size 4`
- `--mm-processor-cache-type shm`
- `--mm-processor-cache-gb 20`
- `--mm-encoder-tp-mode data`
- `--gpu-memory-utilization 0.85`
- `--max-model-len 36864`
- `--async-scheduling`
- `--additional-config '{"enable_cpu_binding": true}'`

同时默认导出：

- `HCCL_BUFFSIZE=1024`
- `VLLM_ASCEND_ENABLE_PREFETCH_MLP=1`
- `HCCL_OP_EXPANSION_MODE=AIV`
- `CPU_AFFINITY_CONF=1`
- `VLLM_USE_V1=1`
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`

运行前都可以通过环境变量覆盖。

## 最小验证

服务起来后先做 smoke：

```bash
curl -s http://127.0.0.1:8000/v1/models
```

拿到模型列表后，再进行精度或性能测试。
