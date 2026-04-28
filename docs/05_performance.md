# 性能说明

## 当前性能文档定位

这一版 `0428` 目录不再把历史上的 `phase3-local-only / local-plus`
实验结果作为主视角保留。

当前性能分析的重点应该放在：

- `phase0` 基线
- `phase3 direct encode`
- 同一输入口径下的热点迁移

## 推荐观察项

### API server

- `MediaConnector.fetch_image_async`
- `AsyncMicrobatchTokenizer.encode`
- `ProcessorInputs.get_mm_hashes`
- `Qwen3VLMultiModalProcessor._call_hf_processor`
- `ShmObjectStoreSenderCache.get_and_update_items_with_callback`
- `SingleWriterShmObjectStorage.batch_copy_to_buffer`

### engine core

- `engine_core_time`
- `engine_core_decode_only_time`

### tp worker

- `Qwen3VLMultiModalProcessor._call_hf_processor`
- `ImageMediaIO.load_file`
- `MsgpackSerde.deserialize`
- `tp_worker_execute_model_decode_only`

## 当前阶段结论

在当前设计目标下，性能分析要围绕下面这个问题展开：

`phase3 direct encode` 是否成功把 API server 的热点下沉到 worker，同时保持功能稳定和可接受的精度。

## 后续建议

后续继续分析性能时，建议：

1. 固定输入 transport
2. 固定 completion token 上限
3. 先比较 `phase0` 和 `phase3 direct encode`
4. 再局部分析 worker 热点是否值得继续优化
