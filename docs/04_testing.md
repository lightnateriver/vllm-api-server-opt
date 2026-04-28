# 测试方式

## 1. Smoke

服务启动后先确认 API 可用：

```bash
curl -s http://127.0.0.1:8000/v1/models
```

## 2. 当前推荐精度验证

当前版本更推荐围绕 `phase3 direct encode` 做功能与精度验证。

建议顺序：

1. `L0 local_path`
2. `L0.5 local_path`
3. `MME local_path`
4. 如需补 transport 验证，再测试 `http`

## 3. 标准性能测试

标准脚本默认使用 13 组数据：

- `round_0 ~ round_2` 预热
- `round_3 ~ round_12` 正式统计

执行命令：

```bash
python /path/to/0428/scripts/run_multimodal_baseline.py
```

脚本会输出并归档：

- TTFT
- TPOT
- E2E
- API / engine core / tp worker 热点函数统计

如需固定输出上限，避免不同 phase 因生成长度不同带来口径偏移：

```bash
export VLLM_BENCHMARK_MAX_COMPLETION_TOKENS=72
python /path/to/0428/scripts/run_multimodal_baseline.py
```

## 4. 精度抓取与对比

### 抓取

```bash
python /path/to/0428/scripts/capture_phase_outputs.py \
  --label phase0 \
  --out /path/to/0428/results/phase0_outputs.json
```

### 比较

```bash
python /path/to/0428/scripts/compare_phase_outputs.py \
  --left /path/to/0428/results/phase0_outputs.json \
  --right /path/to/0428/results/phase3_outputs.json \
  --out /path/to/0428/results/phase0_vs_phase3_compare.json
```

建议的精度对比组合：

- `phase0` vs `phase1`
- `phase0` vs `phase2`
- `phase0` vs `phase3`

当前不再建议把 `phase3-local-only` 作为正式比较对象。

## 5. phase1 cache hit 验证

这个脚本会构造重复图片输入，比较命中前后的 API 侧热点时间：

```bash
python /path/to/0428/scripts/phase1_repeat_hit_test.py
```

重点关注：

- `ProcessorInputs.get_mm_hashes`
- 请求总耗时

## 6. 当前已知状态

- `phase3 direct encode` 在 `L0 local_path` 下功能稳定
- 视频 `local_path` 请求已修复，不再触发历史的 `Encoder cache miss`
- 当前主要待提升项是 `L0.5` 多图精度
