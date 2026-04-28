# HTTP输入 Phase4 优化设计

## 1. 文档目标

本文档描述 `0428/` 目录中新引入的 `phase4` 方案。

`phase4` 的目标不是替换 `phase3 direct encode`，而是补齐一个明确的能力缺口：

- 让 `http` 输入也能获得接近 `local_path + phase3 direct encode` 的收益
- 且不修改 `vllm` / `vllm-ascend` 源码
- 只通过 `0428/bundle/vllm_phase2_plugin` 的运行期 patch 实现

一句话概括：

`phase4 = API server 先把 http 图片下载并缓存到本地，再复用 phase3 direct encode 的 worker 侧本地路径链路。`

---

## 2. 背景与问题定义

### 2.1 phase3 已解决什么

`phase3 direct encode` 已经在 `local_path` 输入下实现了 API server 下沉：

- API server 不再承担本地图像主读取
- API server 不再承担本地图像主预处理
- worker 按 TP/ViT-DP 负责自己分片的图片读取、HF preprocess 和 encode

这条链路已经证明对 `local_path` 是有效的。

### 2.2 phase3 在 http 输入下的局限

在 `phase3` 但还没有 `phase4` 时，`http` 输入虽然“功能可用”，但并不能享受真正的 direct encode 优化收益。

原因是：

1. `http` URL 不是本地路径
2. API patch 无法像 `local_path` 那样直接构造 `_Phase3LocalImageRef`
3. 因此 `phase3_local_mm_ready=False`
4. worker 侧不会进入现有 `phase3 direct encode` 本地路径流程
5. 请求最终会回到 stock 的图片抓取与处理逻辑

这意味着：

- `http` 输入在 `phase3` 下更多是“兼容支持”
- 不是“真正优化后的下沉路径”

### 2.3 本阶段的核心目标

`phase4` 要解决的问题是：

`如何让远端 http 图片请求，在不修改 stock vLLM 源码的前提下，复用 phase3 direct encode 的本地路径优化链路。`

---

## 3. 设计目标

### 3.1 必须满足的目标

1. 不修改 `vllm` / `vllm-ascend` 源码
2. 只改 `0428` 下的 patch、脚本、文档
3. 不影响 `phase0 ~ phase3` 已有行为
4. `phase4` 必须只在 `VLLM_ASCEND_API_OPT_PHASE >= 4` 时生效
5. 远端 `http/https` 输入优先支持
6. 本地静态 `http` 输入自然兼容
7. 复用现有 `phase3 direct encode` worker 主链路
8. 失败时能回落到 stock `http` 抓图逻辑，保证服务可用

### 3.2 非目标

本阶段不做：

1. 不做图像内容 hash 去重
2. 不做跨 transport 的统一内容级缓存
3. 不做视频 http 输入的 phase4 优化
4. 不改 worker 让其直接下载远端 http 资源

---

## 4. 为什么选择“API server 本地缓存池”

我们考虑过两类思路。

### 4.1 方案A：worker 直接下载 http 图片

优点：

- API server 看起来更轻

缺点：

1. 每个 worker/rank 都可能重复下载
2. 下载动作落到 worker，调度、超时、失败重试更难统一控制
3. 同一请求中，多 rank 对同一 URL 的一致性和缓存复用更难收敛
4. 与当前 `phase3` 已有“按本地路径 direct encode”的实现复用度低

### 4.2 方案B：API server 下载到本地缓存池，再当作 local_path 走 phase3

优点：

1. 只下载一次
2. 更容易做 TTL、锁、大小限制、trace
3. 下载完成后天然变成“本地文件”
4. 可直接复用现有 `phase3 direct encode` worker 流程
5. 对 worker 改动最小

缺点：

1. API server 增加一次 URL 到本地文件的 materialize 动作
2. 需要额外缓存目录和生命周期管理

### 4.3 结论

本期选择方案B：

`API server 本地缓存池 + 物化到本地 + 复用 phase3 direct encode`

这是当前改动最小、风险最小、最符合现有代码演进方向的方案。

---

## 5. 总体方案

## 5.1 方案定义

当请求中的图片是 `http/https` URL 且当前 phase 为 `4` 时：

1. API server 识别该输入是 HTTP 图片
2. 先按 canonical URL 计算缓存 key
3. 在本地缓存目录中检查是否存在可复用的本地文件
4. 若命中，则直接得到本地缓存文件路径
5. 若未命中，则由 API server 下载远端图片到缓存目录
6. 下载成功后，构造一个“带本地路径的 phase3 图片引用对象”
7. 下游继续沿用 `phase3 direct encode` 的本地路径透传与 worker encode 流程

关键点：

- transport 身份仍然记录为 `media_mode=http`
- 但 `_phase2_local_path` 指向本地缓存文件
- 所以下游 worker 可以把它当成本地图片读取

## 5.2 目标数据流

### phase3 下的 http 原链路

```text
Client
  |
  v
API Server
  |
  |-- 识别到 http URL
  |-- 无法构造 local_path phase3 ref
  |-- 继续走 stock fetch/decode
  v
Engine Core / Worker
  |
  |-- phase3_local_mm_ready = false
  |-- 不进入 direct encode 本地路径主链
  v
Response
```

### phase4 下的 http 新链路

```text
Client
  |
  v
API Server
  |
  |-- 识别到 http URL
  |-- canonicalize URL
  |-- 查询本地 HTTP cache pool
  |   |-- hit  -> 直接拿到 local cached file
  |   |-- miss -> 下载远端图片到本地缓存
  |
  |-- 构造 phase3-style local image ref
  |   media_mode = http
  |   local_path = /tmp/vllm_ascend_http_cache/<sha1>.jpg
  v
Renderer / Engine Input
  |
  |-- phase3_local_mm_ready = true
  |-- phase2_local_mm_paths 透传缓存后的 local_path
  v
TP Worker
  |
  |-- 复用 phase3 direct encode
  |-- 读取 cached local file
  |-- HF preprocess
  |-- direct encode
  |-- 回填 encoder cache
  v
Response
```

---

## 6. 核心设计细节

## 6.1 缓存 key

`phase4` 不按图像内容计算 hash，而按 URL 身份计算。

当前 key 规则：

```text
cache_key = sha1(canonical_http_url)
```

其中 `canonical_http_url` 规则：

- 只接受 `http` / `https`
- 去掉 fragment
- 保留 path / query

设计原因：

1. 计算便宜
2. 不需要先读取整图内容
3. 与 transport 身份天然绑定
4. 满足本阶段“远端 URL 复用”的主要目标

## 6.2 transport 身份与本地路径并存

这是 `phase4` 最关键的设计点。

缓存后的图片虽然已经落盘为本地文件，但它的逻辑来源仍然是 `http`。

因此 patch 中同时保留两类信息：

- `media_mode = http`
- `local_path = /tmp/.../<cache_key>.<suffix>`

这意味着：

1. 下游 direct encode 可以复用本地文件
2. 调试和 trace 里仍然能看出它来自 HTTP 输入
3. 不会把它错误伪装成真正的用户 `local_path`

## 6.3 API 侧缓存池目录

默认目录：

```text
/tmp/vllm_ascend_http_cache
```

可通过环境变量覆盖：

```bash
export VLLM_ASCEND_HTTP_CACHE_DIR=/path/to/cache_dir
```

缓存池内容包括：

1. 图片文件本体
2. 同 key 的 JSON meta 文件
3. 同 key 的 lock 文件

## 6.4 TTL

默认 TTL：

```text
86400 秒
```

可通过环境变量覆盖：

```bash
export VLLM_ASCEND_HTTP_CACHE_TTL_S=86400
```

语义：

- TTL 内且本地文件存在，认为 cache hit
- TTL 超过后重新下载覆盖
- 如果 TTL < 0，则视为永久有效，直到文件被人工清理

## 6.5 文件锁

为了避免并发请求对同一个 URL 同时下载，`phase4` 使用基于文件的排它锁：

```text
<cache_key>.lock
```

流程：

1. 进入 materialize 流程
2. 打开 lock 文件
3. `fcntl.flock(..., LOCK_EX)`
4. 二次检查 meta 是否已被其它进程填充
5. 命中则直接复用
6. 未命中才下载

这样可以避免：

- 重复下载
- 半成品文件被其它请求抢读

## 6.6 下载限制

当前实现提供以下保护项：

### 超时

```bash
export VLLM_ASCEND_HTTP_TIMEOUT_S=30
```

### 最大文件大小

默认 `64MB`：

```bash
export VLLM_ASCEND_HTTP_MAX_FILE_BYTES=67108864
```

### 分块大小

默认 `1MB`：

```bash
export VLLM_ASCEND_HTTP_CHUNK_BYTES=1048576
```

### User-Agent

```bash
export VLLM_ASCEND_HTTP_USER_AGENT=vllm-ascend-phase4-http-cache/1.0
```

---

## 7. 代码落点

## 7.1 `patches/phase1.py`

`phase4` 的主实现位于：

`0428/bundle/vllm_phase2_plugin/patches/phase1.py`

主要新增逻辑：

1. HTTP 缓存目录和配置读取
2. `canonical_http_url -> cache_key` 计算
3. meta / lock 文件管理
4. API 侧下载与本地文件物化
5. HTTP 输入构造为 phase3-style local ref
6. 在 `parse_image / fetch_image / fetch_image_async / _image_with_uuid_async` 中接线

## 7.2 `common/mm.py`

补充 source metadata 透传，使 trace/调试能够继续看到：

- `media_mode`
- `local_path`
- `media_uuid`
- `source_key`
- `image_url`

---

## 8. 兼容性与边界

## 8.1 对 phase0~phase3 的隔离

`phase4` 只在下面条件同时成立时生效：

1. `VLLM_ASCEND_API_OPT_PHASE >= 4`
2. 当前输入是 `http/https` 图片 URL
3. materialize 成功

否则行为保持为：

- phase0~phase3 完全不变
- phase4 下若 materialize 失败，则回退 stock http 图片抓取逻辑

## 8.2 为什么 phase4 不影响 local_path

Phase4 不改 `local_path` 的主链路。

优先级是：

1. 先尝试 `local_path -> phase3 direct encode`
2. 只有不是本地路径且 phase4 开启时，才尝试 `http -> local cache -> phase3 direct encode`

## 8.3 为什么 phase4 不影响 base64

`base64` 输入仍保持原有 transport identity 规则，不经过 phase4。

因为：

- `phase4` 的问题域只针对 `http/https`
- base64 没有“远端 URL 可复用下载”的问题

---

## 9. 失败回退策略

只要下列任一条件不满足，就不强行走 Phase4：

1. URL 不是 `http/https`
2. 本地缓存目录创建失败
3. 下载超时
4. 返回非预期响应
5. 文件大小超过限制
6. 下载后的文件无法被 PIL 正常识别为图片

此时统一回退到 stock 行为：

- 继续由原始 `vllm` 图片抓取逻辑处理

这保证了：

- `phase4` 是“优化增强”
- 不是“功能替代”

---

## 10. 方案收益

在理想命中场景下，`phase4` 可以带来以下收益：

1. 远端 HTTP 图片只需下载一次
2. 重复请求可直接复用本地缓存文件
3. 下载完成后，下游进入与 `local_path` 一致的 direct encode worker 主链
4. API server 不再为每个请求重复承担完整的 stock HTTP 图片处理路径

对于本地静态 HTTP 服务场景，收益会更接近 `local_path`。

对于真实远端 HTTP 服务场景，Phase4 至少能保证：

- 首次请求的“下载成本”和“worker direct encode 成本”解耦
- 后续同 URL 请求可复用本地缓存

---

## 11. 风险与注意事项

1. `phase4` 无法消除第一次请求的远端下载成本
2. 如果远端资源内容变化但 URL 不变，在 TTL 期间会命中旧缓存
3. 缓存目录需要可写空间
4. 如果远端图片很多且 URL 离散，缓存目录会持续增长，需要后续补后台清理策略
5. 当前实现主要针对图片，不扩展到视频

---

## 12. 启动方式

新增启动脚本：

```bash
/mnt/sfs_turbo/codes/lzp/0428/bundle/start_phase4_server.sh
```

等价于：

```bash
export VLLM_ASCEND_API_OPT_PHASE=4
/mnt/sfs_turbo/codes/lzp/0428/bundle/start_phase_server.sh
```

可选环境变量：

```bash
export VLLM_ASCEND_HTTP_CACHE_DIR=/tmp/vllm_ascend_http_cache
export VLLM_ASCEND_HTTP_CACHE_TTL_S=86400
export VLLM_ASCEND_HTTP_TIMEOUT_S=30
export VLLM_ASCEND_HTTP_MAX_FILE_BYTES=67108864
export VLLM_ASCEND_HTTP_CHUNK_BYTES=1048576
export VLLM_ASCEND_HTTP_USER_AGENT=vllm-ascend-phase4-http-cache/1.0
```

---

## 13. 验证计划

本阶段最小验证集：

1. 启动 `phase4 tp4`
2. 使用 `http` 输入跑 `L0`
3. 使用 `http` 输入跑 `L0.5`
4. 使用 `http` 输入跑 `MME`

验证点：

1. 功能是否通过
2. 是否不再走 stock http fallback 主路径
3. trace 中是否出现：
   - `phase4_http_cache_hit`
   - `phase4_http_cache_fill`
   - `api_parse_image_phase4_http_ref`
   - `worker_phase3_direct_encode`
4. 精度是否与现有稳定结果保持一致或可接受

---

## 14. 结论

`phase4` 的本质不是发明一条全新 worker http 流程，而是做一个非常工程化的桥接：

`HTTP URL -> API server 本地缓存池 -> local cached file -> 复用 phase3 direct encode`

它兼顾了：

- 远端 HTTP 优先场景
- 本地 HTTP 兼容
- 不修改 stock vLLM 源码
- 对 phase0~phase3 的严格隔离
- 最小实现改动面

这是当前 `0428` 演进到下一阶段时最稳妥、最可控的 HTTP 优化方案。
