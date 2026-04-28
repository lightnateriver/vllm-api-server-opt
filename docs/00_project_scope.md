# 项目述求

## 背景

本项目面向 `Qwen3-VL / Qwen3.5-VL` 类多模态服务场景，重点是降低
API server 在本地多模态输入下的 CPU 侧负担。

典型目标负载：

- 文本：约 `10k` token
- 图像：`40` 张图片
- 部署：`TP4/TP8` + `VIT DP`

原始链路里，API server 会承担：

- 本地图像读取
- PIL decode
- HF processor 图像预处理
- multimodal hash
- 图像相关 shm copy

这些逻辑容易成为 TTFT 和端到端耗时的瓶颈。

## 当前目标

当前版本的交付目标已经从“探索多条 phase3 变体”收敛为：

- 保留 `phase0 ~ phase3` 分阶段验证能力
- 将 `phase3 direct encode` 固化为唯一继续演进的 `phase3` 主方案
- 删除 `phase3-local-only / local-plus` 这类错误或过时实验路径的接手入口
- 明确记录当前已支持：
  - `local_path`
  - `http`

## 核心诉求

1. 不修改原始安装版 `vllm` / `vllm-ascend`
2. 统一通过插件注册和运行期 patch 生效
3. phase 行为由环境变量控制，便于回退
4. API server 尽量只保留调度和轻量请求编排语义
5. worker 承担更多图像处理与 multimodal encode 工作
6. 保留完整的部署、精度验证和性能分析路径

## 当前阶段结论

- `phase3 direct encode` 已经完成主链路落地
- `local_path` 输入已经稳定支持
- `http` 输入已经打通
- 视频请求在 `phase3` 下已通过 patch 修复为稳定路径
- 当前主要待继续优化的是多图精度，而不是服务可用性
