# 配音环境与下载

## 按钮职责

- 模型行「下载」只下载该模型及必要模型资产，不安装 Python 或运行环境。
- 模型行「检查」只校验本地模型；缺失或损坏时报告错误，不自动下载。
- 「环境」内安装/检查运行依赖。Edge TTS 无本地模型，模型行的检查仅检查其在线服务客户端依赖，不代表在线合成已通过。
- 显式选择 CPU 或 GPU；GPU 检查/执行失败不会静默回退 CPU。CPU 使用独立依赖环境，实际可用性仍取决于上游模型、内存和本机测试。

## 项目内位置

以下相对于项目目录：

| 内容 | 路径 |
| --- | --- |
| PyTorch GPU 环境/源码 | `AppData/tts/<引擎>/managed-v2/` |
| PyTorch CPU 环境/源码 | `AppData/tts/<引擎>/managed-v2-cpu/` |
| Edge TTS 环境 | `AppData/tts/edge-tts/managed-v2/` |
| 共享 pip、模型库、运行临时缓存 | `AppData/tts/_cache/` |
| 模型 | `AppData/models/<模型目录>/` |
| audio.cpp 程序 | `AppData/tts/audio.cpp/<版本>/` |
| audio.cpp 模型 | `AppData/models/audio.cpp/<模型 ID>/` |
| 配音句子缓存 | `AppData/cache/dubbing/` |
| 应用诊断日志 | `AppData/logs/`（以应用配置为准） |

使用启动应用的兼容 Python 创建项目内 venv；不会再自动下载完整 Python，也不会向启动应用的全局环境安装配音包。不同引擎依赖可能冲突，因此隔离依赖环境，但复用基础解释器及项目缓存。需要不兼容的 Python 版本时先明确报错。

旧版 `AppData/tts/<引擎>/runtime` 和 `source` 不自动删除或覆盖。仅旧默认路径自动映射新版环境；用户自定义外部路径保留。旧版曾使用的系统 pip 缓存不在新版管理范围内，不自动清理。

取消保留已校验完整下载、源码包与已安装依赖；未完成的单个下载可能需重下，不保证 HTTP 断点续传。重试会重新核验，不凭完成标志跳过依赖完整性。

## 清理

先关闭应用及配音进程。只需重装某个引擎时，可自行删除它的 `managed-v2` / `managed-v2-cpu`；模型目录独立，不必一起删除。共享 `_cache` 可清理，但下一次操作可能重新下载缓存内容。删除整个项目可移除上述默认项目内内容，**不会删除用户另选位置的输出、外部环境或旧系统缓存**。不要把 `AppData` 整体删除当成只清配音：其中也有其他功能数据。

## GPU 与测试边界

AMD RX 9070 / 9070 XT Windows 路径采用官方 ROCm 7.2.1、Python 3.12、PyTorch/torchaudio 2.9.1 配方，要求驱动 26.2.2；程序不安装驱动。安装检查与实际 worker 都枚举设备并执行所选设备的张量检查，诊断包含索引、架构、Torch/HIP、可获取的驱动信息。此基础检查不是所有配音模型的兼容认证，不能据此认定 `hipErrorInvalidImage` 已解决。

环境导入、CPU 张量、模拟设备测试、真实模型推理及 Windows GPU 测试是不同层次；未进行的实测不得标为通过。audio.cpp 当前自动安装包为 Windows 版本，不能把它当作 Linux CPU 安装包。
