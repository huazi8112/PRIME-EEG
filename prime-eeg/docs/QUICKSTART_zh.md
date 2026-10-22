# 中文复现说明

本仓库是 PRIME-EEG 已冻结实现的独立整理版，不是新算法版本，不重选参数。原始脑电、特征缓存、模型权重和论文文件不随仓库发布。

## 1. 安装与快速检查

使用 Python 3.12，在包含 `pyproject.toml`、`src/` 和 `reference/` 的代码包目录运行。若从公开仓库的外层根目录开始，先进入 `prime-eeg/`；若已在本地 `prime_eeg` 代码包目录，则无需再进入子目录：

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m prime_eeg verify-results
```

这些命令适用于 Windows；其他系统使用虚拟环境内对应的 Python。`verify-results` 仅验证随附账本的汇总计算和冻结配置，不代表重新处理了原始脑电。真正从数据复现需要继续第 2 步。

## 2. 准备数据并复现主结果

自行取得 CHB-MIT 1.0.0 官方数据。在示例相对路径 `data/chbmit/` 下放置 `chb01/` 等目录，每个目录保留 EDF 和官方 summary 文本。也可以直接传入自己的数据绝对路径，无须移动大型数据。

```text
.venv\Scripts\python.exe -m prime_eeg build-cache --data-root "data/chbmit" --output "cache/chbmit"
.venv\Scripts\python.exe -m prime_eeg evaluate --cache "cache/chbmit" --output "runs/main" --bootstrap 10000
.venv\Scripts\python.exe -m prime_eeg verify-results --recomputed "runs/main/result.json"
```

所有输出目录必须尚不存在；程序不会覆盖旧缓存或旧结果。完整复现应得到 118/153 次检出、289 次误报、629.522778 h 有效监测时长。第一条命令处理全部 EDF，耗时和磁盘占用明显高于后续评分。

试跑时在构建和评价两个命令中均添加 `--subjects chb19`。试跑结果不是全队列结果。

## 3. 消融、确认延迟与 Siena

英文 README 给出了四个组件消融、报警门控更新、严格发作区间匹配、真实经过时间确认延迟及 Siena 压力测试的独立命令。它们均沿用冻结规则，不进行新的调参。

确认延迟分析还需原始 EDF 头时间，不能把有采集空档的记录简单拼接成连续时间。Siena 需要单独准备官方数据，并严格限定为历史跨数据库压力测试。

## 4. 上传 GitHub

上传本仓库中的源码、配置、说明、测试和 reference 账本即可；不要把 data、cache、runs、虚拟环境或权重添加到版本控制。`.gitignore` 已排除常见数据/缓存/权重格式。发布前仍应查看实际待上传文件清单。

公开仓库为 https://github.com/huazi8112/PRIME-EEG 。本使用说明和程序不会自动提交或上传文件。本版本未授予软件许可证；不要把没有许可证理解为已获得开放使用授权。论文正式书目信息应以最终发表版本为准。

详细的方法边界及与论文结果的对应关系见 `REPRODUCIBILITY.md`；神经网络比较流程的复现覆盖范围见 `BASELINES.md`。
