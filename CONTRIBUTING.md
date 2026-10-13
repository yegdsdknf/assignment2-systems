# 提交与实验记录约定

## 本次整理

以已发布的 `f2c8f5b` 为基础追加一个整理提交：将 `cs336-results/` 的 64 个原始文件移动至 `results/autodl/20260928/`，保留文件内容，增加索引与 SHA-256 清单，停止跟踪 `.autodl/`、`.Trash-*/` 与 Jupyter 自动备份。实现代码和测试不属于本次整理范围。

普通删除只清理最新文件树，旧提交中的归档和平台文件仍在 Git 历史中。需要缩小历史体积时，应另行安排历史重写并协调其他克隆。

## 按用途提交

| 用途 | 内容 | 提交信息示例 |
| --- | --- | --- |
| 代码 | 单项实现或修复、必要测试 | `fix: correct FSDP backward prefetch order` |
| 实验 | 同一配置组的结果、采集脚本和配置说明 | `results: add RTX 6000D DDP comparison` |
| 报告 | 推导、图表、结论与对应数据引用 | `writeup: analyze DDP communication overlap` |
| 整理 | 目录、忽略规则和索引 | `chore: organize AutoDL experiment artifacts` |

一次提交围绕一个目的；代码改动通过相关测试后先提交，再运行正式实验，使结果可以明确引用采集时的提交。

## 实验文件

- 小型 JSON、CSV、复现脚本和简要分析进入 Git；结果目录包含设备、日期及实验类型，文件名区分实现、配置和正式测试/probe/smoke。
- 记录代码提交、是否存在未提交改动、GPU/互联、PyTorch/CUDA、模型、batch、context、精度、随机种子、预热和测量次数；配置相同才能组成受控比较。
- 原始文件保持不变；修正分析时新增分析文件。重跑使用新的运行标识，避免覆盖旧结果。
- 新采集的大型 `.nsys-rep`、`.sqlite`、显存 `.pickle` 和压缩包默认存放在仓库外；Git 记录路径或下载地址、大小与 SHA-256。需要分享时可另行上传至 Release 或专门的数据存储。
- 这次已公开的 64 个文件保留在 Git 内，只移动路径；不重复增加压缩包或导出文件副本。

## 每次提交前

在实际运行项目的 Linux 环境执行以下步骤；本次按用户要求操作 AutoDL。

1. `git fetch origin`，检查当前分支与上游差异；工作区干净且仅落后时可 `git pull --ff-only`。
2. 用 `git status --short` 和 `git diff` 检查文件范围；涉及代码时运行相关测试及配置的检查工具。
3. 用明确的文件路径执行 `git add -- <路径>`，避免把实例目录整体加入仓库。
4. 检查 `git diff --cached --stat`、`git diff --cached` 和 `git diff --cached --check`；确认没有环境、缓存、平台数据库和重复归档。
5. 按用途创建提交，再普通 `git push`；上游有新提交时先检查和处理，不强制覆盖。

涉及多个文件或多天的实现工作，可使用功能分支。已发布提交的纠错默认追加提交；历史重写作为单独操作处理。
