# DS — 大模型（阿里云百炼 OpenAI 兼容端点）象棋引擎

本目录是 `UniChessServer` 的模型插件 `DS`（短名），也是一个独立 git 仓库
（远端 `git@github.com:jeefies/UniChessLLM.git`）。

## 快速入口

- `DS/engine.py`：GameEngine 六方法 + 百炼 HTTP 客户端（仅标准库 `urllib`）。
- `DS/config.json`：`default` / `fast` 两档预设。
- `DS/tests/test_engine.py`：离线单测（不联网）。
- `DS/tests/smoke_live.py`：真实 API 冒烟脚本（手工跑）。

## 运行

本机冒烟（Python ≥3.10，需 `chess`）：

```powershell
& C:\ProgramData\miniconda3\python.exe DS\tests\smoke_live.py
```

离线单测：

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\ProgramData\miniconda3\python.exe -m unittest discover -s DS\tests -v
```

## 配置

`.env` 固定放在仓库根（`Path(__file__).resolve().parent / '.env'`），格式：

```
BAILIAN_MODEL=deepseek-v4.1-flash
BAILIAN_BASE_URL=https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
BAILIAN_API_KEY=sk-...
```

优先级：进程环境变量 > `.env` 文件。三个键均为必填；缺失时 `__init__` 抛
`RuntimeError`，异常信息含**键名**、不含 key 内容。

远端部署后 `chmod 600 DS/.env`。

## 失败语义

不兜底：HTTP 失败或输出无法解析 → 重试耗尽直接抛错。普通对局返回 500；
观战 / 批量对弈 job 报错终止。

## 延迟与 thinking

- `default` 档：单步最多 `2 × 30s = 60s`（`max_attempts=2, timeout_s=30`）。
- `fast` 档：`2 × 10s = 20s`（thinking 关，适合观战）。
- 端点是否支持 `enable_thinking` 参数未知，引擎会自适应探测：首次 400
  疑似不认该参数时同 attempt 去参重发，并将"不支持"结论缓存，后续不再带参。

## 接入 Server

```bash
ln -s /home/jeefy/UniChess/DS ~/UniChess/Server/models/DS
```

`Server/models/DS` 是符号链接（指向 `~/UniChess/DS`），`DS` 不声明
`KIT_FACTORY`，观战 / 批量对弈由 `Kit.serving:game_engine_player_factory`
包装六方法。

## 密钥纪律

`.env` **不入库**（见 `.gitignore`）。`.env.example` 无真实 key，只作文档。
任何日志 / 异常 / 状态返回均不含 key。

## 相关文档

- 计划：`C:\Users\jeefy\.local\share\kilo\plans\1790554155553-ds-llm-chess-engine-plan.md`
- Server 模型插件契约：`Server/README.md` §模型清单
