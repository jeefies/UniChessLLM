# DS AGENTS.md

面向 AI 编码 agent。DS 是大模型（阿里云百炼 OpenAI 兼容端点）象棋引擎，作为 `UniChessServer` 的模型 `DS` 接入。

## 目录

```
DS/
├── .env            # 运行配置（BAILIAN_*），不入库、远端 chmod 600
├── .env.example    # 占位示例（无真实 key）
├── .gitignore      # .env 等
├── engine.py       # GameEngine 六方法 + 百炼客户端
├── config.json     # default / fast 预设
├── README.md       # 运行 / 部署 / 失败语义 / 密钥纪律
├── AGENTS.md       # 本文件
└── tests/
    ├── __init__.py
    ├── test_engine.py  # 离线单测（标准库 unittest，不联网）
    └── smoke_live.py  # 真实 API 冒烟脚本（非 unittest 发现对象）
```

## 环境

- 本机：`C:\ProgramData\miniconda3\python.exe`（Py 3.13，有 `chess`，无 torch）。
- 远端：`/home/jeefy/miniconda3/envs/unichess/bin/python`（Py 3.12，有 chess）。
- 依赖：仅标准库 + `chess`（不新增任何依赖）。

## 纪律

- `.env` 不入库；提交前 `git check-ignore .env`；`.env` 值绝不出现在
  日志 / 异常 / prompt / URL 中。
- 不打印 API key；smoke 脚本只报告状态码 / 耗时 / 响应键名 / 非空性。
- `__init__/setup/human_move` 不发网络请求；网络仅在 `engine_move` 触发。
- `engine_move` 失败直接抛错（用户确认不兜底）。
- 六方法契约对齐 `Server/models/__init__.py`；`engine_move` 返回 dict 必须含
  `"engine_move"`（UCI 字符串）。
- 本引擎**不声明** `KIT_FACTORY`；观战 / 批量对弈走 `Kit.serving` 包装路径。
- 构造参数白名单：`model, timeout_s, max_attempts, temperature, max_tokens,
  history_plies, thinking, extra_request`；未知 kwargs → `TypeError`。
- `enable_thinking` 参数名是百炼 / DashScope 惯例猜测；端点不支持时自适应降级。

## 运行测试

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\ProgramData\miniconda3\python.exe -m unittest discover -s DS\tests -v
```

## 远端部署顺序

本地离线单测 → 本地 smoke → GitHub push → 远端 clone + 离线单测 +
smoke → Server 接入 → 公网验收。任一步失败即停并报告证据。
