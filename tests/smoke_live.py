#!/usr/bin/env python3
"""DS 真实 API 冒烟脚本（非 unittest 发现对象）。

三步：
1. GET {base}/models（连通 + 鉴权证据）
2. POST /chat/completions（thinking 开关 + 降级探测证据）
3. 实例化引擎 → engine_move（default 档）；再跑一次 thinking=false

失败退出码非 0；不含 API key 输出。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from engine import GameEngine, _reset_thinking_probe  # noqa: E402
from engine import _load_env_config  # noqa: E402


def _bearer_get(url, api_key, timeout_s):
    req = urllib.request.Request(
        url,
        method='GET',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json',
        },
    )
    status = None
    raw = ''
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, 'status', None) or resp.getcode()
            raw = resp.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            raw = exc.read().decode('utf-8', 'replace')
        except Exception:
            raw = ''
    except urllib.error.URLError as exc:
        err = exc
    except Exception as exc:
        err = exc
    return status, raw, err


def _post(url, api_key, payload, timeout_s):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        url, data=body, method='POST',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
    )
    status = None
    raw = ''
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, 'status', None) or resp.getcode()
            raw = resp.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            raw = exc.read().decode('utf-8', 'replace')
        except Exception:
            raw = ''
    except urllib.error.URLError as exc:
        err = exc
    except Exception as exc:
        err = exc
    return status, raw, err


def step1_models(cfg):
    base = cfg['BAILIAN_BASE_URL'].rstrip('/')
    url = base + '/models'
    print(f'[1/3] GET {url}')
    t0 = time.perf_counter()
    status, raw, err = _bearer_get(url, cfg['BAILIAN_API_KEY'], 30)
    elapsed = time.perf_counter() - t0
    if err is not None:
        print(f'  ERROR: {err}')
        return False
    print(f'  status={status} elapsed={elapsed:.2f}s')
    if status == 200:
        try:
            data = json.loads(raw)
            ids = []
            if isinstance(data, dict):
                ids = [m.get('id', '?') for m in data.get('data', [])[:5]]
            print(f'  models (first 5): {ids}')
        except json.JSONDecodeError:
            print(f'  non-json body (first 200): {raw[:200]}')
    else:
        print(f'  body (first 300): {raw[:300]}')
    return True


def step2_chat(cfg):
    base = cfg['BAILIAN_BASE_URL'].rstrip('/')
    url = base + '/chat/completions'
    model = cfg['BAILIAN_MODEL']
    print(f'[2/3] POST {url}  model={model}  enable_thinking=true')
    payload = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': 'Please answer with one UCI move from the starting position, e.g. e2e4.'},
        ],
        'temperature': 0.0,
        'max_tokens': 64,
        'stream': False,
    }
    t0 = time.perf_counter()
    status, raw, err = _post(url, cfg['BAILIAN_API_KEY'], payload, 60)
    elapsed = time.perf_counter() - t0
    if err is not None:
        print(f'  ERROR: {err}')
        return False
    print(f'  status={status} elapsed={elapsed:.2f}s')
    if status != 200:
        print(f'  body (first 400): {raw[:400]}')
        return False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f'  non-json body (first 200): {raw[:200]}')
        return False
    choice = ((data.get('choices') or [{}])[0]).get('message', {})
    content = choice.get('content') or ''
    reasoning = choice.get('reasoning_content') or choice.get('reasoning') or ''
    print(f'  content non-empty: {bool(content.strip())}')
    print(f'  reasoning_content non-empty: {bool(reasoning.strip())}')
    print(f'  content (first 200): {content[:200]}')
    if reasoning:
        print(f'  reasoning_chars={len(reasoning)}')
    return True


def step3_engine(cfg):
    print('[3/3] engine default preset (thinking=true)')
    _reset_thinking_probe()
    engine = GameEngine()
    try:
        result = engine.engine_move()
        print(f'  engine_move={result.get("engine_move")}  source={result.get("source")}  engine_ms={result.get("engine_ms", 0):.1f}ms  attempts={result["llm"]["attempts"]}')
        print(f'  reasoning_chars={result["llm"]["reasoning_chars"]}  thinking_param_supported={result["llm"]["thinking_param_supported"]}')
        print(f'  state fen={engine.state()["fen"]}')
    finally:
        engine.cleanup()

    print('   --- thinking=false run ---')
    _reset_thinking_probe()
    engine2 = GameEngine(thinking=False)
    try:
        result2 = engine2.engine_move()
        print(f'  engine_move={result2.get("engine_move")}  engine_ms={result2.get("engine_ms", 0):.1f}ms  attempts={result2["llm"]["attempts"]}')
        print(f'  reasoning_chars={result2["llm"]["reasoning_chars"]}  thinking_param_supported={result2["llm"]["thinking_param_supported"]}')
    finally:
        engine2.cleanup()


def main():
    try:
        cfg = _load_env_config()
    except Exception as exc:
        print(f'FATAL: config error: {exc}')
        sys.exit(2)
    print(f'model={cfg["BAILIAN_MODEL"]}  base={cfg["BAILIAN_BASE_URL"]}')
    ok1 = step1_models(cfg)
    ok2 = step2_chat(cfg) if ok1 else False
    if ok1 and ok2:
        try:
            step3_engine(cfg)
        except Exception as exc:
            print(f'  engine step failed: {exc}')
            sys.exit(3)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
