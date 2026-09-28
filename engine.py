"""DS：大模型（阿里云百炼 OpenAI 兼容端点）国际象棋引擎。

六方法契约对齐 Server/models/__init__.py；HTTP 客户端仅使用标准库 urllib。
构造参数白名单：model / timeout_s / max_attempts / temperature / max_tokens /
history_plies / thinking / extra_request；未知 kwargs 抛 TypeError。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import chess

_ENV_PATH = Path(__file__).resolve().parent / '.env'
_REQUIRED_KEYS = ('BAILIAN_BASE_URL', 'BAILIAN_API_KEY', 'BAILIAN_MODEL')
_ALLOWED_KWARGS = frozenset({
    'model', 'timeout_s', 'max_attempts', 'temperature',
    'max_tokens', 'history_plies', 'thinking', 'extra_request',
})
_MOVE_LINE_RE = re.compile(r'MOVE\s*[:：]\s*(\S+)', re.IGNORECASE)
_UCI_TOKEN_RE = re.compile(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b')
_DEBUG_LOG_PATH = os.environ.get('DS_DEBUG_LOG')


def _debug_log(text: str) -> None:
    if not _DEBUG_LOG_PATH:
        return
    try:
        with open(_DEBUG_LOG_PATH, 'a', encoding='utf-8') as fh:
            fh.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {text}\n')
    except Exception:
        pass


class LLMError(RuntimeError):
    """百炼 API 调用失败或输出无法解析。绝不包含 API key。"""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def _truncate(text: str, n: int = 200) -> str:
    return text if len(text) <= n else text[:n] + '...'


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        raw = path.read_text(encoding='utf-8-sig')
    except FileNotFoundError:
        return values
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _load_env_config(env_path: Path = _ENV_PATH) -> dict[str, str]:
    cfg = parse_env_file(env_path)
    for key in _REQUIRED_KEYS:
        env_value = os.environ.get(key)
        if env_value is not None and env_value != '':
            cfg[key] = env_value
    missing = [k for k in _REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise RuntimeError(
            f'DS 引擎配置缺失：缺少 {", ".join(missing)}；'
            f'请在 {env_path} 或对应环境变量中提供。'
        )
    return {k: cfg[k] for k in _REQUIRED_KEYS}


def _http_chat(base_url: str, api_key: str, payload: dict, timeout_s: float) -> dict:
    url = base_url.rstrip('/') + '/chat/completions'
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        method='POST',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
    )
    if _DEBUG_LOG_PATH:
        safe_payload = json.dumps(payload, ensure_ascii=False)
        _debug_log(f'REQUEST {url}\n  payload: {safe_payload[:1000]}')
    status = None
    raw = ''
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
        raise LLMError(f'百炼端点网络错误：{exc.reason}') from exc
    except Exception as exc:
        raise LLMError(f'百炼 HTTP 客户端错误：{exc}') from exc

    if status is None:
        raise LLMError('百炼 HTTP 响应缺少状态码')

    if _DEBUG_LOG_PATH:
        _debug_log(f'RESPONSE status={status}\n  body: {raw[:2000]}')

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise LLMError(
            f'百炼响应不是合法 JSON（HTTP {status}）：{raw[:200]}',
            status=status,
            body=raw[:500],
        )

    if not (200 <= status < 300):
        raise LLMError(
            f'百炼 API 返回 HTTP {status}：{_extract_error_text(data)[:300]}',
            status=status,
            body=raw[:500],
        )

    try:
        message = data['choices'][0]['message']
    except (KeyError, IndexError, TypeError):
        raise LLMError(
            f'百炼响应缺少 choices[0].message：{raw[:200]}',
            status=status,
        )

    content = message.get('content') or ''
    reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
    if not isinstance(content, str):
        content = str(content)
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    if not content.strip() and not reasoning.strip():
        if _DEBUG_LOG_PATH:
            _debug_log(f'EMPTY_RESPONSE status={status} content={content!r} reasoning={reasoning!r}')
        raise LLMError('百炼响应 content 与 reasoning_content 均为空', status=status)

    return {'content': content, 'reasoning_content': reasoning}


def _extract_error_text(data: Any) -> str:
    err = data.get('error') if isinstance(data, dict) else None
    if isinstance(err, dict):
        return err.get('message') or err.get('type') or str(err)
    return str(err) if err else str(data)[:200]


_thinking_probe_lock = threading.Lock()
_thinking_param_supported: bool | None = None


def _thinking_probe_state() -> bool | None:
    with _thinking_probe_lock:
        return _thinking_param_supported


def _mark_thinking_param_unsupported() -> None:
    global _thinking_param_supported
    with _thinking_probe_lock:
        _thinking_param_supported = False


def _reset_thinking_probe() -> None:
    global _thinking_param_supported
    with _thinking_probe_lock:
        _thinking_param_supported = None


_THINKING_REJECT_KEYWORDS = ('enable_thinking', 'unknown', 'unsupported', 'invalid')


def _is_param_rejection(exc: LLMError) -> bool:
    if exc.status != 400 or not exc.body:
        return False
    text = exc.body.lower()
    return any(k in text for k in _THINKING_REJECT_KEYWORDS)


def render_board(board: chess.Board) -> str:
    lines = ['  a b c d e f g h']
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            row.append(piece.symbol() if piece else '.')
        lines.append(f'{rank + 1} ' + ' '.join(row) + f' {rank + 1}')
    lines.append('  a b c d e f g h')
    return '\n'.join(lines)


def legal_moves_text(board: chess.Board) -> str:
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    return ' '.join(f'{m.uci()}({board.san(m)})' for m in moves)


def format_san_history(records: list[dict], limit: int) -> str:
    if limit <= 0 or not records:
        return '(none)'
    shown = records[-limit:]
    parts: list[str] = []
    for rec in shown:
        san = rec['san']
        no = rec['no']
        white = rec['white']
        if white:
            parts.append(f'{no}. {san}')
        elif not parts:
            parts.append(f'{no}... {san}')
        else:
            parts.append(san)
    return ' '.join(parts)


def build_messages(
    board: chess.Board,
    records: list[dict],
    history_plies: int,
    correction: dict | None = None,
) -> list[dict]:
    color = 'White' if board.turn == chess.WHITE else 'Black'
    system = (
        f'You are a chess grandmaster. You are playing as {color}. '
        'Think step by step internally if you want, but your FINAL answer '
        'must be exactly ONE UCI move token (e.g. e2e4, g1f3, e7e8q for '
        'promotion). No other words, punctuation or explanation.'
    )
    user = '\n'.join([
        f'Position (FEN): {board.fen()}',
        f'Side to move: {color}',
        f'In check: {"yes" if board.is_check() else "no"}',
        f'Castling rights: {board.castling_xfen() or "-"}',
        f"En passant: {chess.square_name(board.ep_square) if board.ep_square else '-'}",
        f'Halfmove clock: {board.halfmove_clock}',
        f'Move history (SAN): {format_san_history(records, history_plies)}',
        'Board (White at bottom, uppercase = White, lowercase = Black, "." = empty):',
        render_board(board),
        f'Legal moves (choose exactly one; UCI(SAN)): {legal_moves_text(board)}',
        'Choose the best move. Reply with ONLY its UCI token (the part before the parentheses).',
    ])
    messages: list[dict] = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]
    if correction is not None:
        messages.append({'role': 'assistant', 'content': correction['raw']})
        messages.append({'role': 'user', 'content': correction['text']})
    return messages


def _build_correction_message(board: chess.Board, previous_raw: str, reason: str) -> str:
    shown = _truncate(previous_raw, 200) if previous_raw else '(empty)'
    return (
        f'Your previous reply was not a valid move ({reason}). '
        f'You replied: "{shown}". '
        'Reply with ONLY one UCI move token chosen from this legal move list: '
        f'{legal_moves_text(board)}'
    )


def _uci_candidates(token: str, board: chess.Board) -> list[str]:
    token = token.strip().lower()
    if not token:
        return []
    candidates = [token]
    if len(token) == 4:
        try:
            from_sq = chess.parse_square(token[:2])
            to_sq = chess.parse_square(token[2:])
        except ValueError:
            return candidates
        for p in ('q', 'r', 'b', 'n'):
            piece = chess.Piece.from_symbol(p.upper())
            m = chess.Move(from_sq, to_sq, promotion=piece.piece_type)
            if m in board.legal_moves:
                candidates.append(token + p)
    return candidates


def _match_legal_uci(token: str, board: chess.Board):
    for cand in _uci_candidates(token, board):
        try:
            move = chess.Move.from_uci(cand)
        except (ValueError, AssertionError):
            continue
        if move in board.legal_moves:
            return move
    return None


def _parse_move_from_text(text: str, board: chess.Board):
    if not text:
        return None, 'empty'
    text = text.strip()
    move = _match_legal_uci(text, board)
    if move is not None:
        return move, 'whole'
    for token in reversed(_MOVE_LINE_RE.findall(text)):
        token = token.strip('"\'`.,;:!?').lower()
        move = _match_legal_uci(token, board)
        if move is not None:
            return move, 'move-line'
    for token in reversed(_UCI_TOKEN_RE.findall(text)):
        token = token.lower()
        move = _match_legal_uci(token, board)
        if move is not None:
            return move, 'uci-token'
    for token in reversed(text.split()):
        san = token.strip('.,;:!?\'"`')
        if not san:
            continue
        try:
            move = board.parse_san(san)
        except Exception:
            continue
        if move in board.legal_moves:
            return move, 'san-token'
    return None, 'unparseable'


def parse_engine_move(content: str, reasoning: str | None, board: chess.Board):
    move, reason = _parse_move_from_text(content, board)
    if move is not None:
        return move, reason
    if reasoning:
        move2, reason2 = _parse_move_from_text(reasoning, board)
        if move2 is not None:
            return move2, 'reasoning:' + reason2
    return None, f'content:{reason}; reasoning:{(reason or "empty")}'


class GameEngine:
    IMPLEMENTED: bool = True
    NOT_IMPLEMENTED_REASON: str = ''
    _chat_impl = staticmethod(_http_chat)

    def __init__(self, **kwargs: Any) -> None:
        unknown = sorted(set(kwargs) - _ALLOWED_KWARGS)
        if unknown:
            raise TypeError(f'unsupported DS engine kwargs: {unknown}')
        self._env = _load_env_config()
        self._base_url = self._env['BAILIAN_BASE_URL']
        self._api_key = self._env['BAILIAN_API_KEY']
        self.model = str(kwargs.get('model') or self._env['BAILIAN_MODEL']).strip()
        if not self.model:
            raise ValueError('model 不能为空')
        self.timeout_s = float(kwargs.get('timeout_s', 30.0))
        if self.timeout_s <= 0:
            raise ValueError('timeout_s 必须 > 0')
        self.max_attempts = int(kwargs.get('max_attempts', 2))
        if self.max_attempts < 1:
            raise ValueError('max_attempts 必须 >= 1')
        self.temperature = float(kwargs.get('temperature', 0.2))
        if not 0 <= self.temperature <= 2:
            raise ValueError('temperature 必须在 [0, 2] 内')
        self.max_tokens = int(kwargs.get('max_tokens', 512))
        if self.max_tokens < 1:
            raise ValueError('max_tokens 必须 >= 1')
        self.history_plies = int(kwargs.get('history_plies', 24))
        if self.history_plies < 0:
            raise ValueError('history_plies 必须 >= 0')
        self.thinking = bool(kwargs.get('thinking', True))
        extra = kwargs.get('extra_request', {})
        if extra is None:
            extra = {}
        if not isinstance(extra, dict):
            raise TypeError('extra_request 必须是 dict')
        if 'messages' in extra:
            raise ValueError('extra_request 不得包含 messages（会覆盖整个对话）')
        self.extra_request = dict(extra)

        self._lock = threading.RLock()
        self._board = chess.Board()
        self._san_history: list[str] = []
        self._move_records: list[dict[str, Any]] = []
        self._last_llm: dict[str, Any] | None = None
        self.setup()

    def _build_payload(self, messages: list[dict]) -> dict:
        payload: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'stream': False,
        }
        if _thinking_probe_state() is not False:
            payload['enable_thinking'] = bool(self.thinking)
        payload.update(self.extra_request)
        return payload

    def _chat_once(self, messages: list[dict]) -> dict:
        payload = self._build_payload(messages)
        try:
            return self._chat_impl(self._base_url, self._api_key, payload, self.timeout_s)
        except LLMError as exc:
            if _is_param_rejection(exc) and 'enable_thinking' in payload:
                _mark_thinking_param_unsupported()
                degraded = dict(payload)
                degraded.pop('enable_thinking', None)
                return self._chat_impl(self._base_url, self._api_key, degraded, self.timeout_s)
            raise

    def setup(self, fen: str | None = None) -> dict:
        with self._lock:
            if fen is None:
                board = chess.Board()
            else:
                if not isinstance(fen, str):
                    raise TypeError(f'fen must be str or None, got {type(fen).__name__}')
                board = chess.Board(fen)
                if not board.is_valid():
                    raise ValueError(f'invalid FEN position: {fen!r}')
            self._board = board
            self._san_history = []
            self._move_records = []
            self._last_llm = None
            return self.state()

    def human_move(self, uci: str) -> dict:
        with self._lock:
            if not isinstance(uci, str):
                raise TypeError(f'uci must be str, got {type(uci).__name__}')
            move = chess.Move.from_uci(uci)
            if move not in self._board.legal_moves:
                raise ValueError(f'illegal move {uci!r} for current position')
            san = self._board.san(move)
            no = self._board.fullmove_number
            white = self._board.turn == chess.WHITE
            self._board.push(move)
            self._san_history.append(san)
            self._move_records.append({
                'uci': move.uci(),
                'san': san,
                'actor': 'human',
                'no': no,
                'white': white,
            })
            return self.state()

    def engine_move(self) -> dict:
        with self._lock:
            outcome = self._board.outcome(claim_draw=True)
            if outcome is not None:
                raise ValueError('cannot choose an engine move from a terminal position')
            started = time.perf_counter()
            messages = build_messages(self._board, self._move_records, self.history_plies)
            previous_raw: str = ''
            attempts = 0
            reason = ''
            move = None
            for attempt in range(1, self.max_attempts + 1):
                attempts = attempt
                try:
                    reply = self._chat_once(messages)
                except LLMError as exc:
                    reason = str(exc)
                    continue
                content = reply.get('content') or ''
                reasoning = reply.get('reasoning_content') or ''
                previous_raw = _truncate(content, 200)
                move, reason = parse_engine_move(content, reasoning, self._board)
                if move is not None:
                    break
                messages.append({'role': 'assistant', 'content': content})
                messages.append({
                    'role': 'user',
                    'content': _build_correction_message(self._board, previous_raw, reason),
                })
            else:
                raise LLMError(
                    f'DS 引擎在 {attempts} 次尝试后仍未获得合法着法：{reason}'
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            san = self._board.san(move)
            no = self._board.fullmove_number
            white = self._board.turn == chess.WHITE
            self._board.push(move)
            self._san_history.append(san)
            self._move_records.append({
                'uci': move.uci(),
                'san': san,
                'actor': 'engine',
                'engine_ms': elapsed_ms,
                'no': no,
                'white': white,
            })
            llm_meta: dict[str, Any] = {
                'engine_move': move.uci(),
                'model': self.model,
                'attempts': attempts,
                'raw': previous_raw,
                'reasoning_chars': len(reasoning),
                'thinking': bool(self.thinking),
                'thinking_param_supported': _thinking_probe_state(),
            }
            self._last_llm = llm_meta
            return {
                'engine_move': move.uci(),
                'source': 'llm',
                'engine_ms': elapsed_ms,
                'llm': llm_meta,
                'state': self.state(),
            }

    def state(self) -> dict:
        with self._lock:
            outcome = self._board.outcome(claim_draw=True)
            last = self._move_records[-1] if self._move_records else None
            engine_ms = None
            if last is not None and last.get('actor') == 'engine':
                engine_ms = last.get('engine_ms')
            return {
                'fen': self._board.fen(),
                'turn': 'white' if self._board.turn == chess.WHITE else 'black',
                'legal_moves': sorted(m.uci() for m in self._board.legal_moves),
                'history': [m.uci() for m in self._board.move_stack],
                'san_history': list(self._san_history),
                'last_move': None if last is None else last['uci'],
                'last_move_san': None if last is None else last['san'],
                'last_move_actor': None if last is None else last['actor'],
                'engine_ms': engine_ms,
                'in_check': self._board.is_check(),
                'game_over': outcome is not None,
                'result': None if outcome is None else outcome.result(),
                'eval': None,
                'llm': dict(self._last_llm) if self._last_llm else None,
            }

    def undo(self) -> dict:
        with self._lock:
            for _ in range(min(2, len(self._board.move_stack))):
                self._board.pop()
                if self._san_history:
                    self._san_history.pop()
                if self._move_records:
                    self._move_records.pop()
            self._last_llm = None
            return self.state()

    def cleanup(self) -> None:
        with self._lock:
            self._san_history = []
            self._move_records = []
            self._last_llm = None
            return None


__all__ = ['GameEngine', 'LLMError', '_reset_thinking_probe']
