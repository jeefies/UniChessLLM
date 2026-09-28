"""DS 引擎离线单测（标准库 unittest，不联网）。

覆盖：env 解析、prompt 组装、render_board、解析链、thinking 自适应、
重试、六方法端到端、构造参数校验。
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (  # noqa: E402
    GameEngine,
    LLMError,
    _ENV_PATH,
    _reset_thinking_probe,
    _thinking_probe_state,
    _load_env_config,
    build_messages,
    format_san_history,
    legal_moves_text,
    parse_env_file,
    parse_engine_move,
    render_board,
)

_BAILIAN_BASE = 'https://example.test/v1'
_BAILIAN_KEY = 'sk-test-sentinel-not-real'
_BAILIAN_MODEL = 'test-model'


def _env_patch():
    return mock.patch.dict(os.environ, {
        'BAILIAN_BASE_URL': _BAILIAN_BASE,
        'BAILIAN_API_KEY': _BAILIAN_KEY,
        'BAILIAN_MODEL': _BAILIAN_MODEL,
    })


def _make_records(board, ucis):
    records = []
    for uci in ucis:
        move = chess.Move.from_uci(uci)
        records.append({
            'san': board.san(move),
            'no': board.fullmove_number,
            'white': board.turn == chess.WHITE,
        })
        board.push(move)
    return records


class TestEnvParsing(unittest.TestCase):
    def setUp(self):
        _reset_thinking_probe()

    def test_parse_env_file_bom_comment_quotes_empty(self):
        with tempfile.NamedTemporaryFile(
            'w', suffix='.env', delete=False, encoding='utf-8-sig'
        ) as f:
            f.write('BAILIAN_MODEL=foo\n')
            f.write('# comment\n')
            f.write('BAILIAN_BASE_URL="https://x"\n')
            f.write("BAILIAN_API_KEY='bar'\n")
            f.write('\n')
            f.write('EMPTY=\n')
            path = Path(f.name)
        try:
            out = parse_env_file(path)
            self.assertEqual(out['BAILIAN_MODEL'], 'foo')
            self.assertEqual(out['BAILIAN_BASE_URL'], 'https://x')
            self.assertEqual(out['BAILIAN_API_KEY'], 'bar')
            self.assertEqual(out.get('EMPTY', None), '')
        finally:
            path.unlink()

    def test_env_overrides_file(self):
        with tempfile.NamedTemporaryFile(
            'w', suffix='.env', delete=False, encoding='utf-8-sig'
        ) as f:
            f.write('BAILIAN_MODEL=file-model\n')
            f.write('BAILIAN_BASE_URL=https://file\n')
            f.write('BAILIAN_API_KEY=file-key\n')
            path = Path(f.name)
        try:
            with mock.patch.dict(os.environ, {
                'BAILIAN_BASE_URL': _BAILIAN_BASE,
                'BAILIAN_API_KEY': _BAILIAN_KEY,
                'BAILIAN_MODEL': _BAILIAN_MODEL,
            }):
                cfg = _load_env_config(path)
            self.assertEqual(cfg['BAILIAN_MODEL'], _BAILIAN_MODEL)
            self.assertEqual(cfg['BAILIAN_BASE_URL'], _BAILIAN_BASE)
            self.assertEqual(cfg['BAILIAN_API_KEY'], _BAILIAN_KEY)
        finally:
            path.unlink()

    def test_missing_key_error_message_no_key_content(self):
        with tempfile.NamedTemporaryFile(
            'w', suffix='.env', delete=False, encoding='utf-8-sig'
        ) as f:
            f.write('BAILIAN_BASE_URL=https://x\n')
            f.write('BAILIAN_API_KEY=SECRET_VALUE\n')
            path = Path(f.name)
        try:
            with self.assertRaises(RuntimeError) as cm:
                _load_env_config(path)
            msg = str(cm.exception)
            self.assertIn('BAILIAN_MODEL', msg)
            self.assertNotIn('SECRET_VALUE', msg)
        finally:
            path.unlink()


class TestRenderBoard(unittest.TestCase):
    def test_start_position(self):
        board = chess.Board()
        expected = '\n'.join([
            '  a b c d e f g h',
            '8 r n b q k b n r 8',
            '7 p p p p p p p p 7',
            '6 . . . . . . . . 6',
            '5 . . . . . . . . 5',
            '4 . . . . . . . . 4',
            '3 . . . . . . . . 3',
            '2 P P P P P P P P 2',
            '1 R N B Q K B N R 1',
            '  a b c d e f g h',
        ])
        self.assertEqual(render_board(board), expected)

    def test_mid_game(self):
        board = chess.Board()
        for uci in ('e2e4', 'e7e5', 'g1f3', 'b8c6'):
            board.push(chess.Move.from_uci(uci))
        out = render_board(board)
        self.assertIn('N', out)
        self.assertIn('N', out)
        self.assertEqual(out.splitlines()[0], '  a b c d e f g h')

    def test_promotion(self):
        board = chess.Board('8/P7/8/8/8/8/8/4K2k w - - 0 1')
        out = render_board(board)
        self.assertIn('P', out)
        self.assertEqual(out.splitlines()[2], '7 P . . . . . . . 7')

    def test_en_passant_square_rendered(self):
        fen = 'rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3'
        board = chess.Board(fen)
        out = render_board(board)
        self.assertIn('.', out)


class TestFormatSanHistory(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(format_san_history([], 24), '(none)')

    def test_limit_zero(self):
        self.assertEqual(
            format_san_history([{'san': 'e4', 'no': 1, 'white': True}], 0),
            '(none)',
        )

    def test_white_black_pairs(self):
        records = [
            {'san': 'e4', 'no': 1, 'white': True},
            {'san': 'e5', 'no': 1, 'white': False},
            {'san': 'Nf3', 'no': 2, 'white': True},
        ]
        self.assertEqual(format_san_history(records, 24), '1. e4 e5 2. Nf3')

    def test_truncation(self):
        records = [
            {'san': f'm{i}', 'no': i // 2 + 1, 'white': i % 2 == 0}
            for i in range(30)
        ]
        out = format_san_history(records, 5)
        self.assertEqual(out.split()[0], '13...')
        self.assertIn('m25', out)
        self.assertNotIn('m24', out)

    def test_black_starts(self):
        records = [{'san': 'e5', 'no': 1, 'white': False}]
        self.assertEqual(format_san_history(records, 1), '1... e5')


class TestBuildMessages(unittest.TestCase):
    def test_shape(self):
        board = chess.Board()
        records = _make_records(board, ['e2e4', 'e7e5', 'g1f3'])
        msgs = build_messages(board, records, 2)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]['role'], 'system')
        self.assertIn('UCI', msgs[0]['content'])
        self.assertEqual(msgs[0]['content'].split()[0], 'You')
        self.assertEqual(msgs[1]['role'], 'user')
        self.assertIn(board.fen(), msgs[1]['content'])
        self.assertIn('Move history (SAN): 1... e5 2. Nf3', msgs[1]['content'])
        self.assertIn('Legal moves (choose exactly one; UCI(SAN)):', msgs[1]['content'])

    def test_truncate_history(self):
        board = chess.Board()
        records = _make_records(board, ['e2e4', 'e7e5', 'g1f3'])
        msgs = build_messages(board, records, 1)
        text = msgs[1]['content']
        self.assertIn('2. Nf3', text)
        self.assertNotIn('1. e4', text)

    def test_correction_appended(self):
        board = chess.Board()
        records = _make_records(board, [])
        corr = {'raw': 'bad', 'text': 'try again'}
        msgs = build_messages(board, records, 0, correction=corr)
        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[2]['role'], 'assistant')
        self.assertEqual(msgs[3]['role'], 'user')


class TestLegalMovesText(unittest.TestCase):
    def test_start_position_contains_e4(self):
        board = chess.Board()
        text = legal_moves_text(board)
        self.assertIn('e2e4(e4)', text)
        self.assertIn('g1f3(Nf3)', text)


class TestParsing(unittest.TestCase):
    def test_whole_uci(self):
        board = chess.Board()
        move, reason = parse_engine_move('e2e4', '', board)
        self.assertIsNotNone(move)
        self.assertEqual(move.uci(), 'e2e4')
        self.assertEqual(reason, 'whole')

    def test_move_line(self):
        board = chess.Board()
        move, reason = parse_engine_move('I think MOVE: d2d4', '', board)
        self.assertIsNotNone(move)
        self.assertEqual(move.uci(), 'd2d4')
        self.assertEqual(reason, 'move-line')

    def test_san_token(self):
        board = chess.Board()
        move, reason = parse_engine_move('I will play Nf3 now', '', board)
        self.assertIsNotNone(move)
        self.assertEqual(move.uci(), 'g1f3')
        self.assertEqual(reason, 'san-token')

    def test_promotion_append_q(self):
        board = chess.Board('8/P7/8/8/8/8/8/4K2k w - - 0 1')
        move, reason = parse_engine_move('a7a8', '', board)
        self.assertIsNotNone(move)
        self.assertEqual(move.uci(), 'a7a8q')

    def test_garbage_fails(self):
        board = chess.Board()
        move, reason = parse_engine_move('hello world', '', board)
        self.assertIsNone(move)
        self.assertIn('unparseable', reason)

    def test_move_line_beats_noise_token(self):
        board = chess.Board()
        text = 'e2e4 looks bad. MOVE: d2d4'
        move, reason = parse_engine_move(text, '', board)
        self.assertIsNotNone(move)
        self.assertEqual(move.uci(), 'd2d4')
        self.assertEqual(reason, 'move-line')

    def test_content_empty_falls_back_to_reasoning(self):
        board = chess.Board()
        move, reason = parse_engine_move('', '  MOVE: g1f3  ', board)
        self.assertIsNotNone(move)
        self.assertEqual(move.uci(), 'g1f3')
        self.assertTrue(reason.startswith('reasoning:'))

    def test_both_empty_fails(self):
        board = chess.Board()
        move, reason = parse_engine_move('', None, board)
        self.assertIsNone(move)
        self.assertIn('empty', reason)


class TestThinkingProbe(unittest.TestCase):
    def setUp(self):
        _reset_thinking_probe()
        self._env_patcher = _env_patch()
        self._env_patcher.start()

    def tearDown(self):
        self._env_patcher.stop()

    def test_enable_thinking_in_payload_when_thinking_true(self):
        engine = GameEngine()
        self.addCleanup(engine.cleanup)
        captured = []

        def fake(base_url, api_key, payload, timeout_s):
            captured.append(dict(payload))
            return {'content': 'e2e4', 'reasoning_content': ''}

        engine._chat_impl = staticmethod(fake)
        engine.engine_move()
        self.assertTrue(captured[0].get('enable_thinking', False))

    def test_400_rejection_caches_unsupported_and_retries_without(self):
        engine = GameEngine()
        self.addCleanup(engine.cleanup)
        captured = []

        def fake(base_url, api_key, payload, timeout_s):
            captured.append(dict(payload))
            if len(captured) == 1 and 'enable_thinking' in payload:
                raise LLMError(
                    'unknown parameter enable_thinking',
                    status=400,
                    body='{"error":{"message":"unknown parameter enable_thinking"}}',
                )
            return {'content': 'e2e4', 'reasoning_content': ''}

        engine._chat_impl = staticmethod(fake)
        engine.engine_move()
        self.assertEqual(len(captured), 2)
        self.assertIn('enable_thinking', captured[0])
        self.assertNotIn('enable_thinking', captured[1])
        self.assertEqual(_thinking_probe_state(), False)

    def test_after_reset_probe_retries_with_param(self):
        engine = GameEngine()
        self.addCleanup(engine.cleanup)
        captured = []

        def fake(base_url, api_key, payload, timeout_s):
            captured.append(dict(payload))
            return {'content': 'e2e4', 'reasoning_content': ''}

        engine._chat_impl = staticmethod(fake)
        engine.engine_move()
        self.assertTrue(captured[0].get('enable_thinking', False))
        _reset_thinking_probe()
        captured.clear()
        engine.setup()
        engine.engine_move()
        self.assertTrue(captured[0].get('enable_thinking', False))

    def test_extra_request_merged_and_messages_rejected(self):
        engine = GameEngine(extra_request={'top_p': 0.9})
        self.addCleanup(engine.cleanup)
        captured = []

        def fake(base_url, api_key, payload, timeout_s):
            captured.append(dict(payload))
            return {'content': 'e2e4', 'reasoning_content': ''}

        engine._chat_impl = staticmethod(fake)
        engine.engine_move()
        self.assertEqual(captured[0]['top_p'], 0.9)
        with self.assertRaises(ValueError):
            GameEngine(extra_request={'messages': []})


class TestRetry(unittest.TestCase):
    def setUp(self):
        _reset_thinking_probe()
        self._env_patcher = _env_patch()
        self._env_patcher.start()

    def tearDown(self):
        self._env_patcher.stop()

    def test_retry_after_unparseable(self):
        engine = GameEngine(max_attempts=2)
        self.addCleanup(engine.cleanup)
        calls = []

        def fake(base_url, api_key, payload, timeout_s):
            calls.append(dict(payload))
            if len(calls) == 1:
                return {'content': 'garbage', 'reasoning_content': ''}
            return {'content': 'e2e4', 'reasoning_content': ''}

        engine._chat_impl = staticmethod(fake)
        result = engine.engine_move()
        self.assertEqual(result['engine_move'], 'e2e4')
        self.assertEqual(result['llm']['attempts'], 2)
        self.assertEqual(len(calls), 2)
        self.assertIn('e2e4', calls[1]['messages'][-1]['content'])

    def test_exhaust_retries_raises(self):
        engine = GameEngine(max_attempts=2)
        self.addCleanup(engine.cleanup)

        def fake(base_url, api_key, payload, timeout_s):
            return {'content': 'garbage', 'reasoning_content': ''}

        engine._chat_impl = staticmethod(fake)
        with self.assertRaises(LLMError) as cm:
            engine.engine_move()
        self.assertIn('2 次尝试', str(cm.exception))
        self.assertNotIn(_BAILIAN_KEY, str(cm.exception))

    def test_http_failure_no_key_leak(self):
        engine = GameEngine(max_attempts=2)
        self.addCleanup(engine.cleanup)

        def fake(base_url, api_key, payload, timeout_s):
            raise LLMError('boom', status=500, body='server error')

        engine._chat_impl = staticmethod(fake)
        with self.assertRaises(LLMError) as cm:
            engine.engine_move()
        self.assertNotIn(_BAILIAN_KEY, str(cm.exception))


class TestConstructor(unittest.TestCase):
    def setUp(self):
        _reset_thinking_probe()
        self._env_patcher = _env_patch()
        self._env_patcher.start()

    def tearDown(self):
        self._env_patcher.stop()

    def test_rejects_unknown_kwarg(self):
        with self.assertRaises(TypeError):
            GameEngine(ckpt='whatever')

    def test_validates_timeout(self):
        with self.assertRaises(ValueError):
            GameEngine(timeout_s=0)

    def test_validates_temperature(self):
        with self.assertRaises(ValueError):
            GameEngine(temperature=3)

    def test_extra_request_must_be_dict(self):
        with self.assertRaises(TypeError):
            GameEngine(extra_request='bad')
        with self.assertRaises(ValueError):
            GameEngine(extra_request={'messages': []})

    def test_model_from_env_default(self):
        engine = GameEngine()
        self.addCleanup(engine.cleanup)
        self.assertEqual(engine.model, _BAILIAN_MODEL)


class TestSixMethodE2E(unittest.TestCase):
    def setUp(self):
        _reset_thinking_probe()
        self._env_patcher = _env_patch()
        self._env_patcher.start()

    def tearDown(self):
        self._env_patcher.stop()

    def _make(self, **kwargs):
        engine = GameEngine(**kwargs)
        self.addCleanup(engine.cleanup)
        return engine

    def test_setup_human_engine_round_trip(self):
        engine = self._make()
        st = engine.setup()
        self.assertIn('fen', st)
        self.assertEqual(st['fen'], chess.STARTING_FEN)
        engine.human_move('e2e4')
        self.assertEqual(len(engine.state()['history']), 1)

        def fake(base_url, api_key, payload, timeout_s):
            return {'content': 'e7e5', 'reasoning_content': ''}

        engine._chat_impl = staticmethod(fake)
        result = engine.engine_move()
        self.assertIn('engine_move', result)
        self.assertEqual(result['engine_move'], 'e7e5')
        self.assertEqual(len(engine.state()['history']), 2)

    def test_engine_rejects_terminal(self):
        engine = self._make()
        engine.setup('6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1')
        while not engine.state()['game_over']:
            legal = engine.state()['legal_moves']
            if not legal:
                break
            engine.human_move(legal[0])
        with self.assertRaises(ValueError):
            engine.engine_move()

    def test_undo_rolls_back(self):
        engine = self._make()
        engine.human_move('e2e4')
        self.assertEqual(len(engine.state()['history']), 1)
        engine.undo()
        self.assertEqual(len(engine.state()['history']), 0)

    def test_state_shape_contains_eval_none_and_llm(self):
        engine = self._make()

        def fake(base_url, api_key, payload, timeout_s):
            return {'content': 'e7e5', 'reasoning_content': 't' * 42}

        engine._chat_impl = staticmethod(fake)
        engine.human_move('e2e4')
        result = engine.engine_move()
        st = engine.state()
        for k in (
            'fen', 'legal_moves', 'history', 'san_history', 'last_move',
            'last_move_san', 'last_move_actor', 'engine_ms', 'in_check',
            'game_over', 'result', 'eval', 'llm',
        ):
            self.assertIn(k, st)
        self.assertIsNone(st['eval'])
        self.assertIsNotNone(st['llm'])
        self.assertEqual(result['engine_move'], 'e7e5')
        self.assertEqual(result['llm']['model'], _BAILIAN_MODEL)
        self.assertEqual(result['llm']['reasoning_chars'], 42)
        self.assertTrue(result['llm']['thinking'])

    def test_cleanup_idempotent(self):
        engine = self._make()
        engine.cleanup()
        engine.cleanup()
        st = engine.state()
        self.assertEqual(st['san_history'], [])
        self.assertIsNone(st['llm'])

    def test_implemented_and_no_kit_factory(self):
        self.assertTrue(GameEngine.IMPLEMENTED)
        self.assertFalse(hasattr(GameEngine, 'KIT_FACTORY'))

    def test_invalid_fen_rejected(self):
        engine = self._make()
        with self.assertRaises(ValueError):
            engine.setup('not a fen')

    def test_human_move_invalid_form(self):
        engine = self._make()
        with self.assertRaises(ValueError):
            engine.human_move('e2e5')

    def test_isolated_instances(self):
        e1 = self._make()
        e2 = self._make()
        e1.human_move('e2e4')
        e2.human_move('d2d4')
        self.assertNotEqual(e1.state()['fen'], e2.state()['fen'])


if __name__ == '__main__':
    unittest.main()
