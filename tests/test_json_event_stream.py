"""Streaming policy metadata never trusts an unvalidated/truncated JSON event."""
import json

import pytest

from verifier.llm.json_event_stream import JsonEventStream


def parse(raw, *, chunk_size=7, **limits):
    parser = JsonEventStream(**limits)
    result = []
    for start in range(0, len(raw), chunk_size):
        result.extend(parser.feed(raw[start:start + chunk_size]))
        assert len(parser.token) <= parser.string_limit
        assert parser.frame_size <= parser.frame_limit or parser.invalid
    terminal = parser.finish()
    if terminal is not None:
        result.append(terminal)
    return result


def test_large_body_before_actual_search_type_preserves_structure_and_redacts_text():
    raw = json.dumps({'type': 'item.completed', 'item': {
        'query': 'private query' * 200000, 'type': 'web_search', 'id': 'known'}}).encode()
    [(event, redacted)] = parse(raw, chunk_size=65536)
    assert event == {'type': 'item.completed', 'item': {
        'query': None, 'type': 'web_search', 'id': 'known'}}
    assert redacted and 'private query' not in json.dumps(event)


def test_forged_event_identifiers_in_long_command_output_are_not_structure():
    value = {'type': 'item.completed', 'item': {'type': 'command_execution',
        'aggregated_output': '{"type":"web_search"}' * 10000}}
    [(event, redacted)] = parse(json.dumps(value).encode(), chunk_size=65536)
    assert redacted and event['item']['type'] == 'command_execution'
    assert event['item']['aggregated_output'] is None


def test_chunked_unicode_escapes_quotes_and_multiple_terminal_lines():
    values = [{'type': 'item.completed', 'item': {'type': 'agent_message',
        'text': '汉字\\\"\n'}}, {'type': 'turn.completed'}]
    raw = '\n'.join(json.dumps(v, ensure_ascii=False) for v in values).encode()
    assert parse(raw, chunk_size=1) == [(v, False) for v in values]


@pytest.mark.parametrize('tail', [b'\\q"}}', b'\\u00xz"}}', b'\x00"}}',
    b'\xff"}}', b'"}} trailing', b'"}} ,', b'"}', b'\\', b'\xe6'])
def test_malformed_tail_after_redacted_body_still_fails_closed(tail):
    raw = b'{"type":"item.completed","item":{"type":"web_search","query":"' + b'x' * 10000 + tail
    [(event, _)] = parse(raw, chunk_size=11)
    assert event is None


def test_oversized_key_and_structural_frame_fail_closed_instead_of_hiding_fields():
    [(event, _)] = parse(json.dumps({'type': 'item.completed', 'x' * 10000: 'value'}).encode())
    assert event is None
    [(event, _)] = parse(json.dumps({'type': 'item.completed', 'array': [0] * 2000}).encode(), frame_limit=256)
    assert event is None


def test_invalid_line_does_not_hide_following_valid_search_event():
    raw = b'{bad}\n' + json.dumps({'type': 'item.completed', 'item': {'type': 'web_search'}}).encode() + b'\n'
    assert parse(raw) == [(None, False), ({'type': 'item.completed', 'item': {'type': 'web_search'}}, False)]


@pytest.mark.parametrize('suffix', [b'"', b'"' + b'x' * 10000, b'"\\u00'])
def test_unfinished_trailing_string_cannot_disappear_from_an_otherwise_valid_event(suffix):
    raw = json.dumps({'type': 'item.completed', 'item': {'type': 'web_search'}}).encode() + suffix
    [(event, _)] = parse(raw)
    assert event is None
