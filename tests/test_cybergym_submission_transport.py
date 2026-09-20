import asyncio
from contextlib import asynccontextmanager, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from experiments.cybergym import submission as s

CID = 'a' * 64
JOB = 'verification:' + 'b' * 32
EVAL_JOB = 'verification:' + 'c' * 32
ENDPOINT = 'http://gv-roles-sidecar-' + 'd' * 16 + ':8000'


def tool(body, *, structured=False, error=False):
    return SimpleNamespace(isError=error, content=[] if structured else [SimpleNamespace(type='text', text=json.dumps(body))],
                           structuredContent=body if structured else None)


def observation(exit_code=77, wrapper_exit=0):
    return {'candidate_id': CID, 'authority_scope': 'candidate_signal', 'environment': 'vulnerable',
            'verified': False, 'exit_code': wrapper_exit, 'execution_error': None,
            'timed_out': False, 'process_started': True, 'process_completed': True,
            'stdout_truncated': False, 'stderr_truncated': False,
            'stdout': json.dumps({'schema': 'gv-cybergym-vul-observation-v1', 'config_sha256': 'PRIVATE_PIN',
                                  'candidate_sha256': CID, 'candidate_size': 10, 'task_id': 'arvo:123',
                                  'environment': 'vulnerable', 'authority_scope': 'candidate_signal',
                                  'response': {'task_id': 'arvo:123', 'poc_id': 'public-id', 'exit_code': exit_code,
                                               'output': 'public target diagnostic', 'flag': 'PRIVATE_FLAG'}})}


def job(identifier=JOB, operation='cyber_register_candidate', state='running', result=None):
    value = {'job_id': identifier, 'tool': operation, 'state': state, 'pending': state in ('queued', 'running'), 'verified': False}
    if state == 'completed':
        value.update(result=result, result_is_retained=True)
    return tool(value)


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    async def call_tool(self, name, arguments, **kwargs):
        self.calls.append((name, arguments, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


async def no_sleep(seconds):
    pass


def test_origin_and_token_are_environment_only():
    assert s.endpoint_from_environment({'GV_SIDECAR_URL': ENDPOINT, 'GV_MCP_BEARER_TOKEN': 'secret-token'}) == (ENDPOINT + '/mcp', 'secret-token')
    assert s.endpoint_from_environment({'GV_SIDECAR_URL': ENDPOINT + '/mcp', 'GV_MCP_BEARER_TOKEN': 'secret-token'})[0] == ENDPOINT + '/mcp'


@pytest.mark.parametrize('url', ['http://localhost:8000', 'https://example.com/mcp', ENDPOINT + '/other', ENDPOINT + '?secret',
                                     ENDPOINT + '#secret', ENDPOINT.replace('http://', 'http://secret@'), ENDPOINT + '/mcp/'])
def test_bad_origin_static(url):
    with pytest.raises(s.SubmissionError, match='^sidecar_origin_invalid$'):
        s.endpoint_from_environment({'GV_SIDECAR_URL': url, 'GV_MCP_BEARER_TOKEN': 'PRIVATE_TOKEN'})


@pytest.mark.parametrize('token', [None, '', 'SECRET\nHEADER', 'secret value'])
def test_bad_token_static(token):
    with pytest.raises(s.SubmissionError, match='^sidecar_authorization_unavailable$'):
        s.endpoint_from_environment({'GV_SIDECAR_URL': ENDPOINT, 'GV_MCP_BEARER_TOKEN': token})


def test_candidate_path_absolute_and_relative_without_mutation(tmp_path):
    root = tmp_path.resolve()
    file = root / 'candidate.bin'
    file.write_bytes(b'original')
    before = file.stat()
    assert s.candidate_path(str(file), root=root) == str(file)
    assert s.candidate_path('candidate.bin', root=root, cwd=root) == str(file)
    assert file.read_bytes() == b'original'
    assert file.stat().st_mtime_ns == before.st_mtime_ns


def test_foreign_and_symlink_paths_rejected(tmp_path):
    root = tmp_path / 'root'; root.mkdir()
    outside = tmp_path / 'outside'; outside.write_bytes(b'original')
    (root / 'alias').symlink_to(outside)
    (root / 'directory-alias').symlink_to(tmp_path, target_is_directory=True)
    for path in (outside, root / 'alias', root / 'directory-alias' / 'outside', root / '..' / 'outside', root):
        with pytest.raises(s.SubmissionError, match='^candidate_path_invalid$'):
            s.candidate_path(str(path), root=root)


def test_fifo_rejected_without_blocking(tmp_path):
    import os
    file = tmp_path / 'fifo'; os.mkfifo(file)
    with pytest.raises(s.SubmissionError, match='^candidate_path_invalid$'):
        s.candidate_path(str(file), root=tmp_path.resolve())


def test_candidate_surrogate_path_is_a_static_path_error(tmp_path):
    with pytest.raises(s.SubmissionError, match='^candidate_path_invalid$'):
        s.candidate_path('bad\ud800path', root=tmp_path.resolve())


def test_structured_and_text_tool_results_are_both_supported():
    assert s.decode_tool_result(tool({'candidate_id': CID})) == {'candidate_id': CID}
    assert s.decode_tool_result(tool({'candidate_id': CID}, structured=True)) == {'candidate_id': CID}


def test_conflicting_structured_and_text_fail_closed():
    result = tool({'candidate_id': CID})
    result.structuredContent = {'candidate_id': 'f' * 64}
    with pytest.raises(s.SubmissionError, match='^tool_response_invalid$'):
        s.decode_tool_result(result)


@pytest.mark.parametrize('left,right', [(True, 1), (1, True), (1, 1.0), (1.0, 1)])
def test_structured_text_equality_preserves_json_types(left, right):
    result = tool({'value': left})
    result.structuredContent = {'value': right}
    with pytest.raises(s.SubmissionError, match='^tool_response_invalid$'):
        s.decode_tool_result(result)


def test_real_fastmcp_string_wrapper_fixture_without_network():
    from mcp.server.fastmcp.tools.base import Tool
    async def existing_tool() -> str:
        return json.dumps({'candidate_id': CID})
    content, structured = asyncio.run(Tool.from_function(existing_tool).run({}, convert_result=True))
    assert set(structured) == {'result'}
    result = SimpleNamespace(isError=False, content=content, structuredContent=structured)
    assert s.decode_tool_result(result) == {'candidate_id': CID}


@pytest.mark.parametrize('value', [float('nan'), object()])
def test_malformed_structured_job_is_protocol_error_not_infinite_reconnect(value):
    malformed = tool({'bad': value}, structured=True)
    session = FakeSession([job(), malformed])
    state = s.Submission('/task/candidate')
    @asynccontextmanager
    async def factory():
        yield session
    with pytest.raises(s.SubmissionError, match='^tool_response_invalid$'):
        asyncio.run(s.perform(state, factory, sleep=no_sleep))
    assert len(session.calls) == 2


def test_tool_error_does_not_echo_body():
    with pytest.raises(s.SubmissionError, match='^mcp_tool_rejected$'):
        s.decode_tool_result(tool({'error': 'PRIVATE_TOKEN PRIVATE_URL'}, error=True))


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '[]', 'PRIVATE_TOKEN'])
def test_json_fail_closed(raw):
    result = SimpleNamespace(isError=False, content=[SimpleNamespace(type='text', text=raw)], structuredContent=None)
    with pytest.raises(s.SubmissionError, match='^tool_response_invalid$'):
        s.decode_tool_result(result)


def test_register_then_evaluate_only_with_false_completion_and_true_target_exit():
    session = FakeSession([tool({'candidate_id': CID}), tool(observation())])
    result = asyncio.run(s.Submission('/task/candidate').run(session, sleep=no_sleep))
    assert [row[0] for row in session.calls] == ['cyber_register_candidate', 'cyber_evaluate_candidate']
    assert session.calls[1][1] == {'candidate_id': CID}
    assert result['verified'] is False
    assert result['candidate_id'] == CID
    assert result['process']['wrapper_exit_code'] == 0
    assert result['target_exit_code'] == 77
    assert 'PRIVATE_' not in json.dumps(result)
    assert all(row[2] == {'read_timeout_seconds': None} for row in session.calls)


def test_inline_completed_results_only_pin_handles_and_recover_retained_envelopes():
    register_inline = {'candidate_id': 'f' * 64, 'job_id': JOB, 'job_state': 'completed',
                       'pending': False, 'result_is_retained': True}
    evaluate_inline = {'candidate_id': 'f' * 64, 'stdout': 'not trusted JSON',
                       'job_id': EVAL_JOB, 'job_state': 'completed',
                       'pending': False, 'result_is_retained': True}
    session = FakeSession([tool(register_inline), job(state='completed', result={'candidate_id': CID}),
                           tool(evaluate_inline),
                           job(EVAL_JOB, 'cyber_evaluate_candidate', 'completed', observation())])
    result = asyncio.run(s.Submission('/task/candidate').run(session, sleep=no_sleep))
    assert result['candidate_id'] == CID and result['target_exit_code'] == 77
    assert [(name, args) for name, args, _ in session.calls] == [
        ('cyber_register_candidate', {'path': '/task/candidate'}),
        ('verification_job', {'job_id': JOB}),
        ('cyber_evaluate_candidate', {'candidate_id': CID}),
        ('verification_job', {'job_id': EVAL_JOB}),
    ]


def test_actual_dispatcher_inline_registration_shape_recovers_without_reregistering(tmp_path):
    import hashlib
    from verifier.mcp_server import Service, _ToolDispatcher
    project, state = tmp_path / 'project', tmp_path / 'state'
    project.mkdir(); state.mkdir()
    proposed = project / 'candidate'
    proposed.write_bytes(b'synthetic dispatcher fixture')
    service = Service(str(state), profile='cyber', project_root=str(project),
                      kernel_factory=lambda _: SimpleNamespace())
    dispatcher = _ToolDispatcher(service)
    first = asyncio.run(dispatcher.call('cyber_register_candidate', str(proposed), ''))
    assert service._jobs._done[first['job_id']].wait(2)
    actual_inline = asyncio.run(dispatcher.call('cyber_register_candidate', str(proposed), ''))
    assert actual_inline['job_state'] == 'completed'
    assert actual_inline['pending'] is False and actual_inline['result_is_retained'] is True
    assert not {'state', 'tool', 'result'} & actual_inline.keys()
    cid = hashlib.sha256(proposed.read_bytes()).hexdigest()

    class ActualSession:
        def __init__(self):
            self.calls = []
        async def call_tool(self, name, arguments, **kwargs):
            self.calls.append((name, arguments))
            if name == 'cyber_register_candidate':
                response = await dispatcher.call(name, arguments['path'], '')
                # The inline payload cannot select a different candidate.
                return tool({**response, 'candidate_id': 'f' * 64})
            if name == 'verification_job':
                return tool(service.verification_job(arguments['job_id']))
            assert name == 'cyber_evaluate_candidate' and arguments == {'candidate_id': cid}
            value = observation(); value['candidate_id'] = cid
            inner = json.loads(value['stdout']); inner['candidate_sha256'] = cid
            value['stdout'] = json.dumps(inner)
            return tool(value)

    session = ActualSession()
    result = asyncio.run(s.Submission(str(proposed)).run(session, sleep=no_sleep))
    assert result['candidate_id'] == cid and result['verified'] is False
    assert [name for name, _ in session.calls] == [
        'cyber_register_candidate', 'verification_job', 'cyber_evaluate_candidate']
    assert session.calls[1][1] == {'job_id': actual_inline['job_id']}
    assert len([event for event in service.log.replay()
                if event['type'] == 'cyber_candidate_registered']) == 1


@pytest.mark.parametrize('change', [lambda b: b.update(pending=0),
                                   lambda b: b.update(result_is_retained=1),
                                   lambda b: b.update(tool='cyber_finalize_candidate'),
                                   lambda b: b.update(job_state='running')])
def test_malformed_inline_metadata_is_static_not_reconnected(change):
    value = {'job_id': JOB, 'job_state': 'completed', 'pending': False, 'result_is_retained': True}
    change(value)
    session = FakeSession([tool(value)])
    state = s.Submission('/task/candidate')
    with pytest.raises(s.SubmissionError):
        asyncio.run(state.run(session, sleep=no_sleep))
    assert len(session.calls) == 1 and state.job_id == JOB


def test_initial_nested_completed_job_is_also_recovered_from_the_handle():
    session = FakeSession([job(state='completed', result={'candidate_id': 'f' * 64}),
                           job(state='completed', result={'candidate_id': CID}),
                           tool(observation())])
    result = asyncio.run(s.Submission('/task/candidate').run(session, sleep=no_sleep))
    assert result['candidate_id'] == CID
    assert session.calls[1][0:2] == ('verification_job', {'job_id': JOB})


def test_negative_candidate_is_observation_not_error_or_score():
    result = s.public_observation(observation(exit_code=0), CID)
    assert result['target_exit_code'] == 0 and result['verified'] is False
    assert 'score' not in json.dumps(result) and 'error_code' not in result


def test_observation_returns_validated_opaque_handles_without_granting_authority():
    value = observation(); value.update(artifact_id='e' * 64, effect_ref='effect:12', effect_claim_id='f' * 64)
    result = s.public_observation(value, CID)
    assert result['candidate_id'] == CID and result['artifact_id'] == 'e' * 64
    assert result['effect_ref'] == 'effect:12' and result['effect_claim_id'] == 'f' * 64
    assert result['verified'] is False and result['authority_scope'] == 'candidate_signal'


@pytest.mark.parametrize('change', [lambda b: b.update(artifact_id=True), lambda b: b.update(effect_claim_id='../private'),
                                   lambda b: b.update(effect_ref='effect:0'), lambda b: b.update(effect_ref='effect:9223372036854775808')])
def test_observation_optional_handles_fail_closed(change):
    value = observation(); change(value)
    with pytest.raises(s.SubmissionError, match='^candidate_observation_invalid$'):
        s.public_observation(value, CID)


def test_pending_polls_exact_handle_without_retry_budget_or_new_create():
    session = FakeSession([job()] + [job() for _ in range(7)] + [job(state='completed', result={'candidate_id': CID}),
                          job(EVAL_JOB, 'cyber_evaluate_candidate'), job(EVAL_JOB, 'cyber_evaluate_candidate', 'completed', observation())])
    result = asyncio.run(s.Submission('/task/candidate').run(session, sleep=no_sleep))
    assert result['target_exit_code'] == 77
    assert sum(name == 'cyber_register_candidate' for name, _, _ in session.calls) == 1
    assert sum(name == 'cyber_evaluate_candidate' for name, _, _ in session.calls) == 1
    polls = [args['job_id'] for name, args, _ in session.calls if name == 'verification_job']
    assert polls == [JOB] * 8 + [EVAL_JOB]


def test_disconnect_after_job_handle_only_reconnects_same_job():
    first = FakeSession([job(), RuntimeError('PRIVATE_TOKEN')])
    second = FakeSession([job(state='completed', result={'candidate_id': CID}), tool(observation())])
    sessions = [first, second]
    @asynccontextmanager
    async def factory():
        yield sessions.pop(0)
    result = asyncio.run(s.perform(s.Submission('/task/candidate'), factory, sleep=no_sleep))
    assert result['target_exit_code'] == 77
    assert [name for name, _, _ in first.calls] == ['cyber_register_candidate', 'verification_job']
    assert [name for name, _, _ in second.calls] == ['verification_job', 'cyber_evaluate_candidate']


def test_disconnected_evaluation_never_creates_another_evaluation():
    first = FakeSession([tool({'candidate_id': CID}), job(EVAL_JOB, 'cyber_evaluate_candidate'), RuntimeError('PRIVATE_TOKEN')])
    second = FakeSession([job(EVAL_JOB, 'cyber_evaluate_candidate', 'completed', observation())])
    sessions = [first, second]
    @asynccontextmanager
    async def factory():
        yield sessions.pop(0)
    result = asyncio.run(s.perform(s.Submission('/task/candidate'), factory, sleep=no_sleep))
    assert result['target_exit_code'] == 77
    assert [name for name, _, _ in first.calls] == ['cyber_register_candidate', 'cyber_evaluate_candidate', 'verification_job']
    assert second.calls[0][0:2] == ('verification_job', {'job_id': EVAL_JOB})
    assert len(second.calls) == 1


def test_grouped_origin_guard_failure_is_not_retried_as_a_disconnect():
    session = FakeSession([job(), ExceptionGroup('PRIVATE_URL', [s.SubmissionError('sidecar_origin_changed')])])
    @asynccontextmanager
    async def factory():
        yield session
    with pytest.raises(s.SubmissionError, match='^sidecar_origin_changed$'):
        asyncio.run(s.perform(s.Submission('/task/candidate'), factory, sleep=no_sleep))
    assert len(session.calls) == 2


@pytest.mark.parametrize('status', [401, 403])
def test_fixed_authorization_rejection_is_static_not_an_infinite_retry(status):
    import httpx
    request = httpx.Request('POST', ENDPOINT + '/mcp')
    denied = httpx.HTTPStatusError('PRIVATE_TOKEN PRIVATE_URL', request=request,
                                  response=httpx.Response(status, request=request))
    session = FakeSession([job(), ExceptionGroup('PRIVATE_BODY', [denied])])
    state = s.Submission('/task/candidate')
    @asynccontextmanager
    async def factory():
        yield session
    with pytest.raises(s.SubmissionError, match='^sidecar_authorization_rejected$'):
        asyncio.run(s.perform(state, factory, sleep=no_sleep))
    assert len(session.calls) == 2 and state.job_id == JOB


@pytest.mark.parametrize('change', [lambda b: b.update(pending=0), lambda b: b.update(result_is_retained=1),
                                   lambda b: b.update(tool='cyber_finalize_candidate'), lambda b: b.update(job_id=True)])
def test_completed_job_envelope_fields_are_strict(change):
    response = job(state='completed', result={'candidate_id': CID})
    body = json.loads(response.content[0].text); change(body)
    session = FakeSession([tool(body)])
    with pytest.raises(s.SubmissionError):
        asyncio.run(s.Submission('/task/candidate').run(session, sleep=no_sleep))
    assert len(session.calls) == 1


def test_initial_response_unknown_never_blindly_resubmits():
    session = FakeSession([RuntimeError('PRIVATE_TOKEN')])
    @asynccontextmanager
    async def factory():
        yield session
    with pytest.raises(s.SubmissionError, match='^submission_response_unknown$'):
        asyncio.run(s.perform(s.Submission('/task/candidate'), factory, sleep=no_sleep))
    assert len(session.calls) == 1


@pytest.mark.parametrize('identifier', [True, 123, 'a' * 63, 'A' * 64, '../private'])
def test_registration_identifier_strict(identifier):
    session = FakeSession([tool({'candidate_id': identifier})])
    with pytest.raises(s.SubmissionError, match='^candidate_registration_invalid$'):
        asyncio.run(s.Submission('/task/candidate').run(session, sleep=no_sleep))
    assert len(session.calls) == 1


def test_changed_job_id_and_failed_jobs_never_become_success():
    for response, code in ((job(EVAL_JOB), 'verification_job_identity_changed'), (job(state='failed'), 'verification_job_not_completed')):
        session = FakeSession([job(), response])
        with pytest.raises(s.SubmissionError, match='^' + code + '$'):
            asyncio.run(s.Submission('/task/candidate').run(session, sleep=no_sleep))


@pytest.mark.parametrize('mutate', [lambda b: b.update(verified=True), lambda b: b.update(exit_code=True),
                                   lambda b: b.update(candidate_id='f' * 64), lambda b: b.update(authority_scope='task_outcome')])
def test_eval_binding_and_typed_process_fields(mutate):
    value = observation(); mutate(value)
    with pytest.raises(s.SubmissionError):
        s.public_observation(value, CID)


def test_wrapper_failure_stays_distinct_from_target_status():
    value = observation(wrapper_exit=1)
    value['stdout'] = json.dumps({'schema': 'gv-cybergym-vul-observation-v1', 'error': 'PRIVATE_URL'})
    result = s.public_observation(value, CID)
    assert result['process']['wrapper_exit_code'] == 1
    assert result['target_exit_code'] is None and result['verified'] is False
    assert result['error_code'] == 'candidate_evaluator_error'
    assert 'PRIVATE_' not in json.dumps(result)


def test_request_origin_guard_denies_redirect_destination():
    asyncio.run(s.guard_request(SimpleNamespace(url=ENDPOINT + '/mcp', method='POST'), ENDPOINT + '/mcp'))
    for value in ('http://example.com/mcp', ENDPOINT + '/other'):
        with pytest.raises(s.SubmissionError, match='^sidecar_origin_changed$'):
            asyncio.run(s.guard_request(SimpleNamespace(url=value, method='POST'), ENDPOINT + '/mcp'))


def test_http_session_disables_proxies_redirects_and_all_sdk_read_deadlines():
    import httpx
    import mcp
    import mcp.client.streamable_http as transport
    captured = {}
    class Client:
        def __init__(self, **kwargs):
            captured['http'] = kwargs
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    @asynccontextmanager
    async def streams(url, **kwargs):
        captured['transport'] = (url, kwargs)
        yield (None, None, None)
    class Session:
        def __init__(self, *args, **kwargs):
            captured['session'] = kwargs
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def initialize(self):
            captured['initialized'] = True
    async def run():
        async with s.mcp_session(ENDPOINT + '/mcp', 'PRIVATE_TOKEN'):
            pass
    with patch.object(httpx, 'AsyncClient', Client), patch.object(transport, 'streamable_http_client', streams), patch.object(mcp, 'ClientSession', Session):
        asyncio.run(run())
    assert captured['http']['trust_env'] is False and captured['http']['follow_redirects'] is False
    assert captured['http']['timeout'] is None
    assert captured['session']['read_timeout_seconds'] is None
    assert captured['transport'][1]['terminate_on_close'] is False
    assert captured['initialized'] is True


def test_help_and_no_args_need_no_environment_or_sdk():
    with patch.object(s.os, 'environ', {}), redirect_stdout(io.StringIO()) as output:
        assert s.main([]) == 0
    assert 'submit.sh' in output.getvalue()
    with pytest.raises(SystemExit) as error, redirect_stdout(io.StringIO()):
        s.main(['-h'])
    assert error.value.code == 0


def test_cli_error_does_not_echo_private_arguments():
    with redirect_stdout(io.StringIO()) as output:
        assert s.main(['--url', 'PRIVATE_URL', '--token', 'PRIVATE_TOKEN']) == 2
    assert 'PRIVATE_' not in output.getvalue()
    assert json.loads(output.getvalue())['verified'] is False
