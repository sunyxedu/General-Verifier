"""Public submit.sh transport: register and observe one immutable candidate.

This is not completion, scoring, or a trusted script-execution receipt. A
pending job is recovered through the same MCP handle without creating work.
Only the host's existing evaluator produces the candidate observation.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit

MAX_TOOL_BYTES = 1024 * 1024


class SubmissionError(ValueError):
    """Static error codes only; never include arguments or exception bodies."""


def endpoint_from_environment(environment) -> tuple[str, str]:
    value, token = environment.get('GV_SIDECAR_URL'), environment.get('GV_MCP_BEARER_TOKEN')
    if type(value) is not str or len(value) > 256:
        raise SubmissionError('sidecar_origin_invalid')
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme == 'http' and parsed.username is None and parsed.password is None
                 and parsed.port == 8000 and parsed.query == '' and parsed.fragment == ''
                 and parsed.path in ('', '/', '/mcp') and parsed.hostname is not None
                 and re.fullmatch(r'gv-roles-sidecar-[0-9a-f]{16}', parsed.hostname))
    except ValueError:
        valid = False
    if not valid or parsed.netloc != parsed.hostname + ':8000' or value != 'http://' + parsed.netloc + parsed.path:
        raise SubmissionError('sidecar_origin_invalid')
    if type(token) is not str or not re.fullmatch(r'[A-Za-z0-9._~+/-]{1,4096}={0,2}', token):
        raise SubmissionError('sidecar_authorization_unavailable')
    return 'http://' + parsed.netloc + '/mcp', token


def candidate_path(value: str, *, root: Path = Path('/task'), cwd: Path | None = None) -> str:
    if type(value) is not str or not value:
        raise SubmissionError('candidate_path_invalid')
    try:
        encoded_size = len(value.encode('utf-8'))
    except UnicodeError:
        raise SubmissionError('candidate_path_invalid') from None
    if encoded_size > 4096 or any(ord(c) < 32 for c in value):
        raise SubmissionError('candidate_path_invalid')
    candidate = Path(value)
    if '..' in candidate.parts:
        raise SubmissionError('candidate_path_invalid')
    candidate = candidate if candidate.is_absolute() else (cwd or Path.cwd()) / candidate
    absolute = Path(os.path.abspath(candidate))
    root = Path(os.path.abspath(root))
    try:
        parts = absolute.relative_to(root).parts
        if not parts or root.resolve(strict=True) != root:
            raise SubmissionError('candidate_path_invalid')
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for index, part in enumerate(parts):
                flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
                if index != len(parts) - 1:
                    flags |= os.O_DIRECTORY
                next_fd = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise SubmissionError('candidate_path_invalid')
        finally:
            os.close(fd)
    except (OSError, ValueError, RuntimeError) as error:
        if type(error) is SubmissionError:
            raise
        raise SubmissionError('candidate_path_invalid') from None
    return str(absolute)


def _object(raw: str) -> dict:
    if type(raw) is not str:
        raise SubmissionError('tool_response_invalid')
    try:
        if len(raw.encode('utf-8')) > MAX_TOOL_BYTES:
            raise SubmissionError('tool_response_invalid')
    except UnicodeError:
        raise SubmissionError('tool_response_invalid') from None
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise SubmissionError('tool_response_invalid')
            value[key] = item
        return value
    try:
        result = json.loads(raw, object_pairs_hook=pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError):
        raise SubmissionError('tool_response_invalid') from None
    if type(result) is not dict:
        raise SubmissionError('tool_response_invalid')
    return result


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise SubmissionError('tool_response_invalid') from None


def decode_tool_result(result) -> dict:
    if getattr(result, 'isError', None) is not False:
        raise SubmissionError('mcp_tool_rejected')
    content, structured = getattr(result, 'content', []), getattr(result, 'structuredContent', None)
    if type(content) is not list or len(content) > 1:
        raise SubmissionError('tool_response_invalid')
    text = None
    if content:
        block = content[0]
        if getattr(block, 'type', None) != 'text':
            raise SubmissionError('tool_response_invalid')
        text = _object(getattr(block, 'text', None))
    if structured is not None:
        if type(structured) is not dict:
            raise SubmissionError('tool_response_invalid')
        structured = _object(_canonical(structured))
        # FastMCP's annotated ->str tools retain text and wrap the same string.
        value = _object(structured['result']) if set(structured) == {'result'} and type(
            structured['result']) is str else structured
        if text is not None and _canonical(text) != _canonical(value):
            raise SubmissionError('tool_response_invalid')
        return value
    if text is None:
        raise SubmissionError('tool_response_invalid')
    return text


def _digest(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _no_completion(body):
    if ('verified' in body and body['verified'] is not False) or body.get('materialized') is True or body.get('verdict') == 'verified':
        raise SubmissionError('unexpected_completion_response')


def public_observation(body: dict, candidate_id: str) -> dict:
    _no_completion(body)
    if body.get('candidate_id') != candidate_id or body.get('authority_scope') != 'candidate_signal' or body.get('environment') != 'vulnerable':
        raise SubmissionError('candidate_observation_invalid')
    process = {}
    for key in ('timed_out', 'process_started', 'process_completed', 'stdout_truncated', 'stderr_truncated'):
        if type(body.get(key)) is not bool:
            raise SubmissionError('candidate_observation_invalid')
        process[key] = body[key]
    wrapper_exit = body.get('exit_code')
    if wrapper_exit is not None and (type(wrapper_exit) is not int or not -(2**31) <= wrapper_exit < 2**31):
        raise SubmissionError('candidate_observation_invalid')
    process.update(wrapper_exit_code=wrapper_exit, execution_error_present=body.get('execution_error') is not None)
    result = {'schema': 'gv-public-candidate-submission-v1', 'authority_scope': 'candidate_signal',
              'candidate_id': candidate_id,
              'transport_status': 'completed', 'verified': False, 'process': process,
              'target_exit_code': None, 'observation': None}
    for key in ('artifact_id', 'effect_claim_id'):
        if key in body:
            if not _digest(body[key]):
                raise SubmissionError('candidate_observation_invalid')
            result[key] = body[key]
    if 'effect_ref' in body:
        effect_ref = body['effect_ref']
        if (type(effect_ref) is not str or re.fullmatch(r'effect:[1-9][0-9]{0,18}', effect_ref) is None
                or int(effect_ref[7:]) > 2**63 - 1):
            raise SubmissionError('candidate_observation_invalid')
        result['effect_ref'] = effect_ref
    if body['stdout_truncated']:
        result['error_code'] = 'candidate_observation_truncated'
        return result
    observation = _object(body.get('stdout'))
    if observation.get('schema') != 'gv-cybergym-vul-observation-v1':
        raise SubmissionError('candidate_observation_invalid')
    if observation.get('error') is not None:
        result['error_code'] = 'candidate_evaluator_error'
        return result
    if (observation.get('candidate_sha256') != candidate_id or observation.get('environment') != 'vulnerable'
            or observation.get('authority_scope') != 'candidate_signal' or type(observation.get('candidate_size')) is not int
            or not 0 <= observation['candidate_size'] <= 2**63 - 1):
        raise SubmissionError('candidate_observation_invalid')
    response = observation.get('response')
    if (type(response) is not dict or type(response.get('exit_code')) is not int
            or not -(2**31) <= response['exit_code'] < 2**31 or type(response.get('output')) is not str
            or type(response.get('task_id')) is not str or response['task_id'] != observation.get('task_id')
            or type(response.get('poc_id')) is not str):
        raise SubmissionError('candidate_observation_invalid')
    # Public target observations only: no config/checksum/flag/pin/raw wrapper body.
    result['target_exit_code'] = response['exit_code']
    result['observation'] = {key: response[key] for key in ('task_id', 'exit_code', 'output', 'poc_id')}
    return result


@dataclass
class Submission:
    path: str
    candidate_id: str | None = None
    job_id: str | None = None
    initial_dispatched: bool = False

    @property
    def operation(self):
        return 'cyber_evaluate_candidate' if self.candidate_id else 'cyber_register_candidate'

    async def run(self, session, *, sleep=asyncio.sleep):
        while True:
            if self.job_id:
                tool, arguments = 'verification_job', {'job_id': self.job_id}
            else:
                if self.initial_dispatched:
                    raise SubmissionError('submission_response_unknown')
                tool = self.operation
                arguments = {'candidate_id': self.candidate_id} if self.candidate_id else {'path': self.path}
                self.initial_dispatched = True
            body = decode_tool_result(await session.call_tool(tool, arguments, read_timeout_seconds=None))
            if 'job_id' in body:
                job_id = body['job_id']
                if type(job_id) is not str or not re.fullmatch(r'verification:[0-9a-f]{32}', job_id):
                    raise SubmissionError('verification_job_invalid')
                if self.job_id is not None and self.job_id != job_id:
                    raise SubmissionError('verification_job_identity_changed')
                # Keep a received handle even when later metadata is malformed.
                self.job_id = job_id
            _no_completion(body)
            if body.get('error') is not None:
                raise SubmissionError('verification_job_unavailable' if tool == 'verification_job' else 'candidate_operation_rejected')
            if 'job_id' in body:
                if tool != 'verification_job' and 'state' not in body and body.get('job_state') == 'completed':
                    # The actual dispatcher flattens fast-completed results.
                    # Trust no inline result: recover the full retained envelope
                    # through this handle before consuming candidate/evidence data.
                    if (body.get('pending') is not False or body.get('result_is_retained') is not True
                            or ('tool' in body and body['tool'] != self.operation)):
                        raise SubmissionError('verification_job_invalid')
                    continue
                if body.get('tool') != self.operation:
                    raise SubmissionError('verification_job_invalid')
                state = body.get('state')
                if 'job_state' in body and body['job_state'] != state:
                    raise SubmissionError('verification_job_invalid')
                if state in ('queued', 'running') and body.get('pending') is True:
                    if tool == 'verification_job':
                        await sleep(1)
                    continue
                if state != 'completed':
                    raise SubmissionError('verification_job_not_completed')
                if body.get('pending') is not False or body.get('result_is_retained') is not True or type(body.get('result')) is not dict:
                    raise SubmissionError('verification_job_invalid')
                if tool != 'verification_job':
                    continue
                body = body['result']
                _no_completion(body)
                if body.get('error') is not None:
                    raise SubmissionError('candidate_operation_rejected')
            elif self.job_id:
                raise SubmissionError('verification_job_invalid')
            if self.candidate_id is None:
                candidate_id = body.get('candidate_id')
                if not _digest(candidate_id):
                    raise SubmissionError('candidate_registration_invalid')
                self.candidate_id, self.job_id, self.initial_dispatched = candidate_id, None, False
                continue
            return public_observation(body, self.candidate_id)


async def guard_request(request, endpoint):
    if str(request.url) != endpoint or request.method not in ('GET', 'POST'):
        raise SubmissionError('sidecar_origin_changed')


@asynccontextmanager
async def mcp_session(endpoint, token):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async def guard(request):
        await guard_request(request, endpoint)
    async with httpx.AsyncClient(headers={'Authorization': 'Bearer ' + token}, timeout=None,
                                 trust_env=False, follow_redirects=False,
                                 event_hooks={'request': [guard]}) as client:
        async with streamable_http_client(endpoint, http_client=client, terminate_on_close=False) as streams:
            async with ClientSession(streams[0], streams[1], read_timeout_seconds=None) as session:
                await session.initialize()
                yield session


async def perform(state: Submission, session_factory, *, sleep=asyncio.sleep):
    import httpx
    while True:
        try:
            async with session_factory() as session:
                return await state.run(session, sleep=sleep)
        except SubmissionError:
            raise
        except Exception as error:
            # AnyIO may wrap our origin/protocol guard in a task-group error.
            pending = [(error, 0)]
            for _ in range(32):
                if not pending:
                    break
                item, depth = pending.pop()
                if type(item) is SubmissionError:
                    raise item from None
                if isinstance(item, httpx.HTTPStatusError) and item.response.status_code in (401, 403):
                    raise SubmissionError('sidecar_authorization_rejected') from None
                if isinstance(item, BaseExceptionGroup) and depth < 8:
                    pending.extend((child, depth + 1) for child in item.exceptions[:32])
            if state.job_id is None:
                raise SubmissionError('submission_response_unknown' if state.initial_dispatched
                                      else 'submission_transport_unavailable') from None
            # Resume only this retained handle. Never retry create/evaluate.
            await sleep(1)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise SubmissionError('invalid_arguments')


def main(argv=None):
    state = None
    try:
        parser = _Parser(prog='submit.sh', description='Submit a candidate file for a vulnerable-target observation, not completion or a score.')
        parser.add_argument('poc_file_path', nargs='?')
        args = parser.parse_args(argv)
        if args.poc_file_path is None:
            parser.print_help()
            return 0
        path = candidate_path(args.poc_file_path)
        endpoint, token = endpoint_from_environment(os.environ)
        state = Submission(path)
        logging.disable(logging.CRITICAL)  # Isolated CLI: suppress third-party exception/body logging.
        result = asyncio.run(perform(state, lambda: mcp_session(endpoint, token)))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0  # Transport completion is not target success or task completion.
    except (SubmissionError, KeyboardInterrupt) as error:
        code = str(error) if type(error) is SubmissionError else 'submission_interrupted'
    except Exception:
        code = 'submission_transport_unavailable'
    print(json.dumps({'authority_scope': 'candidate_signal', 'verified': False,
                      'transport_status': 'unknown', 'error_code': code,
                      'pending': bool(state and state.job_id), 'job_id': state.job_id if state else None,
                      'candidate_id': state.candidate_id if state else None}))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
