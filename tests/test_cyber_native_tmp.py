"""Fixed temporary paths, private CLI captures, and retained transport bytes."""
import json
import os
from pathlib import Path
import runpy
import tomllib

import pytest

from verifier.llm.schema import RoleError
from verifier.profiles.cyber.backend import CyberCodexBackend
from verifier.profiles.cyber.native_tmp import PrivateTmp


def test_retention_omits_aliases_and_special_files_before_snapshot(tmp_path):
    from verifier.execution.discharge import _snapshot
    private, work = tmp_path / "private", tmp_path / "work"
    private.mkdir(mode=0o700)
    work.mkdir(mode=0o700)
    foreign = tmp_path / "foreign"
    foreign.write_text("EXTERNAL_SYNTHETIC_CONTENT")
    context = PrivateTmp(private, retain_parent=work)
    root = Path(context.path)
    (root / "nested").mkdir()
    (root / "nested" / "body.bin").write_bytes(b"\x00\xffexact\n")
    (root / "alias").symlink_to(foreign)
    (root / "directory-alias").symlink_to(tmp_path, target_is_directory=True)
    os.link(foreign, root / "shared")
    os.mkfifo(root / "pipe")
    capture = Path(context.capture_path)
    (capture / "final.txt").write_text("PRIVATE_CLI_FINAL")
    context.close()
    retained = work / root.name
    assert (retained / "nested/body.bin").read_bytes() == b"\x00\xffexact\n"
    assert not capture.exists() and not root.exists()
    manifest = json.loads((work / (root.name + ".binding.json")).read_text())
    assert manifest["basis"] == "transport_only"
    assert manifest["complete_capture_implied"] is False
    assert set(manifest["omitted_symlink_special_or_shared_entries"]) == {
        "alias", "directory-alias", "shared", "pipe"}
    snapshot = _snapshot(str(work))
    assert not any("EXTERNAL_SYNTHETIC_CONTENT" in value or "PRIVATE_CLI_FINAL" in value
                   for value in snapshot.values())
    assert foreign.read_text() == "EXTERNAL_SYNTHETIC_CONTENT"


def test_tmp_grant_requires_a_host_bound_call(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    backend = CyberCodexBackend(cwd=str(state), native_private_tmp=True)
    for role in ("discharge", "judger", "oracles"):
        args = backend._execution_policy_args(role, {"world_mode": "live"})
        grants = next(tomllib.loads("grants=" + arg.split("=", 1)[1])["grants"]
                      for arg in args if ".filesystem=" in arg)
        assert grants.get("/tmp") != "write"
    with pytest.raises(RoleError, match="no host binding"):
        backend._execution_command("discharge", {}, ["should-never-execute"])


def test_non_linux_binding_rejected_before_role_execution(tmp_path, monkeypatch):
    import verifier.profiles.cyber.backend as module
    monkeypatch.setattr(module.sys, "platform", "darwin")
    backend = CyberCodexBackend(cwd=str(tmp_path), native_private_tmp=True)
    with pytest.raises(RoleError, match="requires Linux"):
        backend.complete("discharge", "synthetic", {})


@pytest.mark.parametrize("alias", [False, True])
def test_declared_workdir_cannot_redirect_temporary_retention(tmp_path, monkeypatch, alias):
    import verifier.profiles.cyber.backend as module
    monkeypatch.setattr(module.sys, "platform", "linux")
    state, foreign = tmp_path / "state", tmp_path / "foreign"
    (state / "work").mkdir(parents=True)
    foreign.mkdir()
    destination = foreign
    if alias:
        destination = state / "work" / "alias"
        destination.symlink_to(foreign, target_is_directory=True)
    backend = CyberCodexBackend(cwd=str(state), native_private_tmp=True)
    with pytest.raises(RoleError):
        backend.complete("discharge", "synthetic", {"workdir": str(destination)})
    assert not list(foreign.iterdir())


NATIVE_TAIL = r'''
import concurrent.futures,hashlib,struct,zlib
from types import SimpleNamespace
from verifier.llm.codex_backend import _CapturedImage
from verifier.profiles.cyber.native_tmp import PrivateTmp
from experiments.cybergym.sidecar import SandboxedVerifierBackend
outside=pathlib.Path('/tmp/GV_OUTSIDE_TMP')
outside.write_text('synthetic outside sentinel')
provider='{name="synthetic",base_url="http://127.0.0.1:'+str(s.server_port)+'/v1",wire_api="responses",requires_openai_auth=false,supports_websockets=false,request_max_retries=0,stream_max_retries=0}'
def chunk(t,b):return struct.pack('!I',len(b))+t+b+struct.pack('!I',zlib.crc32(t+b)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',1,1,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b'\x00\x10\x20\x30'))+chunk(b'IEND',b'')
captures=[]
class ProbeBackend(SandboxedVerifierBackend):
 def _execution_policy_args(self,role,variables):
  return [*super()._execution_policy_args(role,variables),'-c','model_provider="synthetic"','-c','model_providers.synthetic='+provider]
 def _execution_capture_dir(self,role,variables):
  path=super()._execution_capture_dir(role,variables);captures.append(path);return path
 def _captured_images(self,role,variables):
  return (_CapturedImage('synthetic.png','image/png',png,hashlib.sha256(png).hexdigest()),) if role=='judger' else ()
b=ProbeBackend(cwd='/probe/state',project_root='/task')
for role in ('discharge','judger'):b.roles[role]=SimpleNamespace(system='',schema={'probe':str})
def call(role):
 work='/probe/state/work/'+role
 pathlib.Path(work).mkdir(parents=True)
 return b.complete(role,'Execute the synthetic probe.',{'world_mode':'live','workdir':work})
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(call,('discharge','judger')))
retained=list(pathlib.Path('/probe/state/work/discharge').glob('gv-private-tmp-*'))
directories=[p for p in retained if p.is_dir()]
assert len(directories)==1
assert (directories[0]/'GV_FIXED_TMP').read_bytes()==b'EXACT_TEMP_BYTES'
assert not list(pathlib.Path('/probe/state/work/judger').glob('gv-private-tmp-*'))
assert captures and all('/verifier-home/' in p and not pathlib.Path(p).exists() for p in captures)
assert not pathlib.Path('/tmp/GV_FIXED_TMP').exists() and outside.exists()
# Identity replacement must be rejected before even an ordinary command runs.
context=PrivateTmp('/probe/state/verifier-home')
original=pathlib.Path(context.path)
saved=original.with_name(original.name+'-saved');original.rename(saved);original.mkdir(mode=0o700)
negative=subprocess.run(context.command(['/usr/bin/python3','-c',"print('UNTRUSTED_LAUNCH')"]),capture_output=True,text=True,timeout=10)
assert negative.returncode==125 and 'UNTRUSTED_LAUNCH' not in negative.stdout
context.close()
import shutil
shutil.rmtree(saved)
s.shutdown()
print(json.dumps({'results':results,'request_count':len(requests),'image_inputs':sum(r['image_inputs'] for r in requests),'function_outputs':[x for r in requests for x in r['outputs']],'private_captures_removed':True,'fixed_tmp_retained':True,'source_replacement_rejected':True,'real_model_calls':0,'task_data_used':False}))
'''


@pytest.mark.skipif(os.environ.get("GV_CYBERGYM_HELPER_NATIVE") != "1",
                   reason="set GV_CYBERGYM_HELPER_NATIVE=1 for pinned native CLI regression")
def test_native_fixed_tmp_isolation_and_concurrent_role_captures(tmp_path):
    from experiments.cybergym import run
    source = runpy.run_path(str(Path(__file__).with_name("test_cybergym_helper_native.py")))["NATIVE_SCRIPT"]
    prefix = source.split("\nb=CyberCodexBackend(", 1)[0]
    prefix = prefix.replace("requests.append({'outputs':outputs})", "requests.append({'outputs':outputs,'image_inputs':sum(c.get('type')=='input_image' for item in inputs for c in item.get('content',[]) if isinstance(c,dict))})")
    prefix = prefix.replace("print('PROBE_EXECUTED');", "print('PROBE_EXECUTED'); "
        "pathlib.Path('/tmp/GV_FIXED_TMP').write_bytes(b'EXACT_TEMP_BYTES'); "
        "print('OUTSIDE_ISOLATED' if not pathlib.Path('/tmp/GV_OUTSIDE_TMP').exists() else 'TMP_ISOLATION_BUG');")
    prefix = prefix.replace("except (OSError, PermissionError): print('PRIVATE_ISOLATED')", "except (OSError, PermissionError): print('PRIVATE_ISOLATED')\\ntry: list(pathlib.Path('/probe/state/verifier-home').iterdir()); print('CAPTURE_ACCESS_BUG')\\nexcept (OSError, PermissionError): print('CAPTURE_ISOLATED')")
    script = tmp_path / "probe.py"
    script.write_text(prefix + NATIVE_TAIL)
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    result = run.execute("docker", "run", "--rm", "--pull=never", "--network", "none",
        "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--security-opt", "seccomp=unconfined",
        "--tmpfs", "/probe:rw,exec,uid=1000,gid=1000",
        "--tmpfs", "/task:rw,exec,uid=1000,gid=1000",
        "--mount", "type=bind,source=" + str(home) + ",target=/codex-home",
        "--tmpfs", "/codex-home/tmp:rw,exec,uid=1000,gid=1000,mode=700,size=64m",
        "--mount", "type=bind,source=" + str(Path(__file__).resolve().parents[1]) + ",target=/gv,readonly",
        "--mount", "type=bind,source=" + str(script) + ",target=/probe.py,readonly",
        "--env", "PYTHONPATH=/gv", "--entrypoint", "python3", run.IMAGE, "/probe.py",
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout)
    assert report["results"] == [{"probe": "done"}, {"probe": "done"}]
    assert report["request_count"] == 4
    assert report["image_inputs"] >= 1
    assert report["real_model_calls"] == 0 and report["task_data_used"] is False
    output = "\n".join(row["output"] for row in report["function_outputs"])
    assert output.count("PROBE_EXECUTED") == output.count("OUTSIDE_ISOLATED") == 2
    assert output.count("PRIVATE_ISOLATED") == 2
    assert output.count("CAPTURE_ISOLATED") == 2
    assert "TMP_ISOLATION_BUG" not in output and "PRIVATE_ACCESS_BUG" not in output
    assert "CAPTURE_ACCESS_BUG" not in output
    assert "SYNTHETIC_PRIVATE_MARKER" not in result.stdout + result.stderr
