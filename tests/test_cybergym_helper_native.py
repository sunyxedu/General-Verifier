"""Opt-in real pinned CLI concurrency regression; local fixture, no model calls."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.cybergym import run


NATIVE_SCRIPT = r"""
import os,json,gzip,threading,subprocess,pathlib,shlex
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from verifier.profiles.cyber.backend import CyberCodexBackend
requests=[]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_POST(self):
  raw=self.rfile.read(int(self.headers['Content-Length']))
  if self.headers.get('Content-Encoding')=='gzip':raw=gzip.decompress(raw)
  r=json.loads(raw); inputs=r.get('input',[])
  outputs=[x for x in inputs if x.get('type')=='function_call_output']
  requests.append({'outputs':outputs})
  if not outputs:
   peers=[subprocess.Popen(['codex','--version'],env={**os.environ,'CODEX_HOME':'/codex-home','HOME':'/codex-home'},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) for _ in range(4)]
   for peer in peers:peer.wait()
   item={'id':'fc_probe','type':'function_call','call_id':'call_probe','name':'exec_command','namespace':'functions','arguments':json.dumps({'cmd':'/usr/bin/python3 -c '+shlex.quote("import pathlib; print('PROBE_EXECUTED');\ntry: pathlib.Path('/codex-home/auth.json').read_text(); print('PRIVATE_ACCESS_BUG')\nexcept (OSError, PermissionError): print('PRIVATE_ISOLATED')"),'max_output_tokens':100,'yield_time_ms':1000}),'status':'completed'}
   events=[{'type':'response.output_item.added','output_index':0,'item':{**item,'arguments':'','status':'in_progress'}},{'type':'response.function_call_arguments.delta','item_id':item['id'],'output_index':0,'delta':item['arguments']},{'type':'response.function_call_arguments.done','item_id':item['id'],'output_index':0,'arguments':item['arguments']},{'type':'response.output_item.done','output_index':0,'item':item}]
  else:
   content={'type':'output_text','text':'{"probe":"done"}','annotations':[],'logprobs':[]}
   item={'id':'msg_probe','type':'message','role':'assistant','phase':'final_answer','status':'completed','content':[content]}
   events=[{'type':'response.output_item.added','output_index':0,'item':{**item,'content':[],'status':'in_progress'}},{'type':'response.content_part.added','item_id':item['id'],'output_index':0,'content_index':0,'part':{**content,'text':''}},{'type':'response.output_text.delta','item_id':item['id'],'output_index':0,'content_index':0,'delta':content['text']},{'type':'response.output_item.done','output_index':0,'item':item}]
  response={'id':'resp_probe','object':'response','created_at':0,'status':'completed','model':'gpt-5.6-luna','output':[item],'parallel_tool_calls':False,'usage':{'input_tokens':0,'output_tokens':0,'total_tokens':0}}
  events=[{'type':'response.created','response':{**response,'status':'in_progress','output':[],'usage':None}},*events,{'type':'response.completed','response':response}]
  wire=''.join('event: '+e['type']+'\ndata: '+json.dumps({**e,'sequence_number':i})+'\n\n' for i,e in enumerate(events)).encode()
  self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
for d in ('/probe/state','/task','/codex-home'):pathlib.Path(d).mkdir(parents=True,exist_ok=True)
pathlib.Path('/codex-home/auth.json').write_text(json.dumps({'OPENAI_API_KEY':'SYNTHETIC_PRIVATE_MARKER'}))
pathlib.Path('/probe/state/task.md').write_text('Synthetic command execution probe; no benchmark data.')
s=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=s.serve_forever,daemon=True).start()
b=CyberCodexBackend(cwd='/probe/state',project_root='/task')
args=b._execution_policy_args('discharge',{'world_mode':'live'})
provider='{name="synthetic",base_url="http://127.0.0.1:'+str(s.server_port)+'/v1",wire_api="responses",requires_openai_auth=false,supports_websockets=false,request_max_retries=0,stream_max_retries=0}'
cmd=['codex','exec',*args,'--json','--skip-git-repo-check','-m','gpt-5.6-luna','-c','model_provider="synthetic"','-c','model_providers.synthetic='+provider,'--','Execute the synthetic probe.']
env={**os.environ,'CODEX_HOME':'/codex-home','HOME':'/codex-home'}
p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=30)
s.shutdown()
print(json.dumps({'returncode':p.returncode,'request_count':len(requests),'function_outputs':[x for r in requests for x in r['outputs']],'stdout':p.stdout[-6000:],'stderr':p.stderr[-2000:],'real_model_calls':0,'task_data_used':False},indent=2))

"""


@pytest.mark.skipif(os.environ.get("GV_CYBERGYM_HELPER_NATIVE") != "1",
                    reason="set GV_CYBERGYM_HELPER_NATIVE=1 for pinned Docker CLI regression")
def test_concurrent_cli_helpers_execute_and_keep_auth_private(tmp_path, monkeypatch):
    # Capture the actual production sidecar's tmpfs declaration, then exercise
    # it under a real bind-backed home with concurrent CLI janitor startups.
    controller = run.Evaluation(tmp_path)
    monkeypatch.setattr(controller, "create_isolated_network", lambda *_: "10.240.0.0/24")
    private = tmp_path / "private"
    private.mkdir()
    captured = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        if args[0] == "run" and "fixture-sidecar" in args:
            captured.extend(args)
            raise Captured
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(run, "docker", capture)
    runtime = dict(network="fixture-net", sidecar="fixture-sidecar", proxy="fixture-egress",
                   volume="fixture-state", evaluator_volume="fixture-evaluator", token="synthetic-token")
    with pytest.raises(Captured):
        controller._create_runtime(tmp_path, tmp_path / "task", private, runtime)
    mounts = [captured[i + 1] for i, arg in enumerate(captured[:-1])
              if arg == "--tmpfs" and captured[i + 1].startswith("/codex-home/tmp:")]
    assert len(mounts) == 1
    script = tmp_path / "probe.py"
    script.write_text(NATIVE_SCRIPT)
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    result = run.execute("docker", "run", "--rm", "--pull=never", "--network", "none",
        "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--security-opt", "seccomp=unconfined",
        "--tmpfs", "/probe:rw,exec,uid=1000,gid=1000",
        "--tmpfs", "/task:rw,exec,uid=1000,gid=1000",
        "--mount", "type=bind,source=" + str(home) + ",target=/codex-home",
        "--tmpfs", mounts[0],
        "--mount", "type=bind,source=" + str(run.PROFILE) + ",target=/gv,readonly",
        "--mount", "type=bind,source=" + str(script) + ",target=/probe.py,readonly",
        "--env", "PYTHONPATH=/gv", "--entrypoint", "python3", run.IMAGE, "/probe.py",
        capture_output=True, text=True, timeout=60)
    report = json.loads(result.stdout)
    assert report["returncode"] == 0 and report["request_count"] == 2
    assert report["real_model_calls"] == 0 and report["task_data_used"] is False
    output = "\n".join(row["output"] for row in report["function_outputs"])
    assert "PROBE_EXECUTED" in output and "PRIVATE_ISOLATED" in output
    assert "PRIVATE_ACCESS_BUG" not in output
    assert "SYNTHETIC_PRIVATE_MARKER" not in result.stdout + result.stderr
