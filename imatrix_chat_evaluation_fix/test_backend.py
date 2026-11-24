"""Offline checks: mock llama.cpp API; no models or network required."""
import sys, types, importlib.util
from pathlib import Path
lm = types.ModuleType('lm_eval')
a = types.ModuleType('lm_eval.api')
m = types.ModuleType('lm_eval.api.model')
m.LM = type('LM', (), {})
sys.modules.update({'lm_eval':lm,'lm_eval.api':a,'lm_eval.api.model':m})
spec=importlib.util.spec_from_file_location('backend',Path(__file__).parent/'imatrix/backend.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
class Reply:
    def __init__(self, output):self.output=output
    def raise_for_status(self):pass
    def json(self):return self.output
class Session:
    def __init__(self, output):self.output=output;self.sent=[]
    def post(self,url,json,timeout):
        self.sent.append((url,json,timeout));return Reply(self.output)
class Req:
    args=('Question: 2 + 2?\nAnswer:',{'until':['\n'],'max_gen_toks':100})
chat=b.GGUFGenerationLM('http://127.0.0.1:8080',max_new_tokens=64,prompt_format='chat')
cs=Session({'choices':[{'message':{'content':'4'}}],'usage':{'completion_tokens':1}})
chat._local.session=cs
assert chat._one(Req())=='4'
url,payload,_=cs.sent[0]
assert url.endswith('/v1/chat/completions')
assert payload['messages']==[{'role':'user','content':Req.args[0]}]
assert payload['max_tokens']==64 and payload['stop']==['\n']
assert chat.requests_completed==1
raw=b.GGUFGenerationLM('http://127.0.0.1:8080',prompt_format='raw')
rs=Session({'choices':[{'text':'4'}]})
raw._local.session=rs
assert raw._one(Req())=='4'
assert rs.sent[0][0].endswith('/v1/completions') and rs.sent[0][1]['prompt']==Req.args[0]
assert raw.loglikelihood([])==[]
try:raw.loglikelihood([Req()]);raise AssertionError('should block loglikelihood')
except NotImplementedError:pass
print('Offline backend checks passed: chat template API, raw mode, stops, caps, scoring safety.')
