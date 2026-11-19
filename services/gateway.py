"""Bounded ASR/text router for the benchmark's OpenAI-compatible bridge."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import os,json,hmac,urllib.request,urllib.error,threading,time,argparse
STATE={k:dict(active=0,queued=0,completed=0,errors=0) for k in ('asr','text')};LOCK=threading.Lock()
SEMS={k:threading.BoundedSemaphore(int(os.environ.get(k.upper()+'_CONCURRENCY',str(v)))) for k,v in [('asr',2),('text',24)]}
KEY=os.environ['MUBENCH_GATEWAY_KEY'];TEXT_KEY=os.environ['TEXT_API_KEY'];TEXT_URL=os.environ['TEXT_BASE_URL'].rstrip('/');ASR_URL=os.environ.get('ASR_BASE_URL','http://127.0.0.1:18991').rstrip('/')
def url(base,path):return base+path[3:] if base.endswith('/v1') and path.startswith('/v1/') else base+path
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def reply(self,status,body):
  self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def auth(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+KEY)
 def do_GET(self):
  if not self.auth():self.reply(401,b'{}');return
  with LOCK:body=json.dumps(STATE if self.path=='/metrics' else {'ok':True}).encode()
  self.reply(200,body)
 def do_POST(self):
  if not self.auth():self.reply(401,b'{}');return
  raw=self.rfile.read(int(self.headers.get('Content-Length','0')))
  try:
   data=json.loads(raw);audio=any(isinstance(m.get('content'),list) and any(isinstance(p,dict) and ('input_audio' in p or p.get('type')=='input_audio') for p in m['content']) for m in data.get('messages',[]))
   kind='asr' if audio or str(data.get('model','')).startswith('gpt-audio') else 'text';sem=SEMS[kind];status=503;acquired=False
   with LOCK:STATE[kind]['queued']+=1
   try:
    acquired=sem.acquire(timeout=90)
    with LOCK:
     STATE[kind]['queued']-=1
     if acquired:STATE[kind]['active']+=1
    if not acquired:self.reply(503,b'{"error":"queue_timeout"}');return
    upstream=url(ASR_URL if kind=='asr' else TEXT_URL,self.path)
    headers={'Content-Type':'application/json','Authorization':'Bearer '+(KEY if kind=='asr' else TEXT_KEY)}
    req=urllib.request.Request(upstream,data=raw,headers=headers,method='POST')
    # Standard proxy variables are respected for text; loopback should be in NO_PROXY.
    try:
     with urllib.request.urlopen(req,timeout=720) as r:status=r.status;body=r.read()
    except urllib.error.HTTPError as e:status=e.code;body=e.read()
    body=body.replace(TEXT_KEY.encode(),b'[redacted]');self.reply(status,body)
   finally:
    with LOCK:
     if acquired:STATE[kind]['active']-=1
     STATE[kind]['completed']+=1;STATE[kind]['errors']+=int(status!=200)
    if acquired:sem.release()
  except (BrokenPipeError,ConnectionResetError):pass
  except Exception as e:self.reply(502,json.dumps({'error':type(e).__name__}).encode())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=18992);a=p.parse_args();ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()
