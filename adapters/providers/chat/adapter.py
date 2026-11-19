"""Streaming chat transport; only delivered prefixes enter durable history."""
import json,time,urllib.request,urllib.error

class ProviderError(RuntimeError):
    def __init__(self,status,detail):
        self.status=status;super().__init__('status='+str(status)+' '+detail)

def token_cost(char):
    # Explicit estimate, not Qwen tokenizer tokens and never SSE chunk count.
    return 1.0 if ord(char)>0x2fff else .25

class API:
    def __init__(self,config,audit):
        self.config=config; self.audit=audit
        proxy=config.get('https_proxy')
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy,'http':proxy} if proxy else {}))

    def open(self,payload):
        request=urllib.request.Request(self.config['base_url'].rstrip('/')+('/chat/completions' if self.config['base_url'].rstrip('/').endswith('/v1') else '/v1/chat/completions'),data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Authorization':'Bearer '+self.config['api_key'],'Content-Type':'application/json'})
        try:return self.opener.open(request,timeout=self.config.get('timeout_seconds',150))
        except urllib.error.HTTPError as e:
            detail=e.read().decode(errors='replace').replace(self.config['api_key'],'[redacted]')
            raise ProviderError(e.code,detail[:2000]) from None

    def post_json(self,url,key,path,payload):
        if payload.get("model")=="gpt-6-luna":payload=dict(payload,model=self.config.get("judge_model","gpt-6-luna"))
        # Caller uses same private provider; secrets are never copied to audit.
        t=time.time()
        for attempt in range(3):
            try:
                with self.open(payload) as r:obj=json.load(r)
                if obj.get('model')!=payload['model']:
                    raise ValueError('Unexpected returned model: '+str(obj.get('model')))
                self.audit.append({'model':obj['model'],'usage':obj.get('usage'),'seconds':time.time()-t,'stream':False})
                return obj
            except ProviderError as e:
                if e.status not in (429,500,502,503,504) or attempt==2:raise
                time.sleep(2**attempt)
            except (TimeoutError,ConnectionError):
                if attempt==2:raise
                time.sleep(2**attempt)

    def stream(self,messages,tools):
        payload={'model':self.config.get('sut_model','qwen3.8-flash'),'messages':messages,'tools':tools,'tool_choice':'auto','parallel_tool_calls':False,'stream':True,'stream_options':{'include_usage':True},'max_tokens':8192}
        # No automatic stream replay: a prefix or completed tool can be committed.
        with self.open(payload) as response:
            for line in response:
                if not line.startswith(b'data:'):continue
                data=line[5:].strip()
                if data==b'[DONE]':break
                obj=json.loads(data)
                if obj.get('model') and obj['model']!=self.config.get('sut_model','qwen3.8-flash'):raise ValueError('Wrong SUT returned model')
                if obj.get('error'):raise ValueError('Streaming provider error')
                if obj.get('usage'):self.audit.append({'model':obj.get('model'),'usage':obj['usage'],'stream':True})
                for choice in obj.get('choices',[]):
                    yield choice.get('delta',{}),choice.get('finish_reason')

class JSONBackend:
    def __init__(self,api):self.api=api
    def generate_json(self,messages):
        # Same transport-only hint as the production BridgeJSONProposalBackend.
        if not any('json' in str(row.get('content','')) for row in messages):
            messages=[{'role':'system','content':'Return one valid json object.'},*messages]
        obj=self.api.post_json(None,None,None,{'model':self.api.config.get('simulator_model','gpt-6-luna'),'temperature':.2,'response_format':{'type':'json_object'},'messages':messages})
        content=obj['choices'][0]['message']['content'].strip()
        if content.startswith('```'):
            content='\n'.join(content.splitlines()[1:-1]).strip()
            if content.startswith('json'):content=content[4:].strip()
        return json.loads(content)

class TextStream:
    def __init__(self,api,history,tools):self.api=api;self.history=history;self.tools=tools
    def response(self,on_char,on_tools):
        prefix='';calls={};finish=None;interrupted=False;tail_seen=''
        stream=self.api.stream(self.history,self.tools)
        try:
            for delta,reason in stream:
                finish=reason or finish
                # reasoning_content is deliberately neither spoken nor committed.
                content=delta.get('content') or ''
                for index,ch in enumerate(content):
                    prefix+=ch
                    if on_char(ch,prefix):
                        tail_seen=content[index+1:];interrupted=True;break
                if interrupted:break
                for chunk in delta.get('tool_calls',[]):
                    idx=chunk['index'];row=calls.setdefault(idx,{'id':'','type':'function','function':{'name':'','arguments':''}})
                    if chunk.get('id'):row['id']=chunk['id']
                    function=chunk.get('function',{})
                    row['function']['name']+=function.get('name','')
                    row['function']['arguments']+=function.get('arguments','')
        finally:stream.close()
        message={'role':'assistant','content':prefix or None}
        if not interrupted and calls:
            if finish!='tool_calls':raise ValueError('Incomplete tool-call stream: will not execute')
            complete=[calls[i] for i in sorted(calls)]
            for call in complete:
                if not call['id'] or not call['function']['name']:raise ValueError('Incomplete tool identity')
                json.loads(call['function']['arguments'])
            message['tool_calls']=complete
        if prefix or message.get('tool_calls'):self.history.append(message)
        # The caller inserts interrupting user input only AFTER the prefix.
        if not interrupted and calls:on_tools(message['tool_calls'])
        return {'prefix':prefix,'interrupted':interrupted,'discarded_buffer_suffix':tail_seen,'incomplete_calls_discarded':list(calls.values()) if interrupted else [],'finish_reason':finish}
