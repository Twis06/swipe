const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map(),calls=[],listeners={};
const el=id=>{if(!elements.has(id)) elements.set(id,{dataset:{},setAttribute(){}});return elements.get(id);};
let handler=async(url,options)=>({ok:true,json:async()=>url==='/api/keys'&&!options?.method?{supported:true,permitted:true,supported_codes:['ArrowUp','ArrowDown','KeyK']}:{owned:true}});
const context={window:{},document:{getElementById:el,addEventListener:(name,fn)=>listeners[name]=fn},crypto:{randomUUID:()=> 'test-client'},localStorage:{getItem:()=>null,setItem(){}},AbortController,setTimeout,clearTimeout,setInterval:()=>1,clearInterval(){},fetch:async(...args)=>{calls.push(args);return handler(...args);}};
vm.createContext(context);vm.runInContext(fs.readFileSync('web/key-bindings.js','utf8'),context);
const keys=context.window.swipeKeys;
(async()=>{
 await keys.configure('token'); assert.equal(keys.isArmed(),false);assert.equal(el('enableKeys').disabled,true);
 keys.setListening(true);assert.equal(el('enableKeys').disabled,false);
 await keys.accepted({action_ticket:'ignored'},()=>true);assert.equal(calls.length,1,'No key output without opt-in');
 let resolveArm;handler=(url,o)=>JSON.parse(o.body).action==='arm'?new Promise(r=>resolveArm=r):Promise.resolve({ok:true,json:async()=>({owned:false})});
 const pending=el('enableKeys').onclick();keys.setListening(false);await keys.stop();
 resolveArm({ok:true,json:async()=>({owned:true})});await pending;assert.equal(keys.isArmed(),false,'Late arm must not revive a stopped preview');
 keys.setListening(true);handler=async()=>({ok:true,json:async()=>({owned:true})});await el('enableKeys').onclick();assert.equal(keys.isArmed(),true);
 const count=calls.length;await keys.accepted({},()=>true);await keys.accepted({action_ticket:'stale'},()=>false);assert.equal(calls.length,count,'Missing and stale tickets never fire');
 handler=async()=>{throw Error('Network disconnected')};await keys.accepted({action_ticket:'valid'},()=>true);assert.equal(keys.isArmed(),false,'Network failure disarms');
 console.log('PASS: explicit key opt-in, late arm cancellation, stale result suppression, disconnect disarm');
})().catch(e=>{console.error(e);process.exitCode=1});
