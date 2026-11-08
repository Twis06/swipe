const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map(),acks=[];
const el=id=>{if(!elements.has(id))elements.set(id,{value:'',disabled:false,textContent:'',dataset:{},replaceChildren(){},append(){},add(){}});return elements.get(id);};
const sandbox={console,Math,Date,Number,ArrayBuffer,DataView,Float32Array,Blob,AbortController,setTimeout,clearTimeout,
 document:{getElementById:el,createElement:()=>({append(){}}),addEventListener(){}},window:{AudioWorkletNode:true,addEventListener(){}},Option:class{},
 navigator:{mediaDevices:{enumerateDevices:async()=>[]}},fetch:async()=>({ok:true,json:async()=>({ready:true,token:'test',training_clips:56,training_sessions:2})}),acks};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync('preview.js','utf8'),sandbox);
const tick=()=>new Promise(r=>setImmediate(r));
function response(prediction){return {ok:true,json:async()=>({prediction,reason:'accepted',quality:{event_start_seconds:.8,event_end_seconds:1.2}})};}
function start(){vm.runInContext('active=true;generation++;lastEventEnd=-Infinity;acceptedCaptureId=null;queuedFallback=null;history=[];recorder={port:{postMessage:m=>acks.push(m)},disconnect(){}}',sandbox);}
function candidate(id,phase){return vm.runInContext(`predictCandidate(new Float32Array(96000),48000,generation,48000,${id},'${phase}')`,sandbox);}
(async()=>{
 await tick();start();let resolves=[];sandbox.fetch=()=>new Promise(r=>resolves.push(r));
 const early=candidate(1,'early');await candidate(1,'fallback');assert.equal(resolves.length,1);
 resolves.shift()(response('uncertain'));await early;await tick();assert.equal(resolves.length,1,'Rejected early result must run retained fallback');
 assert.equal(vm.runInContext('history.length',sandbox),0,'Do not display an early rejection');
 resolves.shift()(response('up'));await tick();await tick();assert.equal(vm.runInContext('history.length',sandbox),1);
 await vm.runInContext('stop()',sandbox);

 start();resolves=[];const accepted=candidate(2,'early');await candidate(2,'fallback');resolves.shift()(response('down'));await accepted;await tick();
 assert.equal(resolves.length,0,'Accepted early result discards queued fallback');
 assert.equal(acks.at(-1).id,2);assert.equal(vm.runInContext('history.length',sandbox),1);
 await candidate(2,'fallback');assert.equal(resolves.length,0,'Late fallback for accepted ID is ignored');
 await vm.runInContext('stop()',sandbox);

 start();resolves=[];const cancelled=candidate(3,'early');await candidate(3,'fallback');await vm.runInContext('stop()',sandbox);
 const ackCount=acks.length;resolves.shift()(response('up'));await cancelled;await tick();
 assert.equal(resolves.length,0);assert.equal(acks.length,ackCount);assert.equal(vm.runInContext('history.length',sandbox),0,'Stop clears queued fallback and prevents stale display/ack');
 console.log('PASS: early rejection fallback, accepted ack, late fallback suppression, busy queue and Stop races');
})().catch(error=>{console.error(error);process.exitCode=1});
