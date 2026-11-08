const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const worklet=fs.readFileSync('research/worklets/preview-worklet.js','utf8');
function processor(rate=48000){let Type;const messages=[];vm.runInNewContext(worklet,{AudioWorkletProcessor:class{constructor(){this.port={postMessage:m=>messages.push(m)}}},registerProcessor:(_,type)=>Type=type,Float32Array,Math,sampleRate:rate});return {node:new Type(),messages};}
function feed(p,samples){for(let i=0;i<samples.length;i+=128)p.node.process([[samples.slice(i,i+128)]]);}
const p=processor();feed(p,new Float32Array(60000));assert.equal(p.messages.filter(m=>m.type==='candidate').length,0);
const burst=Float32Array.from({length:4800},(_,i)=>8*Math.sin(i*.2)*Math.sin(Math.PI*i/4800));feed(p,burst);feed(p,new Float32Array(144000));
const candidates=p.messages.filter(m=>m.type==='candidate');assert.equal(candidates.length,1,'One isolated swipe must not generate duplicates');assert.equal(candidates[0].pcm.length,96000);assert.ok(candidates[0].pcm.some(x=>x>7),'Preview must preserve float headroom');
const cancelled=processor();feed(cancelled,new Float32Array(60000));feed(cancelled,burst);cancelled.node.port.onmessage({data:{type:'stop'}});feed(cancelled,new Float32Array(96000));assert.equal(cancelled.messages.filter(m=>m.type==='candidate').length,0,'Stop must discard pending audio');
for(const rate of [44100,48000]){const q=processor(rate);feed(q,new Float32Array(rate*2));feed(q,burst);feed(q,new Float32Array(rate*2));assert.equal(q.messages.find(m=>m.type==='candidate').pcm.length,rate*2);}

// Exercise the real start/stop lifecycle with a permission request resolving
// after Stop, and a pending inference returning after Stop. No browser/mic.
const elements=new Map();const element=id=>{if(!elements.has(id))elements.set(id,{value:'',disabled:false,textContent:'',dataset:{},options:[],replaceChildren(){},append(){},add(){}});return elements.get(id);};
let resolveMedia,stopped=0,contexts=0;
const sandbox={console,Math,Date,Number,ArrayBuffer,DataView,Float32Array,Blob,AbortController,setTimeout,clearTimeout,
  document:{getElementById:element,createElement:()=>({append(){}}),addEventListener(){}},
  window:{AudioWorkletNode:true,addEventListener(){}},Option:class{},
  navigator:{mediaDevices:{enumerateDevices:async()=>[],getUserMedia:()=>new Promise(r=>resolveMedia=r)}},
  fetch:async()=>({ok:true,json:async()=>({ready:true,token:'test',training_clips:56,training_sessions:2})}),
  AudioContext:class{constructor(){contexts++}},AudioWorkletNode:class{}};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync('web/preview.js','utf8'),sandbox);
(async()=>{
  await new Promise(r=>setImmediate(r));
  const starting=vm.runInContext('start()',sandbox);await vm.runInContext('stop()',sandbox);
  resolveMedia({getTracks:()=>[{stop(){stopped++}}]});await starting;
  assert.equal(stopped,1);assert.equal(contexts,0,'A cancelled permission result must not open an audio context');
  let resolveInference;sandbox.fetch=()=>new Promise(r=>resolveInference=r);
  vm.runInContext('active=true;generation=10',sandbox);
  const pending=vm.runInContext('predictCandidate(new Float32Array(96000),48000,10)',sandbox);
  await vm.runInContext('stop()',sandbox);
  resolveInference({ok:true,json:async()=>({prediction:'up'})});await pending;
  assert.equal(element('result').textContent,'Preview stopped','An old response must not reactivate the preview');
  assert.equal(vm.runInContext('history.length',sandbox),0);
  sandbox.first={quality:{event_start_seconds:.9,event_end_seconds:1.1}};
  sandbox.overlap={quality:{event_start_seconds:.4,event_end_seconds:.6}};
  assert.equal(vm.runInContext('overlapsPreviousEvent(first,48000,48000)',sandbox),false);
  assert.equal(vm.runInContext('overlapsPreviousEvent(overlap,72000,48000)',sandbox),true,'Overlapping windows must not produce duplicate results');
  assert.equal(vm.runInContext('overlapsPreviousEvent(first,144000,48000)',sandbox),false,'A distinct later swipe must still be accepted');
  console.log('PASS: capture timing, duplicate suppression, stop cancellation, raw headroom, both rates, permission/inference races');
})().catch(error=>{console.error(error);process.exitCode=1});
