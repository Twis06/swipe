const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const code=fs.readFileSync('research/worklets/adaptive-preview-worklet.js','utf8');
function create(rate){let Type;const messages=[];vm.runInNewContext(code,{AudioWorkletProcessor:class{constructor(){this.port={postMessage:m=>messages.push(m)}}},registerProcessor:(_,t)=>Type=t,Float32Array,Math,sampleRate:rate});return {node:new Type(),messages,rate,signal:[]};}
function feed(p,x){p.signal.push(...x);for(let i=0;i<x.length;i+=128)p.node.process([[x.slice(i,i+128)]]);}
for(const rate of [44100,48000]){
  const p=create(rate);feed(p,new Float32Array(rate*2));
  const tone=Float32Array.from({length:Math.round(rate*.1)},(_,i)=>8*Math.sin(i*.2));feed(p,tone);feed(p,new Float32Array(rate*2));
  const c=p.messages.filter(m=>m.type==='candidate');assert.equal(c.length,1);assert.equal(c[0].pcm.length,rate*2);
  const delay=(c[0].emitFrame-c[0].onsetFrame)/rate;assert.ok(delay>=.2&&delay<.23);assert.ok(c[0].pcm.some(v=>v>7));
  assert.deepEqual(Array.from(c[0].pcm),p.signal.slice(c[0].startFrame,c[0].emitFrame),'Candidate must be exact real sample history');
  const q=create(rate);feed(q,new Float32Array(rate*2));feed(q,new Float32Array(Math.round(rate*1.2)).fill(.1));
  const long=q.messages.find(m=>m.type==='candidate');assert.ok(long);assert.equal(long.endpointReason,'maximum_wait');assert.ok((long.emitFrame-long.onsetFrame)/rate<1.004);
  const cancelled=create(rate);feed(cancelled,new Float32Array(rate*2));feed(cancelled,tone);cancelled.node.port.onmessage({data:{type:'stop'}});feed(cancelled,new Float32Array(rate));assert.equal(cancelled.messages.filter(m=>m.type==='candidate').length,0);
  const early=create(rate);feed(early,new Float32Array(Math.round(rate*1.1)));feed(early,tone);feed(early,new Float32Array(rate));assert.ok(early.messages.find(m=>m.type==='candidate').startFrame>=0,'No unrecorded zero prefix');
}
console.log('PASS: adaptive endpoints, max wait, real samples/headroom, both rates, cancellation, complete history');
