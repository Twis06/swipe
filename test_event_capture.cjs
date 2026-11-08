const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const code=fs.readFileSync('event-preview-worklet.js','utf8');
function create(rate){let Type;const messages=[];vm.runInNewContext(code,{AudioWorkletProcessor:class{constructor(){this.port={postMessage:m=>messages.push(m)}}},registerProcessor:(_,t)=>Type=t,Float32Array,Math,sampleRate:rate});return {node:new Type(),messages,rate,signal:[]};}
function feed(p,x){p.signal.push(...x);for(let i=0;i<x.length;i+=128)p.node.process([[x.slice(i,i+128)]]);}
for(const rate of [44100,48000]){
  const p=create(rate);feed(p,new Float32Array(Math.round(rate*1.2)));
  const tone=Float32Array.from({length:Math.round(rate*.1)},(_,i)=>8*Math.sin(i*.2));feed(p,tone);feed(p,new Float32Array(rate*2));
  const c=p.messages.filter(m=>m.type==='candidate');assert.equal(c.length,1);assert.equal(c[0].pcm.length,rate*2);
  const w=c[0],delay=(w.emitFrame-w.onsetFrame)/rate;assert.ok(delay>=.2&&delay<.23);assert.ok(w.pcm.some(v=>v>7));
  assert.deepEqual(Array.from(w.pcm.slice(w.destinationOffset,w.destinationOffset+w.sourceSamples)),p.signal.slice(w.sourceStartFrame,w.sourceEndFrame));
  assert.equal(w.startFrame+w.destinationOffset,w.sourceStartFrame,'Mapped time must use actual source samples');
  assert.ok(w.pcm.slice(0,w.destinationOffset).every(v=>v===0));assert.ok(w.pcm.slice(w.destinationOffset+w.sourceSamples).every(v=>v===0));
  const q=create(rate);feed(q,new Float32Array(Math.round(rate*1.2)));feed(q,new Float32Array(Math.round(rate*1.2)).fill(.1));
  const long=q.messages.find(m=>m.type==='candidate');assert.ok(long);assert.equal(long.endpointReason,'maximum_wait');assert.ok((long.emitFrame-long.onsetFrame)/rate<1.004);
  const cancelled=create(rate);feed(cancelled,new Float32Array(Math.round(rate*1.2)));feed(cancelled,tone);cancelled.node.port.onmessage({data:{type:'stop'}});feed(cancelled,new Float32Array(rate));assert.equal(cancelled.messages.filter(m=>m.type==='candidate').length,0);
}
console.log('PASS: isolated endpoint, actual samples, synthetic zeros, time mapping, headroom, rates, timeout, cancellation');
