const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
function create(rate){let Type;const messages=[];vm.runInNewContext(fs.readFileSync('cascade-preview-worklet.js','utf8'),{AudioWorkletProcessor:class{constructor(){this.port={postMessage:m=>messages.push(m)}}},registerProcessor:(_,t)=>Type=t,Float32Array,Math,sampleRate:rate});return {node:new Type(),messages,rate,signal:[]};}
function feed(p,x){p.signal.push(...x);for(let i=0;i<x.length;i+=128)p.node.process([[x.slice(i,i+128)]]);}
for(const rate of [44100,48000]){
  const tone=Float32Array.from({length:Math.round(rate*.1)},(_,i)=>8*Math.sin(i*.2));
  const make=()=>{const p=create(rate);feed(p,new Float32Array(Math.round(rate*1.2)));feed(p,tone);feed(p,new Float32Array(Math.round(rate*.15)));return p;};
  const p=make();const early=p.messages.find(m=>m.phase==='early');assert.ok(early);assert.ok((early.emitFrame-early.onsetFrame)/rate<.23);
  assert.equal(early.startFrame+early.destinationOffset,early.sourceStartFrame);
  assert.deepEqual(Array.from(early.pcm.slice(early.destinationOffset,early.destinationOffset+early.sourceSamples)),p.signal.slice(early.sourceStartFrame,early.sourceEndFrame));
  p.node.port.onmessage({data:{type:'accepted',id:early.id}});feed(p,new Float32Array(rate*2));assert.equal(p.messages.filter(m=>m.phase==='fallback').length,0);
  const q=make();const qe=q.messages.find(m=>m.phase==='early');q.node.port.onmessage({data:{type:'accepted',id:qe.id+1}});feed(q,new Float32Array(rate*2));
  const fallback=q.messages.find(m=>m.phase==='fallback');assert.ok(fallback);assert.equal(fallback.id,qe.id);assert.ok((fallback.emitFrame-fallback.onsetFrame)/rate>=1);
  assert.deepEqual(Array.from(fallback.pcm),q.signal.slice(fallback.sourceStartFrame,fallback.sourceEndFrame),'Fallback must retain exact actual2s history');
  const cancelled=make();cancelled.node.port.onmessage({data:{type:'stop'}});feed(cancelled,new Float32Array(rate*2));assert.equal(cancelled.messages.filter(m=>m.phase==='fallback').length,0);
  const long=create(rate);feed(long,new Float32Array(Math.round(rate*1.2)));feed(long,new Float32Array(Math.round(rate*1.2)).fill(.1));assert.equal(long.messages.filter(m=>m.phase==='early').length,0);assert.equal(long.messages.filter(m=>m.phase==='fallback').length,1);
  // Accepted old IDs cannot cancel a later event.
  feed(p,tone);feed(p,new Float32Array(Math.round(rate*.15)));const later=p.messages.filter(m=>m.phase==='early').at(-1);assert.ok(later.id>early.id);
  p.node.port.onmessage({data:{type:'accepted',id:early.id}});feed(p,new Float32Array(rate*2));assert.ok(p.messages.some(m=>m.phase==='fallback'&&m.id===later.id));
}
console.log('PASS: early/fallback timing, accepted-ID cancellation, stale IDs, source samples/timestamps, timeout, stop, both rates');
