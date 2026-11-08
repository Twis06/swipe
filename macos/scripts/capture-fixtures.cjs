// Deterministic public synthetic signal; compare the Swift port with the shipped worklet.
const fs=require('node:fs'),vm=require('node:vm'),crypto=require('node:crypto');
const cases=[];
for(const rate of [44100,48000]) for(const acceptEarly of [false,true]) {
 let Type;const messages=[];
 vm.runInNewContext(fs.readFileSync('web/cascade-preview-worklet.js','utf8'),{AudioWorkletProcessor:class{constructor(){this.port={postMessage:m=>messages.push(m)}}},registerProcessor:(_,t)=>Type=t,Float32Array,Math,sampleRate:rate});
 const node=new Type(),warm=Math.round(rate*1.2),tone=Math.round(rate*.1),length=rate*4;
 const all=Float32Array.from({length},(_,i)=>i>=warm&&i<warm+tone?((i-warm)%29-14)/16:0);
 const candidates=[];let seen=0;
 for(let i=0;i<all.length;i+=128){
  node.process([[all.slice(i,i+128)]]);
  for(const m of messages.slice(seen)) if(m.type==='candidate'){
   candidates.push({...Object.fromEntries(Object.entries(m).filter(([k])=>k!=='pcm'&&k!=='type')),sha256:crypto.createHash('sha256').update(Buffer.from(m.pcm.buffer)).digest('hex')});
   if(acceptEarly&&m.phase==='early')node.port.onmessage({data:{type:'accepted',id:m.id}});
  }
  seen=messages.length;
 }
 cases.push({rate,acceptEarly,candidates});
}
fs.writeFileSync('macos/Tests/SwipeCoreTests/capture-fixtures.json',JSON.stringify(cases,null,2)+'\n');
