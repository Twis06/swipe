// Stateful JSON-lines worker: pause signal time at each candidate for inference.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),readline=require('node:readline');
const codes=Object.fromEntries(['preview-worklet.js','cascade-preview-worklet.js'].map(name=>[name,fs.readFileSync(path.join(__dirname,name),'utf8')]));
const preview=fs.readFileSync(path.join(__dirname,'preview.js'),'utf8');
const duplicateFn=preview.slice(preview.indexOf('function overlapsPreviousEvent('),preview.indexOf('async function predictCandidate('));
let state;
function decode(blob){let rate,encoding,data;for(let i=12;i+8<=blob.length;){const size=blob.readUInt32LE(i+4),tag=blob.toString('ascii',i,i+4);if(tag==='fmt '){encoding=blob.readUInt16LE(i+8);rate=blob.readUInt32LE(i+12);}if(tag==='data')data=blob.subarray(i+8,i+8+size);i+=8+size+size%2;}const width=encoding===3?4:2;return {rate,x:Float32Array.from({length:data.length/width},(_,i)=>encoding===3?data.readFloatLE(i*4):data.readInt16LE(i*2)/32768)};}
function load(cmd){
  const {rate,x}=decode(fs.readFileSync(path.join(__dirname,'data',cmd.session,cmd.file)));
  const lead=cmd.lead==='none'?0:Math.round(rate*1.2),tail=cmd.lead==='none'?0:Math.round(rate*1.5);
  const signal=new Float32Array(lead+x.length+tail);signal.set(x,lead);
  if(cmd.lead==='repeated_recorded_background'){const chunk=x.slice(0,Math.round(rate*.25));for(let i=0;i<lead;i++)signal[i]=chunk[i%chunk.length];}
  const queue=[];let Type,node,onset;
  vm.runInNewContext(codes[cmd.variant],{AudioWorkletProcessor:class{constructor(){this.port={postMessage:m=>{if(m.type==='onset')onset=node.frames;if(m.type==='candidate')queue.push({...m,onsetFrame:m.onsetFrame??onset,emitFrame:m.emitFrame??node.frames,phase:m.phase||'fallback'});}}}},registerProcessor:(_,t)=>Type=t,Float32Array,Math,sampleRate:rate});
  node=new Type();const duplicate={Number};vm.createContext(duplicate);vm.runInContext('let lastEventEnd=-Infinity;\n'+duplicateFn,duplicate);
  state={rate,signal,node,queue,cursor:0,duplicate,variant:cmd.variant};return {rate,frames:signal.length};
}
function step(){
  const s=state;
  while(!s.queue.length&&s.cursor<s.signal.length){const end=Math.min(s.cursor+128,s.signal.length);s.node.process([[s.signal.slice(s.cursor,end)]]);s.cursor=end;}
  if(!s.queue.length)return {done:true};
  const w=s.queue.shift(),b=Buffer.alloc(w.pcm.length*4);for(let i=0;i<w.pcm.length;i++)b.writeFloatLE(w.pcm[i],i*4);
  const {pcm,...meta}=w;return {done:false,...meta,pcm_base64:b.toString('base64')};
}
function classify(cmd){
  const w=cmd.window,result=cmd.result,early=w.phase==='early',accepted=['up','down'].includes(result.prediction);
  // Early failures remain invisible and do not alter last displayed event time.
  if(early&&!accepted)return {displayed:false,duplicate_suppressed:false,acknowledged:false};
  const duplicate=result.prediction!=='background'&&state.duplicate.overlapsPreviousEvent(result,w.startFrame,state.rate);
  if(early&&accepted)state.node.port.onmessage({data:{type:'accepted',id:w.id}});
  return {displayed:result.prediction!=='background'&&!duplicate,duplicate_suppressed:duplicate,acknowledged:early&&accepted};
}
readline.createInterface({input:process.stdin}).on('line',line=>{try{const cmd=JSON.parse(line);const result=cmd.cmd==='load'?load(cmd):cmd.cmd==='step'?step():cmd.cmd==='classify'?classify(cmd):{error:'Unknown command'};process.stdout.write(JSON.stringify(result)+'\n');}catch(error){process.stdout.write(JSON.stringify({error:error.stack})+'\n');}});
