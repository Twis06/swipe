const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const out=process.argv[2];if(!out)throw new Error('Temporary output directory required');
if(process.argv[3]==='suppress'){
  const source=fs.readFileSync(path.join(__dirname,'preview.js'),'utf8');
  const fn=source.slice(source.indexOf('function overlapsPreviousEvent('),source.indexOf('async function predictCandidate('));
  const rows=JSON.parse(fs.readFileSync(path.join(out,'classified.json')));
  for(const row of rows){const context={Number};vm.createContext(context);vm.runInContext('let lastEventEnd=-Infinity;\n'+fn,context);
    for(const window of row.windows){window.duplicate_suppressed=window.result.prediction!=='background'&&context.overlapsPreviousEvent(window.result,window.start_frame,row.rate);}}
  fs.writeFileSync(path.join(out,'classified.json'),JSON.stringify(rows));process.exit(0);
}
function decode(blob){let rate,encoding,data;for(let i=12;i+8<=blob.length;){const size=blob.readUInt32LE(i+4),tag=blob.toString('ascii',i,i+4);if(tag==='fmt '){encoding=blob.readUInt16LE(i+8);rate=blob.readUInt32LE(i+12);}if(tag==='data')data=blob.subarray(i+8,i+8+size);i+=8+size+size%2;}const width=encoding===3?4:2;return {rate,x:Float32Array.from({length:data.length/width},(_,i)=>encoding===3?data.readFloatLE(i*4):data.readInt16LE(i*2)/32768)};}
const continuous=JSON.parse(fs.readFileSync(path.join(__dirname,'diagnostics/continuous/label_review.json'))).take_ids;
const rows=[];
for(const sid of ['20261103T165139_cf7201','20261103T200050_87dcb0',...continuous]){
  const session=JSON.parse(fs.readFileSync(path.join(__dirname,'data',sid,'session.json')));
  for(const clip of session.clips.filter(c=>c.accepted)){
    const {x,rate}=decode(fs.readFileSync(path.join(__dirname,'data',sid,clip.filename)));
    for(const lead of (session.kind==='continuous'?['none']:['zero','repeated_recorded_background'])) for(const variant of ['preview-worklet.js','event-preview-worklet.js']){
      const windows=[];let Type,node,onset;
      const code=fs.readFileSync(path.join(__dirname,variant),'utf8');
      vm.runInNewContext(code,{AudioWorkletProcessor:class{constructor(){this.port={postMessage:m=>{if(m.type==='onset')onset=node.frames;if(m.type==='candidate'){const name=`w${rows.length}_${windows.length}.f32`;const b=Buffer.alloc(m.pcm.length*4);for(let j=0;j<m.pcm.length;j++)b.writeFloatLE(m.pcm[j],j*4);fs.writeFileSync(path.join(out,name),b);windows.push({file:name,start_frame:m.startFrame,end_frame:node.frames,onset_frame:onset,delay_seconds:(node.frames-onset)/rate,endpoint_reason:m.endpointReason||'fixed_wait',source_start_frame:m.sourceStartFrame??m.startFrame,source_end_frame:m.sourceEndFrame??node.frames,source_samples:m.sourceSamples??m.pcm.length,destination_offset:m.destinationOffset??0,peak:Math.max(...m.pcm)});}}}}},registerProcessor:(_,t)=>Type=t,Float32Array,Math,sampleRate:rate});
      node=new Type();const feed=signal=>{for(let i=0;i<signal.length;i+=128)node.process([[signal.slice(i,i+128)]]);};
      if(session.kind!=='continuous'){const intro=new Float32Array(Math.round(rate*1.2));if(lead==='repeated_recorded_background'){const chunk=x.slice(0,Math.round(rate*.25));for(let i=0;i<intro.length;i++)intro[i]=chunk[i%chunk.length];}feed(intro);}feed(x);if(session.kind!=='continuous')feed(new Float32Array(Math.round(rate*1.5)));
      rows.push({session:sid,file:clip.filename,kind:session.kind||'isolated',truth:clip.label,rate,variant,lead,windows});
    }
  }
}
fs.writeFileSync(path.join(out,'windows.json'),JSON.stringify(rows));
