// Replay real continuous samples through the unchanged live worklet.
// No synthetic quiet lead-in/tail. Does not simulate HTTP latency/dropped requests.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const [input,out]=process.argv.slice(2);
if(!input||!out)throw new Error('Usage: node replay_continuous.cjs input.wav output-directory');
const blob=fs.readFileSync(input);let rate,encoding,data;
for(let i=12;i+8<=blob.length;){const size=blob.readUInt32LE(i+4),tag=blob.toString('ascii',i,i+4);if(tag==='fmt '){encoding=blob.readUInt16LE(i+8);rate=blob.readUInt32LE(i+12);}if(tag==='data')data=blob.subarray(i+8,i+8+size);i+=8+size+size%2;}
if(encoding!==3||!data||data.length%4)throw new Error('Expected Float32 WAV');
const x=new Float32Array(data.length/4);for(let i=0;i<x.length;i++)x[i]=data.readFloatLE(i*4);
const windows=[];let Type;
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'..','research/worklets/preview-worklet.js'),'utf8'),{
  AudioWorkletProcessor:class{constructor(){this.port={postMessage:m=>{if(m.type==='candidate')windows.push(m)}}}},
  registerProcessor:(_,type)=>Type=type,Float32Array,Math,sampleRate:rate});
const processor=new Type();for(let i=0;i<x.length;i+=128)processor.process([[x.slice(i,i+128)]]);
fs.mkdirSync(out,{recursive:true});
const rows=windows.map((w,i)=>{const file=`window-${i}.f32`;const b=Buffer.alloc(w.pcm.length*4);for(let j=0;j<w.pcm.length;j++)b.writeFloatLE(w.pcm[j],j*4);fs.writeFileSync(path.join(out,file),b);return {file,start_seconds:w.startFrame/rate,end_seconds:(w.startFrame+w.pcm.length)/rate};});
fs.writeFileSync(path.join(out,'windows.json'),JSON.stringify({rate,windows:rows}));
