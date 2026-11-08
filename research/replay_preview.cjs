// Run recorded clips through the exact live worklet; temporary outputs only.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const out=process.argv[2];if(!out)throw new Error('A temporary output directory is required.');
const code=fs.readFileSync(path.join(__dirname,'..','research/worklets/preview-worklet.js'),'utf8'),rows=[];
function decode(blob){let rate,encoding,data;for(let i=12;i+8<=blob.length;){const n=blob.readUInt32LE(i+4),tag=blob.toString('ascii',i,i+4);if(tag==='fmt '){encoding=blob.readUInt16LE(i+8);rate=blob.readUInt32LE(i+12);}if(tag==='data')data=blob.subarray(i+8,i+8+n);i+=8+n+(n%2);}const width=encoding===3?4:2;const x=new Float32Array(data.length/width);for(let i=0;i<x.length;i++)x[i]=encoding===3?data.readFloatLE(i*4):data.readInt16LE(i*2)/32768;return {x,rate};}
function wav(x,rate){const b=Buffer.alloc(44+x.length*4);b.write('RIFF');b.writeUInt32LE(b.length-8,4);b.write('WAVEfmt ',8);b.writeUInt32LE(16,16);b.writeUInt16LE(3,20);b.writeUInt16LE(1,22);b.writeUInt32LE(rate,24);b.writeUInt32LE(rate*4,28);b.writeUInt16LE(4,32);b.writeUInt16LE(32,34);b.write('data',36);b.writeUInt32LE(x.length*4,40);for(let i=0;i<x.length;i++)b.writeFloatLE(x[i],44+i*4);return b;}
for(const sid of ['20261103T165139_cf7201','20261103T200050_87dcb0']){
  const directory=path.join(__dirname,'..','data',sid),session=JSON.parse(fs.readFileSync(path.join(directory,'session.json')));
  for(const clip of session.clips.filter(c=>c.accepted)){
    const {x,rate}=decode(fs.readFileSync(path.join(directory,clip.filename))),windows=[];let Type;
    vm.runInNewContext(code,{AudioWorkletProcessor:class{constructor(){this.port={postMessage:m=>{if(m.type==='candidate')windows.push(m)}}}},registerProcessor:(_,type)=>Type=type,Float32Array,Math,sampleRate:rate});
    const node=new Type(),feed=signal=>{for(let i=0;i<signal.length;i+=128)node.process([[signal.slice(i,i+128)]])};
    feed(new Float32Array(Math.round(rate*1.2)));feed(x);feed(new Float32Array(Math.round(rate*1.5)));
    const files=windows.map((w,i)=>{const name=`${sid}_${clip.filename}_${i}.wav`;fs.writeFileSync(path.join(out,name),wav(w.pcm,rate));return {file:name,start_seconds:w.startFrame/rate};});
    rows.push({session:sid,file:clip.filename,truth:clip.label,windows:files});
  }
}
fs.writeFileSync(path.join(out,'windows.json'),JSON.stringify(rows));
