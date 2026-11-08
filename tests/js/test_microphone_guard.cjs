const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync('web/app.js','utf8');
async function check(label,deviceId,allowed){
  let graph=false,stopped=false,message='';
  const track={label,getSettings:()=>({deviceId,sampleRate:48000}),stop:()=>stopped=true};
  const elements={connect:{},device:{value:deviceId}};
  const env={stream:null,$:id=>elements[id],window:{AudioWorkletNode:function(){}},
    navigator:{mediaDevices:{getUserMedia:async()=>({getAudioTracks:()=>[track],getTracks:()=>[track]})}},
    AudioContext:class{constructor(){graph=true;throw Error('test stops at audio graph');}},
    disconnect:async()=>{track.stop();env.stream=null;},devices:async()=>{},notice:m=>message=m,update:()=>{}};
  vm.createContext(env);
  if(source.includes('function requireBuiltInMicrophone('))vm.runInContext(source.slice(source.indexOf('function requireBuiltInMicrophone('),source.indexOf('async function connect(')),env);
  vm.runInContext(source.slice(source.indexOf('async function connect('),source.indexOf('function encodeWav(')),env);
  await env.connect();
  assert.equal(graph,allowed,`${label}: audio graph must only open for verified explicit built-in input`);
  assert.ok(stopped);
  if(!allowed)assert.match(message,/built-in/i);
}
(async()=>{
 await check('Default - Example Bluetooth Headset (Bluetooth)','default',false);
 await check('Example Bluetooth Headset (Bluetooth)','bose-id',false);
 await check('Default - MacBook Pro Microphone (Built-in)','default',false);
 await check('Default - MacBook Pro Microphone (Built-in)','resolved-macbook-id',false);
 await check('','unknown',false);
 await check('MacBook Pro Microphone (Built-in)','macbook-id',true);
 console.log('PASS: real connect rejects Bose, default aliases and unknown inputs before audio capture; explicit MacBook input proceeds.');
})().catch(e=>{console.error(e);process.exitCode=1;});
