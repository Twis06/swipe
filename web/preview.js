const el = id => document.getElementById(id);
let token, modelReady = false, active = false, starting = false, generation = 0;
let stream, context, recorder, mute, request, resultTimer;
let history = [];
let lastEventEnd = -Infinity;
let queuedFallback = null, acceptedCaptureId = null;
const paths = {ready:'M50 26v68M30 46l20-20 20 20M30 74l20 20 20-20',up:'M50 98V22M25 47L50 22l25 25',down:'M50 22V98M25 73l25 25 25-25',uncertain:'M33 39c0-22 35-22 35 0 0 16-18 15-18 32M50 87v1',listening:'M25 46v28M40 32v56M55 24v72M70 38v44',error:'M50 30v40M50 87v1'};
function show(kind, title, detail) {
  el('resultStage').dataset.result = kind;
  el('result').textContent = title; el('resultDetail').textContent = detail;
  el('resultIcon').innerHTML = `<svg viewBox="0 0 100 120"><path d="${paths[kind] || paths.ready}"/></svg>`;
}
function notice(text, error=false) {el('previewNotice').textContent=text;el('previewNotice').className=error?'error':'';}
function controls() {
  el('startPreview').disabled = !modelReady || starting || active;
  el('startPreview').textContent = starting ? 'Connecting…' : 'Start preview';
  el('stopPreview').disabled = !active && !starting;
  el('previewDevice').disabled = active || starting;
}
async function devices() {
  const selected=el('previewDevice').value;
  const inputs=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='audioinput');
  el('previewDevice').replaceChildren(new Option('System default — use built-in mic',''));
  inputs.forEach((d,i)=>el('previewDevice').add(new Option(d.label||`Microphone ${i+1}`,d.deviceId)));
  const builtin=inputs.find(d=>/MacBook.*Microphone|Built.in/i.test(d.label));
  if(inputs.some(d=>d.deviceId===selected))el('previewDevice').value=selected;
  else if(builtin)el('previewDevice').value=builtin.deviceId;
}
function wav(pcm, rate) {
  const bytes=new ArrayBuffer(58+pcm.length*4),v=new DataView(bytes);
  const text=(at,s)=>[...s].forEach((c,i)=>v.setUint8(at+i,c.charCodeAt(0)));
  text(0,'RIFF');v.setUint32(4,bytes.byteLength-8,true);text(8,'WAVE');text(12,'fmt ');
  v.setUint32(16,18,true);v.setUint16(20,3,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);
  v.setUint32(28,rate*4,true);v.setUint16(32,4,true);v.setUint16(34,32,true);v.setUint16(36,0,true);
  text(38,'fact');v.setUint32(42,4,true);v.setUint32(46,pcm.length,true);text(50,'data');v.setUint32(54,pcm.length*4,true);
  for(let i=0;i<pcm.length;i++){if(!Number.isFinite(pcm[i]))throw new Error('Invalid microphone samples. Reconnect and try again.');v.setFloat32(58+i*4,pcm[i],true);}
  return new Blob([bytes],{type:'audio/wav'});
}
function drawHistory() {
  el('predictionHistory').replaceChildren();
  for(const item of history){const li=document.createElement('li'),label=document.createElement('span'),time=document.createElement('time');label.textContent=item.label;time.dateTime=item.time.toISOString();time.textContent=item.time.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});li.append(label,time);el('predictionHistory').append(li);}
  el('historyEmpty').hidden=history.length>0;el('clearHistory').disabled=!history.length;
}
async function stop(message='Microphone released. Start again whenever you’re ready.') {
  generation++;active=false;starting=false;clearTimeout(resultTimer);
  window.swipeKeys?.setListening(false);
  void window.swipeKeys?.stop();
  request?.abort();request=null;
  queuedFallback=null;acceptedCaptureId=null;
  recorder?.port.postMessage({type:'stop'});recorder?.disconnect();recorder=null;
  stream?.getTracks().forEach(t=>{t.onended=null;t.stop();});stream=null;
  const previous=context;context=null;if(previous && previous.state!=='closed')await previous.close();
  el('listenState').textContent='Microphone off';el('micName').textContent='Microphone disconnected';
  el('peakText').textContent='— dBFS';el('previewLevel').value=-90;
  show('ready','Preview stopped','Your saved recording sessions are unchanged.');notice(message);controls();
}
function overlapsPreviousEvent(result, windowStart, rate) {
  const start=windowStart/rate+result.quality.event_start_seconds;
  const end=windowStart/rate+result.quality.event_end_seconds;
  if(!Number.isFinite(start)||!Number.isFinite(end))return true;
  if(start<=lastEventEnd+.1)return true;
  lastEventEnd=end;return false;
}
async function predictCandidate(pcm, rate, epoch, windowStart=0, captureId=null, phase='fallback') {
  if(!active || epoch!==generation)return;
  if(captureId!==null && captureId===acceptedCaptureId)return;
  if(request){
    // Retain one fallback if its early request is still pending. Newer fallback
    // replaces older queued work; there is never an unbounded request queue.
    if(phase==='fallback' && captureId!==null)queuedFallback={pcm,rate,epoch,windowStart,captureId,phase};
    return;
  }
  const controller=new AbortController();request=controller;
  const timeout=setTimeout(()=>controller.abort(),6000);
  try {
    const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'audio/wav','X-Recorder-Token':token,'X-Swipe-Client':window.swipeKeys?.client||''},body:wav(pcm,rate),signal:controller.signal});
    const result=await response.json();if(!response.ok)throw new Error(result.error||'Prediction failed.');
    if(!active || epoch!==generation)return;
    if(phase==='early'){
      if(result.prediction!=='up' && result.prediction!=='down')return;
      acceptedCaptureId=captureId;
      if(queuedFallback?.captureId===captureId)queuedFallback=null;
      recorder?.port.postMessage({type:'accepted',id:captureId});
    }
    clearTimeout(resultTimer);el('listenState').textContent='Listening';
    if(result.prediction==='background') {show('listening','Listening for a swipe','That sound was ignored. Try one swipe along the left grille.');return;}
    if(overlapsPreviousEvent(result,windowStart,rate)){show('listening','Listening for a swipe','That swipe was already checked. Wait a moment, then try the next one.');return;}
    const label=result.prediction==='up'?'Up':result.prediction==='down'?'Down':'Uncertain';
    const detail=result.prediction==='up'?'Toward the screen.':result.prediction==='down'?'Toward the palm rest.':result.reason==='unfamiliar_sound'?'That sound did not match the recorded swipes closely enough.':'The sound matched both directions too closely.';
    const sentKey=await window.swipeKeys?.accepted(result,()=>active && epoch===generation);
    if(!active || epoch!==generation)return;
    show(result.prediction,label,sentKey ? `Sent ${sentKey} to the active app.` : detail);
    history.unshift({label,time:new Date()});history=history.slice(0,6);drawHistory();
    notice('Lift your finger, wait a moment, then try the next swipe.');
    resultTimer=setTimeout(()=>{if(active && generation===epoch)show('listening','Listening for a swipe','One movement, then wait for the result.');},5000);
  } catch(error) {
    if(epoch!==generation || !active)return;
    await stop();show('error','Preview interrupted','Start again to reconnect to the local model.');
    notice(error.name==='AbortError'?'The local model did not respond in time. Start the preview again.':error.message,true);
  } finally {
    clearTimeout(timeout);
    if(request===controller){
      request=null;
      const next=queuedFallback;queuedFallback=null;
      if(next && active && next.epoch===generation && next.captureId!==acceptedCaptureId){
        void predictCandidate(next.pcm,next.rate,next.epoch,next.windowStart,next.captureId,next.phase);
      }
    }
  }
}
async function start() {
  if(active || starting || !modelReady)return;
  starting=true;lastEventEnd=-Infinity;queuedFallback=null;acceptedCaptureId=null;const epoch=++generation;controls();notice('Connecting to your microphone…');
  try {
    if(!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode)throw new Error('Use Chrome or Safari at the localhost address.');
    const acquired=await navigator.mediaDevices.getUserMedia({audio:{deviceId:el('previewDevice').value?{exact:el('previewDevice').value}:undefined,channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
    if(epoch!==generation){acquired.getTracks().forEach(t=>t.stop());return;}
    stream=acquired;const track=stream.getAudioTracks()[0];
    context=new AudioContext({sampleRate:track.getSettings().sampleRate||48000});const localContext=context;
    await localContext.resume();await localContext.audioWorklet.addModule('/preview-worklet.js');
    if(epoch!==generation)return;
    recorder=new AudioWorkletNode(localContext,'gesture-preview',{channelCount:1,channelCountMode:'explicit'});
    mute=localContext.createGain();mute.gain.value=0;
    localContext.createMediaStreamSource(stream).connect(recorder);recorder.connect(mute);mute.connect(localContext.destination);
    active=true;starting=false;window.swipeKeys?.setListening(true);track.onended=()=>stop('Microphone disconnected. Start again to reconnect.');
    el('micName').textContent=track.label||'Microphone connected';el('listenState').textContent='Checking background';
    show('listening','Keep still for one second','The preview is measuring the background sound.');
    let warmed=false;
    recorder.port.onmessage=({data})=>{
      if(!active || epoch!==generation)return;
      if(data.type==='level'){
        const db=Math.max(-90,20*Math.log10(data.peak||1e-9));el('peakText').textContent=`Peak ${db.toFixed(0)} dBFS`;el('previewLevel').value=Math.min(0,db);
        if(!data.warming && !warmed){warmed=true;el('listenState').textContent='Listening';show('listening','Listening for a swipe','One movement, then wait for the result.');notice('Preview audio stays in memory and is not saved.');}
      }
      if(data.type==='onset' && !request){clearTimeout(resultTimer);el('listenState').textContent='Checking a sound';show('listening','Checking that sound…','Lift your finger and wait for the result.');}
      if(data.type==='candidate')predictCandidate(data.pcm,localContext.sampleRate,epoch,data.startFrame,data.id??null,data.phase??'fallback');
    };
    await devices();controls();
  } catch(error){if(epoch!==generation)return;await stop();show('error','Couldn’t start the preview','Check microphone access, then try again.');notice(error.message,true);}
}
el('startPreview').onclick=start;el('stopPreview').onclick=()=>stop();
el('clearHistory').onclick=()=>{history=[];drawHistory();};
document.addEventListener('visibilitychange',()=>{if(document.hidden && (active||starting) && !window.swipeKeys?.isArmed())stop('Preview stopped when this tab became hidden. Start again when ready.');});
window.addEventListener('pagehide',()=>{if(active||starting)stop();});
(async()=>{try{const response=await fetch('/api/preview');const info=await response.json();if(!response.ok)throw new Error(info.error||'Local model unavailable.');token=info.token;modelReady=info.ready;await window.swipeKeys?.configure(token);el('modelDescription').textContent=`The experimental recognizer uses ${info.reference_swipes ?? info.training_clips} ${info.reference_swipes ? 'swipe examples' : 'clips'} from ${info.training_sessions} recording sessions. No confidence percentage is shown because its scores are not calibrated probabilities.`;notice('Ready. Start the preview when your finger is in position.');await devices();}catch(error){notice(error.message,true);el('modelDescription').textContent='The local model is unavailable.';show('error','Model unavailable','Restart Swipe with its analysis environment, then reload this page.');}finally{controls();}})();
