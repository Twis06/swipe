const $ = id => document.getElementById(id);
const names = {up: 'Swipe up', down: 'Swipe down', idle: 'Keep still', typing: 'Type normally'};
let token, allSessions = [], session = null, stream, context, source, analyser, recorder, mute;
let running = false, busy = false, captureJob = null, captureID = 0, pending = null;
let blockGate = null, meterFrame = null, livePeak = 0, lastHighLevel = -Infinity, lastPossibleClipping = -Infinity;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const accepted = () => session?.clips.filter(c => c.accepted) || [];
const nextIndex = () => session?.plan.findIndex((_, i) => !accepted().some(c => c.trial_index === i)) ?? -1;

async function api(path, value, binary = false) {
  const response = await fetch(path, value === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': binary ? 'audio/wav' : 'application/json', 'X-Recorder-Token': token},
    body: binary ? value : JSON.stringify(value)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Could not reach local recorder.');
  return result;
}
function notice(message, error = false) { $('notice').textContent = message; $('notice').className = error ? 'error' : ''; }
function showCue(phase, title, detail, label = 'up') {
  $('typingTarget').hidden = !(phase === 'record' && label === 'typing');
  if ($('typingTarget').hidden) { $('typingTarget').value = ''; $('typingTarget').blur(); }
  $('stage').dataset.phase = phase;
  $('phase').textContent = {ready:'Microphone check', prep:'Get ready', record:'Record · 2 seconds', reset:'Lift & reset', paused:'Paused', done:'Session complete', saving:'Saving to this Mac'}[phase] || phase;
  $('cue').textContent = title; $('instruction').textContent = detail;
  const paths = {up:'M50 98V22M25 47L50 22l25 25', down:'M50 22V98M25 73l25 25 25-25', idle:'M35 30v60M65 30v60', typing:'M15 35h70v50H15zM25 47h4m12 0h4m12 0h4m12 0h4M25 59h4m12 0h4m12 0h4m12 0h4M30 72h40'};
  $('gestureIcon').innerHTML = `<svg viewBox="0 0 100 120"><path d="${paths[label]}"/></svg>`;
}
function update() {
  const count = accepted().length, total = session?.plan.length || 0;
  $('saved').textContent = `${count} / ${total}`;
  $('progress').max = total || 1; $('progress').value = count;
  $('trialCount').textContent = session ? `${count} of ${total} saved` : 'No session started';
  $('start').disabled = busy || !stream || !$('confirmed').checked;
  $('start').textContent = pending ? 'Retry saving & continue' : session && nextIndex() >= 0 ? 'Continue session' : session ? 'Start another session' : 'Start session';
  $('pause').disabled = !running;
  $('redo').disabled = busy || !count;
  $('connect').disabled = busy;
  $('connect').textContent = stream ? 'Disconnect microphone' : 'Connect microphone';
  $('device').disabled = !!stream || busy;
  ['size','surface','notes'].forEach(id => $(id).disabled = busy || (!!session && nextIndex() >= 0));
  $('resume').disabled = busy; $('sessions').disabled = busy;
  $('empty').hidden = count > 0; $('export').hidden = !session;
  if (session) { $('export').href = `/api/export/${session.id}`; $('export').download = `${session.id}.zip`; }
  $('clipList').replaceChildren();
  for (const clip of accepted().slice(-6).reverse()) {
    const li = document.createElement('li'), title = document.createElement('span'), info = document.createElement('small');
    title.textContent = `${String(clip.trial_index + 1).padStart(2,'0')} · ${names[clip.label]}`;
    const quality = clip.possible_clipping ? ' · possible clipping' : clip.level_assessment === 'high_peak' ? ' · high peak, preserved' : clip.quality_warning || (clip.encoding !== 'float32' && clip.peak >= .99) ? ' · level warning' : clip.rms < .00001 ? ' · very quiet' : '';
    info.textContent = `Block ${clip.block} · ${clip.duration_seconds.toFixed(1)}s${quality}`;
    li.append(title, info); $('clipList').append(li);
  }
}
function makePlan() {
  const plan = [], full = $('size').value === 'full', perDirection = full ? 10 : 5;
  for (let block = 1; block <= 2; block++) {
    const trials = [];
    for (const label of ['up','down']) for (let i = 0; i < perDirection; i++) trials.push({label, block, pace: ['natural','slow','brisk'][i % 3]});
    for (const label of ['idle','typing']) for (let i = 0; i < (full ? 4 : 2); i++) trials.push({label, block, pace:'natural'});
    for (let i = trials.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [trials[i], trials[j]] = [trials[j], trials[i]]; }
    plan.push(...trials);
  }
  return plan;
}
function levelStatus(peak, highAt, clippingAt, now) {
  const text = `Peak ${Math.max(-90,20*Math.log10(peak || 1e-9)).toFixed(0)} dBFS`;
  if(now-clippingAt<3000) return {kind:'clipping',text:text+' · possible clipping',color:'#a22c24'};
  if(now-highAt<3000) return {kind:'high',text:text+' · high peak',color:'#8a5a18'};
  return {kind:'normal',text,color:''};
}
function drawMeter() {
  if (!analyser) return;
  const samples = new Float32Array(analyser.fftSize); analyser.getFloatTimeDomainData(samples);
  const rms = Math.sqrt(samples.reduce((sum, x) => sum + x*x, 0) / samples.length);
  const db = Math.max(-90, 20 * Math.log10(rms || 1e-9));
  const status=levelStatus(livePeak,lastHighLevel,lastPossibleClipping,performance.now());
  $('level').textContent=status.text; $('level').style.color=status.color;
  const canvas = $('meter'), ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for (let i=0;i<26;i++) { ctx.fillStyle = i < (db+90)/90*26 ? '#14664e' : '#d3ddd2'; ctx.fillRect(i*10,5,6,20); }
  meterFrame = requestAnimationFrame(drawMeter);
}
async function devices() {
  const selected = $('device').value;
  const inputs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
  $('device').replaceChildren(new Option('System default — select built-in mic', ''));
  for (const [i, device] of inputs.entries()) $('device').add(new Option(device.label || `Microphone ${i+1}`, device.deviceId));
  if ([...$('device').options].some(o => o.value === selected)) $('device').value = selected;
  const builtin = inputs.find(d => !['default','communications',''].includes(d.deviceId) && /macbook.*microphone|built-in.*microphone|internal microphone/i.test(d.label));
  if (builtin && ['', 'default', 'communications'].includes(selected)) $('device').value = builtin.deviceId;
}
async function disconnect() {
  running = false;
  if (captureJob) captureJob.reject(new Error('Recording interrupted. The unfinished clip was discarded.'));
  captureJob = null; if (meterFrame) cancelAnimationFrame(meterFrame);
  stream?.getTracks().forEach(t => { t.onended = null; t.stop(); }); stream = null;
  if (context && context.state !== 'closed') await context.close();
  analyser = null; context = null; livePeak=0;lastHighLevel=-Infinity;lastPossibleClipping=-Infinity;
  $('level').textContent='— dBFS';$('level').style.color='';
  $('confirmed').checked = false; $('micStatus').textContent = 'Microphone disconnected'; update();
}
function requireBuiltInMicrophone(track) {
  const id = track?.getSettings().deviceId;
  if (!id || ['default','communications'].includes(id) ||
      /^(default|communications)\b/i.test(track?.label || '') ||
      !/macbook.*microphone|built-in.*microphone|internal microphone/i.test(track?.label || '')) {
    throw new Error(`Detected ${track?.label || 'an unidentified microphone'}. Choose the explicit MacBook built-in microphone in the Microphone menu, then reconnect. System default and headset microphones cannot be used for this session.`);
  }
}
async function connect() {
  if (stream) { await disconnect(); return; }
  $('connect').disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) throw new Error('Use a current Chrome or Safari browser at the localhost address.');
    stream = await navigator.mediaDevices.getUserMedia({audio:{deviceId:$('device').value ? {exact:$('device').value} : undefined, channelCount:1, echoCancellation:false, noiseSuppression:false, autoGainControl:false},video:false});
    await devices();
    requireBuiltInMicrophone(stream.getAudioTracks()[0]);
    context = new AudioContext({sampleRate:stream.getAudioTracks()[0].getSettings().sampleRate || 48000}); await context.resume();
    await context.audioWorklet.addModule('/audio-worklet.js');
    source = context.createMediaStreamSource(stream); analyser = context.createAnalyser(); analyser.fftSize = 2048;
    recorder = new AudioWorkletNode(context, 'capture', {channelCount:1,channelCountMode:'explicit'});
    mute = context.createGain(); mute.gain.value = 0;
    source.connect(analyser); source.connect(recorder); recorder.connect(mute); mute.connect(context.destination);
    recorder.port.onmessage = ({data}) => {
      if (data.type === 'level') {
        livePeak=data.peak;
        if (data.peak>=.999) lastHighLevel=performance.now();
        if (data.flatTopFraction>.001) lastPossibleClipping=performance.now();
        return;
      }
      if (captureJob?.id === data.id) { const job = captureJob; captureJob = null; job.resolve(data.pcm); }
    };
    const track = stream.getAudioTracks()[0];
    track.onended = () => { notice('Microphone disconnected. Reconnect it to continue.', true); disconnect(); };
    $('micStatus').textContent = track.label || 'Microphone connected';
    $('settings').textContent = JSON.stringify({browser_settings:track.getSettings(),capture_sample_rate:context.sampleRate},null,2);
    await devices(); drawMeter();
    notice('Gently rub the left grille and check that the level moves. Confirm the microphone at the right, then start.');
  } catch (e) {
    await disconnect();
    notice(`Microphone unavailable: ${e.message}`,true);
  } finally { update(); }
}
function encodeWav(pcm, rate) {
  // Preserve the actual Float32 samples, including legitimate excursions
  // beyond +/-1. PCM16 clamping previously destroyed diagnostic information.
  const buffer = new ArrayBuffer(58 + pcm.length*4), view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((c,i) => view.setUint8(offset+i,c.charCodeAt(0)));
  text(0,'RIFF'); view.setUint32(4,50+pcm.length*4,true); text(8,'WAVE'); text(12,'fmt ');
  view.setUint32(16,18,true); view.setUint16(20,3,true); view.setUint16(22,1,true);
  view.setUint32(24,rate,true); view.setUint32(28,rate*4,true); view.setUint16(32,4,true); view.setUint16(34,32,true); view.setUint16(36,0,true);
  text(38,'fact'); view.setUint32(42,4,true); view.setUint32(46,pcm.length,true);
  text(50,'data'); view.setUint32(54,pcm.length*4,true);
  for(let i=0;i<pcm.length;i++) {
    if(!Number.isFinite(pcm[i])) throw new Error('Microphone produced invalid audio samples. Reconnect and retry.');
    view.setFloat32(58+i*4,pcm[i],true);
  }
  return new Blob([buffer],{type:'audio/wav'});
}
async function capture() {
  const id = ++captureID;
  let timer;
  try {
    return await new Promise((resolve,reject) => {
      captureJob = {id,resolve,reject};
      timer = setTimeout(() => { if(captureJob?.id===id) { captureJob=null; recorder.port.postMessage({type:'cancel'}); reject(new Error('No audio arrived. Reconnect the microphone and retry.')); } }, 5000);
      recorder.port.postMessage({type:'capture',frames:context.sampleRate*2,id});
    });
  } finally { clearTimeout(timer); }
}
async function delay(ms) {
  const end = performance.now()+ms;
  while (performance.now()<end && running) await sleep(Math.min(50,end-performance.now()));
  return running;
}
async function savePending() {
  if (!pending) return;
  session = await api(`/api/clip/${session.id}/${pending.index}`,pending.blob,true);
  const saved = session.clips.find(c=>c.trial_index===pending.index && c.accepted);
  pending = null;
  if (saved?.possible_clipping ?? saved?.quality_warning) {
    running=false; $('confirmed').checked=false;
    showCue('paused','Possible clipping detected','The clip is saved for review. Lower Input volume in macOS Sound settings, then check a normal swipe before continuing.','idle');
    notice('Flat-topped samples suggest clipping before saving. Collection paused. A high float peak alone does not trigger this pause.',true);
  }
  update();
}
async function run() {
  if (busy || !stream || !$('confirmed').checked) return;
  busy = true; running = true; update();
  try {
    requireBuiltInMicrophone(stream.getAudioTracks()[0]);
    await context.resume();
    if (!session || nextIndex() < 0) {
      session = await api('/api/session',{plan:makePlan(),metadata:{surface:$('surface').value,notes:$('notes').value,finger_area:'left speaker grille',direction_definition:'up=toward screen; down=toward palm rest',encoding:'float32',microphone_label:stream.getAudioTracks()[0].label,audio_settings:stream.getAudioTracks()[0].getSettings(),capture_sample_rate:context.sampleRate,user_agent:navigator.userAgent}});
      blockGate = null;
    }
    await savePending();
    update(); if(running) notice('Follow the silent cues. Use Pause if you need a break; unfinished clips are discarded.');
    while (running && nextIndex() >= 0) {
      const index = nextIndex(), trial = session.plan[index];
      if (trial.block === 2 && blockGate !== session.id && !accepted().some(c => c.block === 2)) {
        blockGate = session.id; running = false;
        showCue('paused','First block saved','Take a short break, lift your hand, then continue for a fresh second block.','idle');
        notice('Block 1 is saved. Continue when you are ready for block 2.'); break;
      }
      const detail = trial.label === 'up' ? `Start at the palm-rest end. One ${trial.pace} swipe toward the screen.` : trial.label === 'down' ? `Start at the screen end. One ${trial.pace} swipe toward the palm rest.` : trial.label === 'idle' ? 'Hands off the laptop. Let the microphone hear the room.' : 'Type a few ordinary words during the recording. Do not touch the grille.';
      for(let n=3;n>0 && running;n--) { showCue('prep',`${names[trial.label]} in ${n}`,detail,trial.label); $('trialFill').style.width='0%'; await delay(700); }
      if (!running) break;
      showCue('record',trial.label === 'idle' ? 'Keep still' : trial.label === 'typing' ? 'Type now' : `${names[trial.label]} now`,detail,trial.label);
      if (trial.label === 'typing') $('typingTarget').focus({preventScroll:true});
      await new Promise(requestAnimationFrame);
      if (!running) break;
      requireBuiltInMicrophone(stream.getAudioTracks()[0]);
      const startTime = performance.now();
      const visual = setInterval(() => {$('trialFill').style.width=`${Math.min(100,(performance.now()-startTime)/20)}%`;},30);
      let pcm;
      try { pcm = await capture(); } finally { clearInterval(visual); }
      if (!running) break;
      $('trialFill').style.width='100%';
      pending = {index,blob:encodeWav(pcm,context.sampleRate)};
      showCue('saving','Hold still','Saving this clip before the next prompt.',trial.label);
      await savePending();
      if (!running) break;
      showCue('reset','Lift & reset','Lift your finger off the grille before moving it back.',trial.label);
      await delay(1700);
    }
    if (nextIndex() < 0) {
      await disconnect();
      showCue('done','Your session is saved','Download the dataset below, or collect another session later.','idle');
      notice(`${accepted().length} WAV clips saved. Microphone disconnected. These recordings are ready for evaluation.`);
    }
  } catch (e) {
    showCue('paused','Let’s pause here',pending ? 'The finished clip is kept in memory. Continue to retry saving it.' : 'Your saved clips are safe. Continue to retry the interrupted trial.','idle');
    notice(e.message,true);
  } finally { running = false; busy = false; update(); }
}
function pause() {
  running = false;
  recorder?.port.postMessage({type:'cancel'});
  if (captureJob) { captureJob.reject(new Error('Paused. The unfinished clip was discarded.')); captureJob = null; }
  showCue('paused','Take your time','Saved clips stay on disk. Continue when you are ready.','idle');
  notice('Paused. Any incomplete trial will be repeated.'); update();
}
$('connect').onclick = connect;
$('confirmed').onchange = update;
$('start').onclick = run;
$('pause').onclick = pause;
$('redo').onclick = async () => {
  const clip = accepted().at(-1); if (!clip || busy) return;
  busy = true; update();
  try { session = await api(`/api/reject/${session.id}/${clip.trial_index}`,{}); pending=null; showCue('paused','Last clip excluded','Continue to record that trial again. The previous take is marked rejected.','idle'); notice('Discarded take kept for audit, excluded from training.'); }
  catch(e) { notice(e.message,true); } finally { busy=false;update(); }
};
$('resume').onclick = () => {
  if (busy || pending) { notice('Save the pending clip before loading another session.',true); return; }
  session = allSessions.find(s => s.id === $('sessions').value); blockGate=null;
  $('surface').value=session.metadata.surface || 'On a desk'; $('notes').value=session.metadata.notes || '';
  $('size').value=session.plan.length > 28 ? 'full' : 'pilot'; $('confirmed').checked=false;
  showCue('paused','Session loaded','Connect and confirm the same built-in microphone, then continue.','idle'); update();
};
document.addEventListener('visibilitychange',()=>{if(document.hidden && running) pause();});
window.addEventListener('beforeunload',e=>{if(busy || pending){e.preventDefault(); e.returnValue='';}});
async function init() {
  try {
    const state = await api('/api/state'); token=state.token; allSessions=state.sessions.filter(s=>s.kind!=='continuous');
    $('storage').textContent=`Saved under ${state.data_directory}`;
    $('resumeArea').hidden=!allSessions.length;
    for(const s of allSessions) $('sessions').add(new Option(`${s.id} · ${s.clips.filter(c=>c.accepted).length}/${s.plan.length}`,s.id));
    $('connection').textContent='Connected · saved locally';
    if(navigator.mediaDevices) await devices();
  } catch(e) {notice(`Local server unavailable: ${e.message}`,true); $('connect').disabled=true;}
}
init();
