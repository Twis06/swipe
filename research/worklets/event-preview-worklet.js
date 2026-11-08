// Experimental v2: isolate source event, then center it in two seconds of zeros.
// Synthetic context changes model noise/gate statistics; not a calibrated input.
class GesturePreview extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pre = new Float32Array(Math.round(sampleRate * .1)); this.cursor = 0;
    this.frames = 0; this.onsetFrame = null; this.cooldownUntil = 0;
    this.noise = .0001; this.warmup = Math.round(sampleRate); this.lastLevel = 0;
    this.quietFrames = 0; this.offThreshold = .0015; this.cancelled = false; this.peak = 0;
    this.capture = null; this.captured = 0; this.sourceStart = 0;
    this.port.onmessage = ({data}) => {if (data.type === 'stop') {this.cancelled = true; this.onsetFrame = null; this.capture = null;}};
  }
  process(inputs) {
    if (this.cancelled) return false;
    const channel = inputs[0]?.[0]; if (!channel) return true;
    let power = 0;
    for (const value of channel) {
      this.pre[this.cursor] = value; this.cursor = (this.cursor+1)%this.pre.length;
      power += value*value; this.peak = Math.max(this.peak, Math.abs(value));
    }
    this.frames += channel.length;
    const rms = Math.sqrt(power/channel.length), warming = this.frames < this.warmup;
    if (warming) this.noise = .98*this.noise+.02*rms;
    const threshold = Math.max(.003, this.noise*6);
    let triggered = false;
    if (!warming && this.onsetFrame === null && this.frames >= this.cooldownUntil) {
      if (rms > threshold) {
        this.onsetFrame = this.frames; this.sourceStart = this.frames-this.pre.length;
        this.quietFrames = 0; this.offThreshold = threshold*.5;
        this.capture = new Float32Array(Math.round(sampleRate*1.2)+256);
        this.capture.set(this.pre.subarray(this.cursor));
        this.capture.set(this.pre.subarray(0,this.cursor), this.pre.length-this.cursor);
        this.captured = this.pre.length; triggered = true;
        this.port.postMessage({type:'onset'});
      } else this.noise = .995*this.noise+.005*rms;
    }
    if (this.onsetFrame !== null) {
      if (!triggered) {this.capture.set(channel,this.captured);this.captured+=channel.length;}
      this.quietFrames = rms < this.offThreshold ? this.quietFrames+channel.length : 0;
      const elapsed = this.frames-this.onsetFrame;
      const quiet = elapsed >= Math.round(sampleRate*.2) && this.quietFrames >= Math.round(sampleRate*.1);
      const timeout = elapsed >= Math.round(sampleRate);
      if (quiet || timeout) {
        const pcm = new Float32Array(Math.round(sampleRate*2));
        const offset = Math.floor((pcm.length-this.captured)/2);
        pcm.set(this.capture.subarray(0,this.captured),offset);
        this.port.postMessage({type:'candidate',pcm,startFrame:this.sourceStart-offset,
          onsetFrame:this.onsetFrame,emitFrame:this.frames,sourceStartFrame:this.sourceStart,
          sourceEndFrame:this.frames,sourceSamples:this.captured,destinationOffset:offset,
          endpointReason:timeout?'maximum_wait':'quiet'},[pcm.buffer]);
        this.onsetFrame=null;this.capture=null;this.cooldownUntil=this.frames+Math.round(sampleRate*.4);
      }
    }
    if(this.frames-this.lastLevel>=sampleRate/10){this.port.postMessage({type:'level',peak:this.peak,warming});this.peak=0;this.lastLevel=this.frames;}
    return true;
  }
}
registerProcessor('gesture-preview',GesturePreview);
