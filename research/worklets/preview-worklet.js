// A bounded two-second buffer. Only candidate gestures leave the worklet.
class GesturePreview extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring = new Float32Array(Math.round(sampleRate * 2));
    this.cursor = 0; this.frames = 0; this.pendingUntil = null; this.cooldownUntil = 0;
    this.noise = .0001; this.warmup = Math.round(sampleRate); this.lastLevel = 0;
    this.cancelled = false; this.peak = 0;
    this.port.onmessage = ({data}) => { if (data.type === 'stop') {this.cancelled = true; this.pendingUntil = null;} };
  }
  process(inputs) {
    if (this.cancelled) return false;
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    let power = 0;
    for (const value of channel) {
      this.ring[this.cursor] = value; this.cursor = (this.cursor + 1) % this.ring.length;
      power += value * value; this.peak = Math.max(this.peak, Math.abs(value));
    }
    this.frames += channel.length;
    const rms = Math.sqrt(power / channel.length);
    const warming = this.frames < this.warmup;
    if (warming) this.noise = .98 * this.noise + .02 * rms;
    const threshold = Math.max(.003, this.noise * 6);
    if (!warming && this.pendingUntil === null && this.frames >= this.cooldownUntil) {
      if (rms > threshold) {
        this.pendingUntil = this.frames + Math.round(sampleRate);
        this.port.postMessage({type: 'onset'});
      } else this.noise = .995 * this.noise + .005 * rms;
    }
    if (this.pendingUntil !== null && this.frames >= this.pendingUntil) {
      const pcm = new Float32Array(this.ring.length);
      pcm.set(this.ring.subarray(this.cursor)); pcm.set(this.ring.subarray(0, this.cursor), this.ring.length - this.cursor);
      this.port.postMessage({type: 'candidate', pcm, startFrame:this.frames-this.ring.length}, [pcm.buffer]);
      this.pendingUntil = null; this.cooldownUntil = this.frames + Math.round(sampleRate * .4);
    }
    if (this.frames - this.lastLevel >= sampleRate / 10) {
      this.port.postMessage({type: 'level', peak: this.peak, warming}); this.peak = 0; this.lastLevel = this.frames;
    }
    return true;
  }
}
registerProcessor('gesture-preview', GesturePreview);
