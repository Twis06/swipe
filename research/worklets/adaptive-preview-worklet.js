// Experimental endpointing; not served by the live app until replay passes.
// Real two-second history only: no zero fill, sample scaling, or resampling.
class GesturePreview extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring = new Float32Array(Math.round(sampleRate * 2));
    this.cursor = 0; this.frames = 0; this.onsetFrame = null; this.cooldownUntil = 0;
    this.noise = .0001; this.warmup = Math.round(sampleRate); this.lastLevel = 0;
    this.quietFrames = 0; this.offThreshold = .0015;
    this.cancelled = false; this.peak = 0;
    this.port.onmessage = ({data}) => {if (data.type === 'stop') {this.cancelled = true; this.onsetFrame = null;}};
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
    if (!warming && this.onsetFrame === null && this.frames >= this.cooldownUntil) {
      if (rms > threshold) {
        this.onsetFrame = this.frames; this.quietFrames = 0; this.offThreshold = threshold * .5;
        this.port.postMessage({type: 'onset'});
      } else this.noise = .995 * this.noise + .005 * rms;
    }
    if (this.onsetFrame !== null) {
      this.quietFrames = rms < this.offThreshold ? this.quietFrames + channel.length : 0;
      const elapsed = this.frames - this.onsetFrame;
      const quiet = elapsed >= Math.round(sampleRate * .2) && this.quietFrames >= Math.round(sampleRate * .1);
      const timeout = elapsed >= Math.round(sampleRate);
      if (this.frames >= this.ring.length && (quiet || timeout)) {
        const pcm = new Float32Array(this.ring.length);
        pcm.set(this.ring.subarray(this.cursor)); pcm.set(this.ring.subarray(0, this.cursor), this.ring.length - this.cursor);
        this.port.postMessage({type: 'candidate', pcm, startFrame: this.frames-this.ring.length,
                              onsetFrame: this.onsetFrame, emitFrame: this.frames,
                              endpointReason: timeout ? 'maximum_wait' : 'quiet'}, [pcm.buffer]);
        this.onsetFrame = null; this.cooldownUntil = this.frames + Math.round(sampleRate * .4);
      }
    }
    if (this.frames - this.lastLevel >= sampleRate / 10) {
      this.port.postMessage({type: 'level', peak: this.peak, warming}); this.peak = 0; this.lastLevel = this.frames;
    }
    return true;
  }
}
registerProcessor('gesture-preview', GesturePreview);
