class Capture extends AudioWorkletProcessor {
  constructor() {
    super(); this.left = 0; this.parts = []; this.id = null; this.levelFrames=0;this.peak=0;
    this.previous=null;this.flatRun=0;this.flatSamples=0;
    this.timing=false;this.captured=0;this.processedFrames=0;
    this.port.onmessage = ({data}) => {
      if (data.type === 'capture') { this.left = data.frames; this.parts = []; this.id = data.id;this.timing=!!data.timing;this.captured=0; }
      if (data.type === 'cancel') { this.left = 0; this.parts = []; }
    };
  }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if(channel) {
      for(const x of channel) {
        this.peak=Math.max(this.peak,Math.abs(x));
        this.flatRun=Math.abs(x)>=.98 ? (x===this.previous ? this.flatRun+1 : 1) : 0;
        if(this.flatRun===3) this.flatSamples+=3;
        else if(this.flatRun>3) this.flatSamples++;
        this.previous=x;
      }
      this.levelFrames+=channel.length;
      if(this.levelFrames>=sampleRate*.1) {
        this.port.postMessage({type:'level',peak:this.peak,flatTopFraction:Math.min(1,this.flatSamples/this.levelFrames)});
        this.peak=0;this.levelFrames=0;this.flatSamples=0;
      }
    }
    if (channel && this.left > 0) {
      if(this.timing && this.captured===0)this.port.postMessage({type:'started',id:this.id,audioTime:(typeof currentFrame==='number'?currentFrame:this.processedFrames)/sampleRate});
      const count = Math.min(channel.length, this.left);
      this.parts.push(channel.slice(0, count)); this.left -= count;this.captured+=count;
      if (this.left === 0) {
        const pcm = new Float32Array(this.parts.reduce((n, p) => n + p.length, 0));
        let offset = 0; for (const part of this.parts) { pcm.set(part, offset); offset += part.length; }
        this.port.postMessage({id: this.id, pcm}, [pcm.buffer]); this.parts = [];
      }
    }
    this.processedFrames+=channel?.length || 128;
    return true;
  }
}
registerProcessor('capture', Capture);
