// Experimental early isolated candidate with original real-history fallback.
// Only {type:'accepted',id} for the current early candidate cancels fallback.
class GesturePreview extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring=new Float32Array(Math.round(sampleRate*2));this.cursor=0;this.frames=0;
    this.noise=.0001;this.warmup=Math.round(sampleRate);this.lastLevel=0;this.peak=0;
    this.serial=0;this.id=null;this.onsetFrame=null;this.cooldownUntil=0;
    this.quietFrames=0;this.offThreshold=.0015;this.capture=null;this.captured=0;this.sourceStart=0;this.earlySent=false;this.cancelled=false;
    this.port.onmessage=({data})=>{
      if(data.type==='stop'){this.cancelled=true;this.clear();}
      else if(data.type==='accepted'&&this.id===data.id&&this.earlySent){this.clear();this.cooldownUntil=this.frames+Math.round(sampleRate*.4);}
    };
  }
  clear(){this.id=null;this.onsetFrame=null;this.capture=null;this.captured=0;this.earlySent=false;this.quietFrames=0;}
  process(inputs){
    if(this.cancelled)return false;
    const channel=inputs[0]?.[0];if(!channel)return true;
    let power=0;for(const value of channel){this.ring[this.cursor]=value;this.cursor=(this.cursor+1)%this.ring.length;power+=value*value;this.peak=Math.max(this.peak,Math.abs(value));}
    this.frames+=channel.length;
    const rms=Math.sqrt(power/channel.length),warming=this.frames<this.warmup;
    if(warming)this.noise=.98*this.noise+.02*rms;
    const threshold=Math.max(.003,this.noise*6);let triggered=false;
    if(!warming&&this.id===null&&this.frames>=this.cooldownUntil){
      if(rms>threshold){
        this.id=++this.serial;this.onsetFrame=this.frames;this.quietFrames=0;this.offThreshold=threshold*.5;this.earlySent=false;
        const preroll=Math.round(sampleRate*.1);this.sourceStart=this.frames-preroll;
        this.capture=new Float32Array(Math.round(sampleRate*1.2)+256);
        for(let i=0;i<preroll;i++)this.capture[i]=this.ring[(this.cursor-preroll+i+this.ring.length)%this.ring.length];
        this.captured=preroll;triggered=true;this.port.postMessage({type:'onset',id:this.id});
      }else this.noise=.995*this.noise+.005*rms;
    }
    if(this.id!==null){
      if(!triggered&&!this.earlySent){this.capture.set(channel,this.captured);this.captured+=channel.length;}
      this.quietFrames=rms<this.offThreshold?this.quietFrames+channel.length:0;
      const elapsed=this.frames-this.onsetFrame,timeout=elapsed>=Math.round(sampleRate);
      const quiet=elapsed>=Math.round(sampleRate*.2)&&this.quietFrames>=Math.round(sampleRate*.1);
      if(timeout){
        const pcm=new Float32Array(this.ring.length);pcm.set(this.ring.subarray(this.cursor));pcm.set(this.ring.subarray(0,this.cursor),this.ring.length-this.cursor);
        this.port.postMessage({type:'candidate',id:this.id,phase:'fallback',pcm,startFrame:this.frames-this.ring.length,
          onsetFrame:this.onsetFrame,emitFrame:this.frames,sourceStartFrame:this.frames-this.ring.length,sourceEndFrame:this.frames,
          sourceSamples:pcm.length,destinationOffset:0},[pcm.buffer]);
        this.clear();this.cooldownUntil=this.frames+Math.round(sampleRate*.4);
      }else if(quiet&&!this.earlySent){
        const pcm=new Float32Array(this.ring.length),offset=Math.floor((pcm.length-this.captured)/2);pcm.set(this.capture.subarray(0,this.captured),offset);
        this.port.postMessage({type:'candidate',id:this.id,phase:'early',pcm,startFrame:this.sourceStart-offset,
          onsetFrame:this.onsetFrame,emitFrame:this.frames,sourceStartFrame:this.sourceStart,sourceEndFrame:this.frames,
          sourceSamples:this.captured,destinationOffset:offset},[pcm.buffer]);
        this.earlySent=true;this.capture=null;
      }
    }
    if(this.frames-this.lastLevel>=sampleRate/10){this.port.postMessage({type:'level',peak:this.peak,warming});this.peak=0;this.lastLevel=this.frames;}
    return true;
  }
}
registerProcessor('gesture-preview',GesturePreview);
