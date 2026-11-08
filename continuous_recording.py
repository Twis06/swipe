"""Save a 15-second continuous take with prompt timing, without segment labels."""
import hashlib
import json
import math
from pathlib import Path

from audio_format import recording_info


def save_take(root, recording_id, blob, payload, timestamp):
    info=recording_info(blob,duration_seconds=15)
    if info['encoding']!='float32':
        raise ValueError('Continuous takes must preserve Float32 samples.')
    mode=payload.get('mode')
    if mode not in ('cued','free'):
        raise ValueError('Choose cued or free alternating movement.')
    displayed=payload.get('displayed_cues',[])
    if not isinstance(displayed,list) or len(displayed)>10 or (mode=='free' and displayed):
        raise ValueError('Invalid prompt timeline.')
    previous=-1
    for cue in displayed:
        index=cue.get('index');actual=cue.get('audio_seconds')
        if (type(index) is not int or not previous<index<10 or cue.get('label')!=('up' if index%2==0 else 'down')
                or not isinstance(actual,(int,float)) or not math.isfinite(actual) or not 0<=actual<=15):
            raise ValueError('Invalid displayed prompt.')
        previous=index
    metadata=payload.get('metadata',{})
    if not isinstance(metadata,dict):raise ValueError('Invalid recording metadata.')
    sid='continuous_'+recording_id
    folder=Path(root)/sid
    digest=hashlib.sha256(blob).hexdigest()
    payload_hash=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    record_path=folder/'session.json'
    if record_path.exists():
        existing=json.loads(record_path.read_text())
        if existing['clips'][0]['sha256']!=digest or existing.get('payload_sha256')!=payload_hash:
            raise ValueError('Recording ID already belongs to a different take.')
        return existing
    planned=[{'index':i,'label':'up' if i%2==0 else 'down','start_seconds':i*1.5,'end_seconds':(i+1)*1.5} for i in range(10)] if mode=='cued' else []
    record={'id':sid,'created_at':timestamp,'protocol_version':3,'kind':'continuous','mode':mode,
            'metadata':metadata,'payload_sha256':payload_hash,
            'label_status':'Prompt intentions only; actual gesture boundaries and directions require review.',
            'planned_cues':planned,'displayed_cues':displayed,
            'timing_note':'Displayed cue times estimate AudioContext time relative to the worklet first sample; not measured finger onset.',
            'plan':[{'label':'continuous','block':1,'pace':'continuous','duration_seconds':15}],
            'clips':[{'filename':'continuous.wav','trial_index':0,'label':'continuous','block':1,'pace':'continuous',
                      'accepted':True,'recorded_at':timestamp,'sha256':digest,**info}]}
    folder.mkdir(parents=True,exist_ok=True)
    temporary=folder/'continuous.wav.tmp';temporary.write_bytes(blob);temporary.replace(folder/'continuous.wav')
    temporary=folder/'session.json.tmp';temporary.write_text(json.dumps(record,indent=2));temporary.replace(record_path)
    return record
