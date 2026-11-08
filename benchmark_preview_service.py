"""Paired decode+inference benchmark; excludes capture/HTTP and never fits."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audio_format import decode_wav
from benchmark_fast_preview import timing_summary
from preview_model import predict as original
from preview_service import PreviewService

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostics/performance/service'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[]
    for sid in ('20261103T165139_cf7201','20261103T200050_87dcb0'):
        record=json.loads((ROOT/'data'/sid/'session.json').read_text())
        paths.extend(ROOT/'data'/sid/c['filename'] for c in record['clips'] if c['accepted'])
    inputs=[(str(path.relative_to(ROOT)),path.read_bytes()) for path in paths]
    service=PreviewService()
    plan={'scope':'Decode plus inference, excluding browser/capture/HTTP. Original Python-list decoder/reference matcher versus deployed array decoder/batched matcher.',
          'inputs':{name:hashlib.sha256(blob).hexdigest() for name,blob in inputs},
          'repeats':5,'seed':339,'engine':json.loads((ROOT/'diagnostics/preview/engine.json').read_text()),
          'model_sha256':hashlib.sha256((ROOT/'diagnostics/preview/model.npz').read_bytes()).hexdigest(),
          'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def baseline(blob):
        samples,rate,_=decode_wav(blob)
        return original(samples,rate,service.model)
    def accelerated(blob):
        result=service.classify(blob)
        return {key:value for key,value in result.items() if key not in ('processing_ms','audio_saved')}
    for _,blob in inputs[:3]:baseline(blob);accelerated(blob)
    times={'baseline':[],'accelerated':[]}
    rows=[]
    rng=np.random.default_rng(339)
    for repeat in range(5):
        for index in rng.permutation(len(inputs)):
            name,blob=inputs[index]
            answers={};row={'file':name,'repeat':repeat}
            for mode in rng.permutation(['baseline','accelerated']):
                start=time.perf_counter()
                answers[mode]=(baseline if mode=='baseline' else accelerated)(blob)
                elapsed=time.perf_counter()-start
                times[mode].append(elapsed);row[mode+'_ms']=elapsed*1000
            if answers['baseline']!=answers['accelerated']:
                raise AssertionError(f'Changed response: {name}')
            rows.append(row)
    summary={name:timing_summary(values) for name,values in times.items()}
    report={'scope':plan['scope'],'unique_inputs':len(inputs),'paired_runs':len(rows),'exact_outputs':True,
            'summary':summary,'median_speedup':summary['baseline']['p50_ms']/summary['accelerated']['p50_ms'],
            'plan_sha256':hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),'timings':rows}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='timings'},indent=2))


if __name__=='__main__':main()
