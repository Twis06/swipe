"""Fixed adaptive-endpoint replay vs live capture; no training or live edits."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from collections import Counter
import numpy as np
from swipe.preview_model import predict
from swipe.preview_service import PreviewService

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/capture_latency'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[ROOT/'data'/sid/'session.json' for sid in ['20261103T165139_cf7201','20261103T200050_87dcb0']+json.loads((ROOT/'diagnostics/continuous/label_review.json').read_text())['take_ids']]
    for session_path in list(inputs):
        s=json.loads(session_path.read_text())
        inputs.extend(session_path.parent/c['filename'] for c in s['clips'] if c['accepted'])
    protected=inputs+[ROOT/'research/worklets/preview-worklet.js',ROOT/'web/preview.js',ROOT/'swipe/preview_model.py',ROOT/'swipe/gesture_model.py',ROOT/'models/preview/model.npz',ROOT/'models/preview/model.json',ROOT/'diagnostics/continuous/segmentation/report.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    plan={'status':'Fixed one-protocol comparison before audio replay; reused data, no parameter selection.',
          'parameters':{'minimum_post_onset_seconds':.2,'quiet_seconds':.1,'off_threshold_on_fraction':.5,'maximum_post_onset_seconds':1,'cooldown_seconds':.4,'real_history_seconds':2,'warmup_seconds':1},
          'isolated_protocol':'Identical synthetic1.2s quiet lead and1.5s tail on both worklets; existing training clips, no fresh accuracy.',
          'continuous_protocol':'Unmodified three15s files, no lead/tail; unknown true directions.',
          'duplicate_suppression':'Executes exact overlapsPreviousEvent function extracted from current preview.js per replay record.',
          'limits':['No HTTP latency/drop simulation.','Delay from detected onset, not measured physical onset.','Same labels/features previously used during development.','Faster endpoint can move/merge event content in two-second windows.'],
          'protected_sha256':hashes,'source_sha256':{n:sha(ROOT/n) for n in ['research/worklets/adaptive-preview-worklet.js','research/compare_capture_latency.py','research/compare_capture_latency.cjs']}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    service=PreviewService()
    with tempfile.TemporaryDirectory(prefix='grille-capture-compare-') as tmp:
        subprocess.run(['node',str(ROOT/'research/compare_capture_latency.cjs'),tmp],check=True)
        rows=json.loads((Path(tmp)/'windows.json').read_text())
        for row in rows:
            for window in row['windows']:
                pcm=np.fromfile(Path(tmp)/window.pop('file'),dtype='<f4')
                window['result']=predict(pcm,row['rate'],service.model)
        (Path(tmp)/'classified.json').write_text(json.dumps(rows))
        subprocess.run(['node',str(ROOT/'research/compare_capture_latency.cjs'),tmp,'suppress'],check=True)
        rows=json.loads((Path(tmp)/'classified.json').read_text())
    segmentation=json.loads((ROOT/'diagnostics/continuous/segmentation/report.json').read_text())
    baselines={r['session']:r['methods']['baseline_6']['intervals_seconds'] for r in segmentation['records'] if r['kind']=='continuous'}
    summary={}
    for variant in ('research/worklets/preview-worklet.js','research/worklets/adaptive-preview-worklet.js'):
        subset=[r for r in rows if r['variant']==variant]
        isolated=[r for r in subset if r['kind']=='isolated']
        continuous=[r for r in subset if r['kind']=='continuous']
        multi=0;reasons=Counter()
        for row in continuous:
            for window in row['windows']:
                a,b=window['start_frame']/row['rate'],window['end_frame']/row['rate']
                overlaps=sum(min(b,y)-max(a,x)>=.04 for x,y in baselines[row['session']])
                window['baseline_burst_overlap_count']=overlaps
                multi+=overlaps>1;reasons[window['result']['reason']]+=1
        swipes=[r for r in isolated if r['truth'] in ('up','down')]
        negatives=[r for r in isolated if r['truth'] not in ('up','down')]
        shown=lambda r:[w for w in r['windows'] if not w['duplicate_suppressed'] and w['result']['prediction']!='background']
        accepted=lambda r:[w for w in shown(r) if w['result']['prediction'] in ('up','down')]
        latency=[w['delay_seconds'] for r in swipes for w in r['windows']]
        first_latency=[r['windows'][0]['delay_seconds'] for r in swipes if r['windows']]
        summary[variant]={'isolated':{'swipe_clips':len(swipes),'swipes_with_candidate':sum(bool(r['windows']) for r in swipes),
            'total_candidate_windows':sum(len(r['windows']) for r in swipes),'accepted_windows':sum(len(accepted(r)) for r in swipes),
            'correct_accepted_windows':sum(w['result']['prediction']==r['truth'] for r in swipes for w in accepted(r)),
            'clips_with_any_wrong_accepted':sum(any(w['result']['prediction']!=r['truth'] for w in accepted(r)) for r in swipes),
            'clips_with_multiple_displayed':sum(len(shown(r))>1 for r in swipes),'negative_clips':len(negatives),
            'negative_clips_with_accepted':sum(bool(accepted(r)) for r in negatives),
            'candidate_onset_delay_median_ms':float(np.median(latency)*1000),
            'candidate_onset_delay_p95_ms':float(np.quantile(latency,.95)*1000),
            'first_candidate_delay_median_ms':float(np.median(first_latency)*1000)},
            'continuous':[{'session':r['session'],'candidate_windows':len(r['windows']),'displayed_results':len(shown(r)),
                           'accepted_directions':len(accepted(r)),
                           'candidate_delay_median_ms':float(np.median([w['delay_seconds'] for w in r['windows']])*1000) if r['windows'] else None}
                          for r in continuous],
            'continuous_total':{'windows_overlapping_multiple_activity_bursts':multi,'candidate_windows':sum(len(r['windows']) for r in continuous),'reason_counts':dict(reasons)}}
    baseline={r['session']+'/'+r['file']:r for r in rows if r['variant']=='research/worklets/preview-worklet.js' and r['kind']=='isolated'}
    changed=[]
    for row in rows:
        if row['variant']!='research/worklets/adaptive-preview-worklet.js' or row['kind']!='isolated':continue
        old=baseline[row['session']+'/'+row['file']]
        display=lambda r:[w['result']['prediction'] for w in r['windows'] if not w['duplicate_suppressed'] and w['result']['prediction']!='background']
        if display(old)!=display(row):changed.append({'session':row['session'],'file':row['file'],'truth':row['truth'],'baseline':display(old),'adaptive':display(row)})
    unchanged=all(sha(ROOT/p)==digest for p,digest in hashes.items())
    assert unchanged
    report={'status':'Experimental replay only; prototype not deployed. No training.', 'summary':summary,'changed_isolated_results':changed,'records':rows,'protected_unchanged':unchanged,'limits':plan['limits']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'summary':summary,'changed_isolated_results':changed,'protected_unchanged':unchanged},indent=2))

if __name__=='__main__':main()
