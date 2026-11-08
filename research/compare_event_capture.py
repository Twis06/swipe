"""One follow-up to failed historical-window endpointing. No training/live edits."""
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
OUT=ROOT/'diagnostics/capture_latency/event_isolated_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def shown(row):return [w for w in row['windows'] if not w['duplicate_suppressed'] and w['result']['prediction']!='background']
def accepted(row):return [w for w in shown(row) if w['result']['prediction'] in ('up','down')]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    oldplan=json.loads((ROOT/'diagnostics/capture_latency/plan.json').read_text())
    hashes=oldplan['protected_sha256']
    assert all(sha(ROOT/p)==digest for p,digest in hashes.items())
    plan={'status':'Single fixed follow-up motivated by v1 historical-window contamination; parameters chosen before v2 replay.',
          'parameters':{'source_pre_roll_seconds':.1,'minimum_post_onset_seconds':.2,'quiet_seconds':.1,'off_threshold_on_fraction':.5,'maximum_post_onset_seconds':1,'cooldown_seconds':.4,'warmup_seconds':1},
          'representation':'Source event only, centered unchanged into two-second zeros; startFrame maps synthetic positions back to actual source time.',
          'isolated_leads':['1.2s synthetic zeros','1.2s repeating originalclip first250ms, synthetic background control'],
          'isolated_tail':'Identical1.5s zeros on both algorithms. First250ms is not independently verified quiet.',
          'continuous':'Original15s files, no lead or tail.',
          'limits':['Source gating has unverified physical boundaries.','Synthetic zeros change background subtraction and gate calibration.','Recorded clips were already used for development/training; no fresh accuracy.','Repeat-background and zero-lead protocols are synthetic, not measured real-user latency.','No network/request drops modeled.'],
          'protected_sha256':hashes,'source_sha256':{n:sha(ROOT/n) for n in ['research/worklets/event-preview-worklet.js','research/compare_event_capture.py','research/compare_event_capture.cjs']}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    service=PreviewService()
    with tempfile.TemporaryDirectory(prefix='grille-event-capture-') as tmp:
        subprocess.run(['node',str(ROOT/'research/compare_event_capture.cjs'),tmp],check=True)
        rows=json.loads((Path(tmp)/'windows.json').read_text())
        for row in rows:
            for window in row['windows']:
                pcm=np.fromfile(Path(tmp)/window.pop('file'),dtype='<f4')
                window['result']=predict(pcm,row['rate'],service.model)
        (Path(tmp)/'classified.json').write_text(json.dumps(rows))
        subprocess.run(['node',str(ROOT/'research/compare_event_capture.cjs'),tmp,'suppress'],check=True)
        rows=json.loads((Path(tmp)/'classified.json').read_text())
    segmentation=json.loads((ROOT/'diagnostics/continuous/segmentation/report.json').read_text())
    baselines={r['session']:r['methods']['baseline_6']['intervals_seconds'] for r in segmentation['records'] if r['kind']=='continuous'}
    summary={}
    for lead in ['zero','repeated_recorded_background','none']:
        summary[lead]={}
        for variant in ['research/worklets/preview-worklet.js','research/worklets/event-preview-worklet.js']:
            subset=[r for r in rows if r['lead']==lead and r['variant']==variant]
            if lead=='none':
                result=[]
                for row in subset:
                    multi=0;reasons=Counter()
                    for w in row['windows']:
                        a,b=w['source_start_frame']/row['rate'],w['source_end_frame']/row['rate']
                        overlaps=sum(min(b,y)-max(a,x)>=.04 for x,y in baselines[row['session']])
                        w['baseline_burst_overlap_count']=overlaps;multi+=overlaps>1;reasons[w['result']['reason']]+=1
                    delays=[w['delay_seconds']*1000 for w in row['windows']]
                    result.append({'session':row['session'],'candidate_windows':len(row['windows']),'displayed_results':len(shown(row)),
                                   'accepted_directions':len(accepted(row)),'source_overlapping_multiple_activity_bursts':multi,
                                   'candidate_delay_median_ms':float(np.median(delays)) if delays else None,
                                   'candidate_delay_p95_ms':float(np.quantile(delays,.95)) if delays else None,'reason_counts':dict(reasons)})
                summary[lead][variant]=result
            else:
                swipes=[r for r in subset if r['truth'] in ('up','down')];neg=[r for r in subset if r['truth'] not in ('up','down')]
                delay=[w['delay_seconds']*1000 for r in swipes for w in r['windows']]
                first=[r['windows'][0]['delay_seconds']*1000 for r in swipes if r['windows']]
                summary[lead][variant]={'swipe_clips':len(swipes),'swipes_with_candidate':sum(bool(r['windows']) for r in swipes),
                  'accepted_windows':sum(len(accepted(r)) for r in swipes),'correct_accepted_windows':sum(w['result']['prediction']==r['truth'] for r in swipes for w in accepted(r)),
                  'clips_with_wrong_accepted':sum(any(w['result']['prediction']!=r['truth'] for w in accepted(r)) for r in swipes),
                  'clips_with_multiple_displayed':sum(len(shown(r))>1 for r in swipes),'negative_clips':len(neg),'negative_clips_with_accepted':sum(bool(accepted(r)) for r in neg),
                  'candidate_delay_median_ms':float(np.median(delay)),'candidate_delay_p95_ms':float(np.quantile(delay,.95)),
                  'first_candidate_delay_median_ms':float(np.median(first))}
    changed=[]
    baseline={(r['session'],r['file'],r['lead']):r for r in rows if r['variant']=='research/worklets/preview-worklet.js' and r['kind']=='isolated'}
    for row in rows:
        if row['variant']!='research/worklets/event-preview-worklet.js' or row['kind']!='isolated':continue
        old=baseline[(row['session'],row['file'],row['lead'])]
        labels=lambda r:[w['result']['prediction'] for w in shown(r)]
        if labels(old)!=labels(row):changed.append({'session':row['session'],'file':row['file'],'lead':row['lead'],'truth':row['truth'],'baseline':labels(old),'event_isolated':labels(row)})
    unchanged=all(sha(ROOT/p)==digest for p,digest in hashes.items());assert unchanged
    report={'status':'Fixed event-isolation replay; not deployed, no training.', 'summary':summary,'changed_isolated_results':changed,'records':rows,'protected_unchanged':unchanged,'limits':plan['limits']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'summary':summary,'changed_isolated_results':changed,'protected_unchanged':unchanged},indent=2))

if __name__=='__main__':main()
