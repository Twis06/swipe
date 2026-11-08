"""Integration diagnostic on reused clips, explicitly not an accuracy estimate."""
import json
from pathlib import Path
import subprocess
import tempfile
from preview_service import PreviewService

ROOT=Path(__file__).resolve().parent
service=PreviewService()
with tempfile.TemporaryDirectory(prefix='grille-preview-replay-') as directory:
    subprocess.run(['node',str(ROOT/'replay_preview.cjs'),directory],check=True)
    rows=json.loads((Path(directory)/'windows.json').read_text())
    for row in rows:
        row['results']=[];last_end=float('-inf')
        for window in row.pop('windows'):
            result=service.classify((Path(directory)/window['file']).read_bytes())
            duplicate=False
            if result['prediction']!='background':
                start=window['start_seconds']+result['quality']['event_start_seconds']
                end=window['start_seconds']+result['quality']['event_end_seconds']
                duplicate=start<=last_end+.1
                if not duplicate:last_end=end
            row['results'].append({**result,'duplicate_suppressed':duplicate})
summary={}
for sid in sorted({r['session'] for r in rows}):
    subset=[r for r in rows if r['session']==sid]
    swipes=[r for r in subset if r['truth'] in ('up','down')]
    negatives=[r for r in subset if r['truth'] not in ('up','down')]
    summary[sid]={'swipes':len(swipes),'swipes_with_a_candidate':sum(bool(r['results']) for r in swipes),
                  'swipes_with_multiple_candidates':sum(len(r['results'])>1 for r in swipes),
                  'background_clips':len(negatives),'background_false_activations':sum(any(p['prediction'] in ('up','down') for p in r['results']) for r in negatives),
                  'swipes_with_multiple_displayed_results':sum(sum(p['prediction']!='background' and not p['duplicate_suppressed'] for p in r['results'])>1 for r in swipes),
                  'accepted_swipe_windows':sum(p['prediction'] in ('up','down') and not p['duplicate_suppressed'] for r in swipes for p in r['results']),
                  'correct_accepted_swipe_windows':sum(p['prediction']==r['truth'] and not p['duplicate_suppressed'] for r in swipes for p in r['results'])}
report={'status':'Integration replay using training clips; not independent accuracy. Synthetic quiet lead-in/tail; real continuous use unvalidated.',
        'summary':summary,'clips':rows}
(ROOT/'diagnostics/preview/replay.json').write_text(json.dumps(report,indent=2))
print(json.dumps(summary,indent=2))
