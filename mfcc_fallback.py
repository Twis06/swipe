"""MFCC-first candidate; consult legacy engine only for uncertain results."""
from mfcc_preview import predict as primary_predict
from selective_preview import predict as legacy_predict

def predict(samples,rate,model,raw_model,aligned_model):
    primary=primary_predict(samples,rate,model)
    if primary['prediction']!='uncertain':return primary
    fallback=legacy_predict(samples,rate,raw_model,aligned_model)
    if fallback['prediction'] in ('up','down'):
        return {**fallback,'recognition_path':'legacy_fallback'}
    return primary
