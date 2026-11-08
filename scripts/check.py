#!/usr/bin/env python3
"""Run the portable public suite; never fit models or read private recordings."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON_TESTS = [
    'test_public_runtime.PublicRuntimeTests',
    'test_public_runtime.PublicEndpointTests',
    'test_continuous',
    'test_preview_engine',
    'test_mfcc_components',
    'test_swipe_feature_families',
    'test_encoding_alignment',
    'test_fast_audio.FastAudioTests.test_invalid_layout_and_nonfinite_are_rejected_both_ways',
    'test_fast_audio.FastAudioTests.test_headroom_and_signed_zero_are_preserved',
    'test_fast_preview.FastPreviewTests.test_random_shapes_radii_and_unreachable_paths_are_exact',
    'test_fast_preview.FastPreviewTests.test_ties_and_near_ties_keep_reference_order',
    'test_fast_preview.FastPreviewTests.test_input_contract_and_empty_reference_bank',
    'test_mfcc_preview.MFCCPreviewTests.test_fallback_routes_only_uncertain_and_only_accepts_direction',
    'test_mfcc_preview.MFCCPreviewTests.test_candidate_silence_and_input_preservation',
    'test_mfcc_preview.MFCCPreviewTests.test_changed_model_is_refused',
]
JAVASCRIPT_TESTS = [
    'test_audio.cjs',
    'test_continuous.cjs',
    'test_microphone_guard.cjs',
    'test_cascade_capture.cjs',
    'test_preview_cascade.cjs',
]


def main():
    node = shutil.which('node')
    if node is None:
        raise SystemExit('Node.js 22+ is required for capture/client checks. Install it and retry.')
    major = int(subprocess.check_output([node, '--version'], text=True).strip().lstrip('v').split('.')[0])
    if major < 22:
        raise SystemExit('Node.js 22+ is required.')
    subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', *PYTHON_TESTS], cwd=ROOT, check=True)
    for test in JAVASCRIPT_TESTS:
        subprocess.run([node, test], cwd=ROOT, check=True)
    print('Public checks passed: no private dataset, microphone access, or model fitting required.')


if __name__ == '__main__':
    main()
