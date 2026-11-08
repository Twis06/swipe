#!/usr/bin/env python3
"""Inspect files eligible for Git publication and verify bundled runtime banks.

This is a targeted artifact/hygiene check, not a comprehensive secret scanner.
Both tracked files and nonignored untracked files are inspected, so accidental
force-adds of ignored recordings are caught too. It does not stage or publish.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = {
    'diagnostics/preview/engine.json',
    'diagnostics/preview/model.json',
    'diagnostics/preview/model.npz',
    'diagnostics/aligned_preview/model.json',
    'diagnostics/aligned_preview/model.npz',
    'diagnostics/mfcc_preview/model.json',
    'diagnostics/mfcc_preview/model.npz',
    'diagnostics/mfcc_preview/benchmark.json',
    'diagnostics/mfcc_preview/benchmark_plan.json',
    'diagnostics/mfcc_preview/fallback_checks/plan.json',
    'diagnostics/mfcc_preview/fallback_checks/report.json',
    'diagnostics/mfcc_preview/replay_fallback/plan.json',
    'diagnostics/mfcc_preview/replay_fallback/report.json',
}
PRIVATE_PARTS = {'data', 'checkpoints', '.venv', 'venv', '__pycache__',
                 '.impeccable', '.codex', '.agents', 'node_modules'}
PRIVATE_SUFFIXES = {'.wav', '.mp3', '.m4a', '.zip', '.gz', '.tar', '.pyc', '.pem', '.key'}
SUSPICIOUS = re.compile(
    r'/(?:Users|home)/[A-Za-z0-9_.-]+/'
    r'|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'
    r'|\bghp_[A-Za-z0-9]{30,}'
    r'|\bgithub_pat_[A-Za-z0-9_]{30,}'
    r'|\bAKIA[A-Z0-9]{16}\b'
)


def main():
    try:
        output = subprocess.check_output(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise SystemExit('Run this check inside a Git checkout (or initialize one with git init).') from exc
    names = sorted(set(output.decode().rstrip('\0').split('\0')) - {''})
    errors = []
    for name in names:
        relative = Path(name)
        path = ROOT / relative
        if (PRIVATE_PARTS.intersection(relative.parts) or name.startswith('docs/local/')
                or relative.suffix.lower() in PRIVATE_SUFFIXES or relative.name.startswith('.env')
                or relative.name == '.DS_Store'):
            errors.append(f'Private/local artifact eligible for publication: {name}')
        if name.startswith('diagnostics/') and name not in DIAGNOSTICS:
            errors.append(f'Unreviewed diagnostic artifact: {name}')
        if relative.suffix == '.npz' and name not in DIAGNOSTICS:
            errors.append(f'Unreviewed binary bank: {name}')
        if path.is_symlink():
            errors.append(f'Symlink needs explicit review: {name}')
            continue
        if not path.is_file():
            errors.append(f'Missing staged/tracked file: {name}')
            continue
        if path.stat().st_size > 5_000_000:
            errors.append(f'Unexpected large file: {name}')
        if path.suffix != '.npz':
            try:
                content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                errors.append(f'Unreviewed binary file: {name}')
            else:
                if SUSPICIOUS.search(content):
                    errors.append(f'Personal path or credential pattern; inspect {name}')
    required = DIAGNOSTICS | {'README.md', 'LICENSE', 'requirements.txt',
                              'server.py', 'preview_service.py', '.github/workflows/ci.yml'}
    errors.extend(f'Missing release file: {name}' for name in sorted(required - set(names)))
    if errors:
        raise SystemExit('\n'.join(errors))
    sys.path.insert(0, str(ROOT))
    from preview_service import PreviewService
    service = PreviewService()  # Verifies bank and implementation hashes.
    if service.info()['recognizer'] != 'mfcc8_fallback':
        raise SystemExit('The release baseline must use mfcc8_fallback.')
    print(f'Release check passed: {len(names)} eligible files; private artifacts excluded; all runtime banks verified.')


if __name__ == '__main__':
    main()
