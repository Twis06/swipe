#!/usr/bin/env python3
"""Loopback-only recorder backend; standard library, no uploads to external services."""
import argparse
import hashlib
import io
import json
import re
import secrets
import struct
import threading
import wave
import zipfile
from swipe.audio_format import recording_info
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
from swipe.continuous_recording import save_take
from swipe.paths import ROOT, WEB_DIR
from swipe.key_bindings import KeyBindings

DATA = ROOT / 'data'
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
PREVIEW_LOCK = threading.Lock()
PREVIEW = None
KEYS = KeyBindings()
LABELS = {'up', 'down', 'idle', 'typing'}

def preview():
    global PREVIEW
    with PREVIEW_LOCK:
        if PREVIEW is None:
            try:
                from swipe.preview_service import PreviewService
                PREVIEW = PreviewService()
            except (ImportError, FileNotFoundError, ValueError) as exc:
                raise RuntimeError(f'Preview unavailable: {exc}. Start with .venv/bin/python -m swipe. If verification fails, restore or verify the model and engine files before restarting; retraining is not required for an engine repair.') from exc
    return PREVIEW

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)

def wav_info(blob):
    return recording_info(blob)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith('GET /api/state'):
            return
        super().log_message(fmt, *args)

    def send(self, status, body, mime='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def local(self):
        return self.headers.get('Host') in {f'localhost:{self.server.server_port}', f'127.0.0.1:{self.server.server_port}'}

    def do_GET(self):
        if not self.local():
            return self.send(403, {'error': 'Local access only.'})
        path = self.path.split('?')[0]
        files = {'/': 'index.html', '/app.js': 'app.js', '/audio-worklet.js': 'audio-worklet.js', '/style.css': 'style.css',
                 '/preview': 'preview.html', '/preview.js': 'preview.js', '/preview.css': 'preview.css', '/preview-worklet.js': 'cascade-preview-worklet.js',
                 '/continuous':'continuous.html','/continuous.js':'continuous.js', '/key-bindings.js':'key-bindings.js'}
        if path in files:
            name = files[path]
            mime = 'text/html' if name.endswith('html') else 'text/css' if name.endswith('css') else 'text/javascript'
            return self.send(200, (WEB_DIR / name).read_bytes(), mime)
        if path == '/api/state':
            sessions = []
            for file in sorted(DATA.glob('*/session.json'), reverse=True):
                sessions.append(json.loads(file.read_text()))
            return self.send(200, {'token': TOKEN, 'sessions': sessions, 'data_directory': str(DATA)})
        if path == '/api/preview':
            try:
                return self.send(200, {**preview().info(), 'token': TOKEN, 'system_actions': KEYS.status()['enabled']})
            except RuntimeError as exc:
                return self.send(503, {'error': str(exc)})
        if path == '/api/keys':
            return self.send(200, KEYS.status(self.headers.get('X-Swipe-Client', '')))
        match = re.fullmatch(r'/api/export/([a-zA-Z0-9_-]+)', path)
        if match:
            folder = DATA / match[1]
            if not (folder / 'session.json').exists():
                return self.send(404, {'error': 'Session not found.'})
            buf = io.BytesIO()
            with LOCK, zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
                for file in folder.iterdir():
                    if file.suffix in {'.json', '.wav'}:
                        z.write(file, f'{folder.name}/{file.name}')
            return self.send(200, buf.getvalue(), 'application/zip')
        return self.send(404, {'error': 'Not found.'})

    def do_POST(self):
        if not self.local() or self.headers.get('X-Recorder-Token') != TOKEN:
            return self.send(403, {'error': 'Reload the collector to reconnect.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            continuous=re.fullmatch(r'/api/continuous/([a-f0-9]{32})',self.path)
            if not 0 < length < (12_000_000 if continuous else 2_000_000):
                raise ValueError('Invalid request size.')
            body = self.rfile.read(length)
            if self.path == '/api/keys':
                payload = json.loads(body)
                try:
                    return self.send(200, KEYS.configure(payload['action'], payload['client'], payload.get('bindings')))
                except PermissionError as exc:
                    return self.send(403, {'error': str(exc)})
            if self.path == '/api/keys/fire':
                payload = json.loads(body)
                try:
                    return self.send(200, KEYS.fire(payload['ticket'], payload['client']))
                except PermissionError as exc:
                    return self.send(403, {'error': str(exc)})
                except RuntimeError as exc:
                    return self.send(503, {'error': str(exc)})
            if continuous:
                metadata=self.headers.get('X-Recording-Metadata','')
                if len(metadata)>16000:raise ValueError('Recording metadata is too large.')
                payload=json.loads(unquote(metadata))
                if not isinstance(payload,dict):raise ValueError('Invalid recording metadata.')
                with LOCK:
                    return self.send(201,save_take(DATA,continuous[1],body,payload,now()))
            if self.path == '/api/predict':
                try:
                    result = preview().classify(body)
                    ticket = KEYS.offer(result, self.headers.get('X-Swipe-Client', ''))
                    if ticket:
                        result = {**result, 'action_ticket': ticket}
                    return self.send(200, result)
                except RuntimeError as exc:
                    return self.send(503, {'error': str(exc)})
                except BlockingIOError as exc:
                    return self.send(429, {'error': str(exc)})
            with LOCK:
                if self.path == '/api/session':
                    payload = json.loads(body)
                    plan = payload['plan']
                    if not isinstance(plan, list) or not 1 <= len(plan) <= 200:
                        raise ValueError('Invalid recording plan.')
                    for trial in plan:
                        if trial['label'] not in LABELS or trial['block'] not in (1, 2):
                            raise ValueError('Invalid trial label or block.')
                    sid = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + secrets.token_hex(3)
                    folder = DATA / sid
                    folder.mkdir(parents=True)
                    record = {'id': sid, 'created_at': now(), 'protocol_version': 2, 'model': 'MacBook Pro 14-inch M3 Pro', 'metadata': payload.get('metadata', {}), 'plan': plan, 'clips': []}
                    atomic_json(folder / 'session.json', record)
                    return self.send(201, record)
                match = re.fullmatch(r'/api/(clip|reject)/([a-zA-Z0-9_-]+)/(\d+)', self.path)
                if not match:
                    return self.send(404, {'error': 'Not found.'})
                action, sid, index = match[1], match[2], int(match[3])
                folder = DATA / sid
                record = json.loads((folder / 'session.json').read_text())
                if not 0 <= index < len(record['plan']):
                    raise ValueError('Unknown trial.')
                accepted = [c for c in record['clips'] if c['trial_index'] == index and c['accepted']]
                if action == 'reject':
                    for clip in accepted:
                        clip['accepted'] = False
                        clip['rejected_at'] = now()
                    atomic_json(folder / 'session.json', record)
                    return self.send(200, record)
                if accepted:
                    return self.send(200, record)  # Safe retry if a previous response was lost.
                info = wav_info(body)
                trial = record['plan'][index]
                attempt = sum(c['trial_index'] == index for c in record['clips']) + 1
                filename = f'{index + 1:03d}_b{trial["block"]}_{trial["label"]}_take{attempt}.wav'
                temp = folder / (filename + '.tmp')
                temp.write_bytes(body)
                temp.replace(folder / filename)
                record['clips'].append({'trial_index': index, **trial, **info, 'filename': filename, 'accepted': True, 'recorded_at': now(), 'sha256': hashlib.sha256(body).hexdigest()})
                atomic_json(folder / 'session.json', record)
                return self.send(201, record)
        except (ValueError, KeyError, TypeError, wave.Error, EOFError) as e:
            return self.send(400, {'error': str(e)})
        except FileNotFoundError:
            return self.send(404, {'error': 'Session not found.'})
        except OSError as e:
            return self.send(500, {'error': f'Could not save recording: {e}'})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Swipe: http://localhost:{args.port}\nRecordings: {DATA}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
