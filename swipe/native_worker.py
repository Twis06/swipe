"""Private stdin/stdout inference bridge for the Swift app. No ports or recordings.

Input: uint32 little-endian length followed by a two-second mono WAV.
Output: one JSON line per message. The first line announces readiness.
"""
import json
import struct
import sys
from swipe.preview_service import PreviewService

MAX_BYTES = 2_000_000

def read_exact(stream, count):
    chunks = bytearray()
    while len(chunks) < count:
        part = stream.read(count - len(chunks))
        if not part:
            if not chunks:
                return None
            raise ValueError('Incomplete native audio message.')
        chunks.extend(part)
    return bytes(chunks)

def serve(source, destination, service):
    def emit(value):
        destination.write(json.dumps(value, allow_nan=False) + '\n')
        destination.flush()
    emit({'ready': True, 'recognizer': service.info()['recognizer']})
    while True:
        header = read_exact(source, 4)
        if header is None:
            return
        size, = struct.unpack('<I', header)
        if not 44 <= size <= MAX_BYTES:
            raise ValueError('Invalid native message size.')
        blob = read_exact(source, size)
        if blob is None:
            raise ValueError('Missing native audio message.')
        try:
            emit(service.classify(blob))
        except (ValueError, BlockingIOError) as exc:
            emit({'error': str(exc)})

def main():
    try:
        serve(sys.stdin.buffer, sys.stdout, PreviewService())
    except (ValueError, ImportError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
