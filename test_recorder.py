import io
import json
import tempfile
import unittest
import wave
import zipfile
import struct
from pathlib import Path
from types import SimpleNamespace
import server

def wav(seconds=2):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes(b'\x00\x10' * int(seconds * 48000))
    return buf.getvalue()

def float_wav(samples, rate=48000):
    pcm=struct.pack('<'+'f'*len(samples),*samples)
    fmt=struct.pack('<HHIIHH',3,1,rate,rate*4,4,32)
    chunks=b'fmt '+struct.pack('<I',len(fmt))+fmt+b'data'+struct.pack('<I',len(pcm))+pcm
    return b'RIFF'+struct.pack('<I',len(chunks)+4)+b'WAVE'+chunks

class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original = server.DATA
        server.DATA = Path(self.temp.name)

    def tearDown(self):
        server.DATA = self.original
        self.temp.cleanup()

    def request(self, path, value=None, token=server.TOKEN):
        obj = object.__new__(server.Handler)
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        obj.path = path
        obj.server = SimpleNamespace(server_port=8765)
        obj.headers = {'Host':'localhost:8765','X-Recorder-Token':token,'Content-Length':str(len(body))}
        obj.rfile = io.BytesIO(body)
        result = []
        obj.send = lambda status, body, mime=None: result.extend([status,body])
        obj.do_GET() if value is None else obj.do_POST()
        return result

    def test_save_retry_reject_and_export(self):
        status, session = self.request('/api/session', {'plan':[{'label':'up','block':1}]})
        self.assertEqual(status,201)
        path = f'/api/clip/{session["id"]}/0'
        status, record = self.request(path,wav())
        self.assertEqual(status,201)
        self.assertEqual(record['clips'][0]['frames'],96000)
        self.assertAlmostEqual(record['clips'][0]['rms'],.125)
        _, retry = self.request(path,wav())
        self.assertEqual(len(retry['clips']),1)
        self.request(f'/api/reject/{session["id"]}/0',{})
        _, redone = self.request(path,wav())
        self.assertEqual([c['accepted'] for c in redone['clips']],[False,True])
        _, archive = self.request(f'/api/export/{session["id"]}')
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            self.assertEqual(len(z.namelist()),3)
            saved = json.loads(z.read(f'{session["id"]}/session.json'))
            self.assertEqual(len([c for c in saved['clips'] if c['accepted']]),1)

    def test_reject_invalid_audio_and_token(self):
        self.assertEqual(self.request('/api/session',{},token='wrong')[0],403)
        _, record = self.request('/api/session',{'plan':[{'label':'down','block':1}]})
        self.assertEqual(self.request(f'/api/clip/{record["id"]}/0',wav(.1))[0],400)
        self.assertFalse(list(server.DATA.glob('*/*.wav')))

    def test_reject_bad_labels_and_index(self):
        self.assertEqual(self.request('/api/session',{'plan':[{'label':'fake','block':1}]})[0],400)
        _, record = self.request('/api/session',{'plan':[{'label':'idle','block':1}]})
        self.assertEqual(self.request(f'/api/clip/{record["id"]}/20',wav())[0],400)

    def test_flat_runs_flag_possible_clipping_but_pairs_do_not(self):
        pairs=float_wav([1,1,.125,-1,-1,-.125]*16000)
        flats=float_wav([1,1,1,.125,-1,-1,-1,-.125]*12000)
        self.assertFalse(server.wav_info(pairs)['possible_clipping'])
        clipped=server.wav_info(flats)
        self.assertTrue(clipped['possible_clipping'])
        self.assertEqual(clipped['level_assessment'],'possible_clipping')
        self.assertEqual(clipped['quality_check_version'],2)
        _, record=self.request('/api/session',{'plan':[{'label':'up','block':1}]})
        _, saved=self.request(f'/api/clip/{record["id"]}/0',flats)
        self.assertTrue(saved['clips'][0]['possible_clipping'])
        self.assertEqual((server.DATA/record['id']/saved['clips'][0]['filename']).read_bytes(),flats)

    def test_float_wav_preserves_over_range_and_reports_quality(self):
        rate=48000
        samples=[1.5,-1.5,.125,-.125]*(rate//2)
        blob=float_wav(samples,rate)
        info=server.wav_info(blob)
        self.assertEqual(info['encoding'],'float32')
        self.assertEqual(info['peak'],1.5)
        self.assertEqual(info['high_level_fraction'],.5)
        self.assertTrue(info['quality_warning'])
        self.assertFalse(info['possible_clipping'],'Preserved over-range samples alone are not clipping evidence')
        _, record=self.request('/api/session',{'plan':[{'label':'up','block':1}]})
        status,saved=self.request(f'/api/clip/{record["id"]}/0',blob)
        self.assertEqual(status,201)
        self.assertEqual((server.DATA/record['id']/saved['clips'][0]['filename']).read_bytes(),blob)

if __name__ == '__main__':
    unittest.main()
