import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from urllib.parse import quote
import zipfile

from swipe import server
from tests.test_recorder import float_wav


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.original=server.DATA;server.DATA=Path(self.temp.name)
        self.blob=float_wav([1.5,-1.5,.125,-.125]*180000)
        self.payload={'mode':'cued','displayed_cues':[{'index':i,'label':'up' if i%2==0 else 'down','audio_seconds':i*1.5+.02} for i in range(10)],'metadata':{'notes':'continuous test'}}

    def tearDown(self):
        server.DATA=self.original;self.temp.cleanup()

    def request(self,path,body=None,payload=None,token=server.TOKEN):
        handler=object.__new__(server.Handler);handler.path=path;handler.server=SimpleNamespace(server_port=8765)
        handler.headers={'Host':'localhost:8765','Content-Length':str(len(body or b'')),'X-Recorder-Token':token,'X-Recording-Metadata':quote(json.dumps(payload if payload is not None else self.payload))}
        handler.rfile=io.BytesIO(body or b'');out=[];handler.send=lambda status,value,mime=None:out.extend([status,value])
        handler.do_GET() if body is None else handler.do_POST()
        return out

    def test_full_take_retry_export_and_raw_float_preservation(self):
        route='/api/continuous/'+'a'*32
        status,record=self.request(route,self.blob)
        self.assertEqual(status,201);self.assertEqual(record['kind'],'continuous');self.assertEqual(record['protocol_version'],3)
        clip=record['clips'][0];self.assertEqual(clip['frames'],720000);self.assertEqual(clip['duration_seconds'],15);self.assertEqual(clip['peak'],1.5)
        saved=server.DATA/record['id']/'continuous.wav';self.assertEqual(saved.read_bytes(),self.blob)
        mtime=saved.stat().st_mtime_ns
        _,again=self.request(route,self.blob);self.assertEqual(again,record);self.assertEqual(saved.stat().st_mtime_ns,mtime)
        _,archive=self.request('/api/export/'+record['id'])
        with zipfile.ZipFile(io.BytesIO(archive)) as z:self.assertEqual(len(z.namelist()),2)
        changed={**self.payload,'metadata':{'notes':'different'}}
        self.assertEqual(self.request(route,self.blob,changed)[0],400)
        self.assertEqual(saved.read_bytes(),self.blob)

    def test_bad_duration_prompt_and_token_are_rejected(self):
        route='/api/continuous/'+'b'*32
        self.assertEqual(self.request(route,self.blob,token='wrong')[0],403)
        self.assertEqual(self.request(route,float_wav([0]*96000))[0],400)
        wrong={**self.payload,'displayed_cues':[{'index':0,'label':'down','audio_seconds':0}]}
        self.assertEqual(self.request(route,self.blob,wrong)[0],400)
        self.assertEqual(list(server.DATA.iterdir()),[])

    def test_free_take_has_no_direction_labels(self):
        _,record=self.request('/api/continuous/'+'c'*32,self.blob,{'mode':'free','displayed_cues':[],'metadata':{}})
        self.assertEqual(record['planned_cues'],[]);self.assertEqual(record['displayed_cues'],[])
        self.assertEqual(record['clips'][0]['label'],'continuous')


if __name__=='__main__':unittest.main()
