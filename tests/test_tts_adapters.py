"""API-contract tests with fake upstream models, not GPU/voice qualification."""
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from app.core.dubbing.adapter_standard import StandardAdapter
from app.core.dubbing.provider import create_provider


class AdapterContracts(unittest.TestCase):
    def test_qwen_variants_and_design_reuse_fixed_voice(self):
        calls = []
        class Model:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                assert kwargs['local_files_only'] is True
                calls.append(('load',path)); return cls()
            def generate_custom_voice(self, **kwargs):
                calls.append(('custom', kwargs)); return [[0.0] * 240], 24000
            def generate_voice_design(self, **kwargs):
                calls.append(('design', kwargs)); return [[0.0] * 240], 24000
            def create_voice_clone_prompt(self, **kwargs):
                calls.append(('prompt',kwargs)); return ['fixed']
            def generate_voice_clone(self, **kwargs):
                calls.append(('clone',kwargs)); return [[0.0] * 240], 24000
        def write(path,audio,rate,**kwargs):
            import wave
            with wave.open(str(path),'wb') as wav:
                wav.setparams((1,2,rate,0,'NONE','not compressed'));wav.writeframes(b'\0\0'*len(audio))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {
            'soundfile': types.SimpleNamespace(write=write),
            'torch': types.SimpleNamespace(float16='fp16',cuda=types.SimpleNamespace(empty_cache=lambda:None)),
            'qwen_tts': types.SimpleNamespace(Qwen3TTSModel=Model)}):
            (Path(folder)/'voice_clone_base').mkdir()
            (Path(folder)/'voice_clone_base/config.json').write_text('{}')
            for engine in ('qwen3-custom', 'qwen3-design', 'qwen3-design'):
                adapter = StandardAdapter(engine, folder, str(Path(folder)/'source'))
                adapter.synthesize(dict(text='你好', language='中文', voice='Vivian', voice_identity='a'*64,
                    description='沉稳清晰的声音', output_path=str(Path(folder)/f'{engine}.wav')))
            custom = next(value for name,value in calls if name=='custom')
            self.assertEqual(custom['speaker'], 'Vivian')
            self.assertNotIn('ref_audio', custom)
            designs = [value for name,value in calls if name=='design']
            self.assertEqual(len(designs),1)  # second worker reuses persisted reference
            self.assertEqual(designs[0]['instruct'], '沉稳清晰的声音')
            self.assertNotEqual(designs[0]['text'],'你好')
            self.assertGreater(len(designs[0]['text']),30)
            self.assertEqual(len([value for name,value in calls if name=='clone']),2)
            self.assertEqual(len([value for name,value in calls if name=='prompt']),2)

    def test_dots_preserves_model_sampling_defaults(self):
        calls = []
        class Runtime:
            @classmethod
            def from_pretrained(cls, path, **kw):
                calls.append(kw); return cls()
            def generate(self, **kw):
                calls.append(kw); return dict(audio=[0.0]*240, sample_rate=24000)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {
            'soundfile': types.SimpleNamespace(write=lambda *a, **k: None),
            'torch':types.SimpleNamespace(float16='fp16'),
            'dots_tts.runtime':types.SimpleNamespace(DotsTtsRuntime=Runtime)}):
            adapter = StandardAdapter('dots',folder,folder)
            adapter.synthesize(dict(text='你好',reference_audio='ref.wav',reference_text='原文',
                language='中文',output_path=str(Path(folder)/'out.wav')))
            self.assertEqual(calls[1]['prompt_audio_path'],'ref.wav')
            self.assertEqual(calls[1]['language'],'zh')
            self.assertNotIn('num_steps',calls[1])
            self.assertNotIn('guidance_scale',calls[1])
            with self.assertRaisesRegex(ValueError, '参考音频'):
                adapter.synthesize(dict(text='你好',output_path=str(Path(folder)/'bad.wav')))

    def test_edge_is_reference_free_and_really_decodes_mp3(self):
        import wave
        calls=[]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); mp3=root/'tone.mp3'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=0.1',str(mp3)],check=True)
            data=mp3.read_bytes()
            class Communicate:
                def __init__(self, **kw): calls.append(kw)
                async def save(self,path): Path(path).write_bytes(data)
            async def list_voices():
                return [dict(ShortName='zh-CN-XiaoxiaoNeural',Locale='zh-CN')]
            with patch.dict(sys.modules,{'edge_tts':types.SimpleNamespace(Communicate=Communicate,list_voices=list_voices)}):
                adapter=StandardAdapter('edge-tts',folder,folder)
                target=root/'out.wav'
                adapter.synthesize(dict(text='测试',voice='zh-CN-XiaoxiaoNeural',
                    reference_audio='PRIVATE.wav',reference_text='PRIVATE',output_path=str(target)))
                self.assertEqual(calls,[dict(text='测试',voice='zh-CN-XiaoxiaoNeural')])
                with self.assertRaisesRegex(ValueError, '语言不一致'):
                    adapter.synthesize(dict(text='Hello',voice='zh-CN-XiaoxiaoNeural',language='English',output_path=str(root/'bad.wav')))
                with wave.open(str(target)) as wav:
                    self.assertEqual(wav.getframerate(),24000)
                    self.assertEqual(wav.getnchannels(),1)
                    self.assertGreater(wav.getnframes(),0)

    def test_edge_provider_does_not_require_local_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            provider=create_provider('edge-tts',sys.executable,str(Path(folder)/'absent-source'),
                str(Path(folder)/'absent-models'),voice='zh-CN-XiaoxiaoNeural')
            self.assertEqual(provider.engine,'edge-tts')
            self.assertEqual(provider.fingerprint()['options']['voice'],'zh-CN-XiaoxiaoNeural')

    def test_provider_fingerprint_includes_options(self):
        with tempfile.TemporaryDirectory() as folder:
            a=create_provider('edge-tts',sys.executable,folder,folder,voice='one')
            b=create_provider('edge-tts',sys.executable,folder,folder,voice='two')
            self.assertNotEqual(a.fingerprint(),b.fingerprint())

    def test_fingerprint_cancellation_and_invalid_manifest_refusal(self):
        import threading
        from app.core.dubbing.pipeline import CancelledError, DubbingError
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'weights.bin').write_bytes(b'x'*32)
            cancel=threading.Event()
            provider=create_provider('edge-tts',sys.executable,folder,folder,cancel_event=cancel)
            cancel.set()
            with self.assertRaises(CancelledError): provider.fingerprint()
            (root/'videocaptioner-models.json').write_text('{}')
            with patch('app.core.tts_runtime.verify_models',return_value=False):
                with self.assertRaisesRegex(DubbingError,'校验失败'):
                    create_provider('qwen3-0.6b',sys.executable,folder,folder)

    def test_worker_imports_adapters_with_real_pth_isolation(self):
        import json
        import shutil
        import sysconfig
        from app.core.dubbing import worker
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            python=root/('python.exe' if sys.platform=='win32' else 'python')
            shutil.copy2(sys.executable,python)
            stdlib=Path(sysconfig.get_path('stdlib'))
            (root/'python._pth').write_text(str(stdlib)+'\n'+str(stdlib/'lib-dynload')+'\n')
            check=subprocess.run([str(python),'-c','import sys; print(sys.flags.isolated)'],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if check.returncode:
                self.skipTest('Copied Python cannot start without platform runtime DLLs')
            self.assertEqual(check.stdout.strip(),'1')
            repo=root/'repo'; repo.mkdir(); (repo/'indextts').mkdir()
            (repo/'indextts/__init__.py').write_text('')
            (repo/'torch.py').write_text('class Cuda:\n def is_available(self):return True\n def get_device_name(self,n):return "Fake"\ncuda=Cuda()\nclass V:hip=None\nversion=V()\n')
            (repo/'indextts/infer_v2.py').write_text('class IndexTTS2:\n def __init__(self,**kw):pass\n def infer(self,**kw):pass\n')
            run=subprocess.run([str(python),str(Path(worker.__file__).resolve()),'--repo',str(repo),
                '--models',str(root),'--engine','indextts2'],input='',stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,text=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stderr+run.stdout)
            self.assertTrue(json.loads(run.stdout.split('DUBBING_JSON:')[1])['ready'])

    def test_design_fingerprint_binds_reference_and_rejects_midjob_damage(self):
        import hashlib
        import json
        from app.core.dubbing.pipeline import DubbingError
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); repo=root/'source'; models=root/'models'
            (repo/'qwen_tts').mkdir(parents=True); (repo/'qwen_tts/__init__.py').write_text('')
            models.mkdir(); (models/'config.json').write_text('{}')
            def make():
                return create_provider('qwen3-design',sys.executable,str(repo),str(models),description='test')
            first=make(); fresh=first.fingerprint()
            another=make(); self.assertNotEqual(fresh,another.fingerprint())
            cache=root/'voice-design-cache'; cache.mkdir()
            data=b'fixed-reference'; digest=hashlib.sha256(data).hexdigest()
            (cache/(first._voice_identity+'.wav')).write_bytes(data)
            (cache/(first._voice_identity+'.json')).write_text(json.dumps(dict(text='reference',sha256=digest)))
            bound=make(); self.assertEqual(bound.fingerprint()['design_reference'],digest)
            (cache/(first._voice_identity+'.wav')).write_bytes(b'damaged')
            changed=make(); self.assertNotEqual(changed.fingerprint(),bound.fingerprint())
            bound.process=types.SimpleNamespace(poll=lambda:None)
            with patch.object(bound,'close'):
                with self.assertRaisesRegex(DubbingError,'删除或损坏'):
                    bound.synthesize('test',None,str(root/'out.wav'))


if __name__ == '__main__': unittest.main()
