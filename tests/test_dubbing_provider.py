"""Exercise the real subprocess protocol using a tiny fake GPU/TTS runtime."""
from pathlib import Path
import sys
import tempfile
import unittest
from app.core.dubbing import DubbingError, IndexTTS2Provider

class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.repo=self.root/'repo'; self.repo.mkdir()
        (self.repo/'indextts').mkdir(); (self.repo/'indextts/__init__.py').write_text('')
        (self.repo/'torch.py').write_text('class Cuda:\n def is_available(self): return True\n def get_device_name(self, n): return "Fake GPU"\ncuda=Cuda()\nclass Version: hip=None\nversion=Version()\n')
        (self.repo/'indextts/infer_v2.py').write_text('''import wave
from pathlib import Path
class IndexTTS2:
 def __init__(self, **kwargs):
  with open('starts.txt', 'a') as f: f.write('start\\n')
  print('An ordinary model diagnostic, not JSON')
 def infer(self, spk_audio_prompt, text, output_path, verbose=False):
  print('Synthesizing')
  with wave.open(output_path, 'wb') as w:
   w.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
   w.writeframes(b'\\0\\0'*2400)
''')
        self.models=self.root/'models'; self.models.mkdir(); (self.models/'config.yaml').write_text('fake: true')
        self.provider=IndexTTS2Provider(sys.executable,str(self.repo),str(self.models),timeout=5)
    def tearDown(self): self.provider.close(); self.temp.cleanup()
    def test_persistent_worker(self):
        self.provider.synthesize('你好','unused.wav',str(self.root/'one.wav'))
        self.provider.synthesize('再见','unused.wav',str(self.root/'two.wav'))
        self.assertEqual((self.repo/'starts.txt').read_text(),'start\n')
        self.assertTrue((self.root/'two.wav').exists())
    def test_gpu_refusal(self):
        (self.repo/'torch.py').write_text('class Cuda:\n def is_available(self): return False\ncuda=Cuda()\n')
        with self.assertRaisesRegex(DubbingError,'CPU fallback is disabled'):
            self.provider.synthesize('你好','unused.wav',str(self.root/'one.wav'))
    def test_model_fingerprint_changes(self):
        first=self.provider.fingerprint()
        (self.models/'weights.bin').write_bytes(b'new model')
        other=IndexTTS2Provider(sys.executable,str(self.repo),str(self.models))
        self.assertNotEqual(first,other.fingerprint())

if __name__=='__main__': unittest.main()

class QwenProviderTests(unittest.TestCase):
    def test_qwen_worker_reuses_voice_prompt(self):
        from app.core.dubbing.provider import Qwen3TTSProvider
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=root/'repo';repo.mkdir();models=root/'models';models.mkdir()
            (models/'config.json').write_text('{}')
            (repo/'torch.py').write_text('class Cuda:\n def is_available(self):return True\n def get_device_name(self,n):return "Fake"\ncuda=Cuda()\nclass Version:hip=None\nversion=Version()\nfloat16="float16"\n')
            (repo/'qwen_tts').mkdir()
            (repo/'qwen_tts/__init__.py').write_text('''class Qwen3TTSModel:
 @classmethod
 def from_pretrained(cls,*a,**k):return cls()
 def create_voice_clone_prompt(self,**k):
  assert k['ref_text']=='Reference words' and k['x_vector_only_mode'] is False
  with open('prompts.txt','a') as f:f.write('once\\n')
  return ['prompt']
 def generate_voice_clone(self,**k):
  assert k['language']=='Chinese' and k['voice_clone_prompt']==['prompt']
  return [[0]*2400],24000
''')
            (repo/'soundfile.py').write_text('''import wave
def write(path,data,rate,subtype=None):
 assert subtype=='PCM_16'
 with wave.open(path,'wb') as w:
  w.setparams((1,2,rate,0,'NONE','not compressed'));w.writeframes(b'\\0\\0'*len(data))
''')
            provider=Qwen3TTSProvider(sys.executable,str(repo),str(models),timeout=5)
            provider.reference_text='Reference words'
            try:
                provider.synthesize('第一句','ref.wav',str(root/'a.wav'))
                provider.synthesize('第二句','ref.wav',str(root/'b.wav'))
                self.assertEqual((repo/'prompts.txt').read_text(),'once\n')
                self.assertTrue((root/'b.wav').exists())
            finally:provider.close()
