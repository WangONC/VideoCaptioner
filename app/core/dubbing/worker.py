"""Run with the dedicated IndexTTS2 environment, never the application's Python.

Official API reference: https://github.com/index-tts/index-tts#python-api
IndexTTS-2 uses indextts.infer_v2.IndexTTS2 (not infer_v2_5).
No model installation/download is performed by this worker.
"""
import argparse
import contextlib
import json
import os
import socket
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--engine',choices=['indextts2','qwen3'],default='indextts2')
    args = parser.parse_args()
    protocol = sys.stdout
    def emit(value):
        protocol.write('DUBBING_JSON:' + json.dumps(value, ensure_ascii=False) + '\n')
        protocol.flush()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        # Upstream auxiliary loaders may use multiple download libraries. Offline
        # flags alone are insufficient; refuse IP connections in this worker.
        original_connect = socket.socket.connect
        def offline_connect(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                raise RuntimeError('Offline dubbing: a required auxiliary model is missing. Install all model assets before running.')
            return original_connect(sock, address)
        socket.socket.connect = offline_connect
        socket.socket.connect_ex = offline_connect
        sys.path.insert(0, str(Path(args.repo).resolve()))
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            # PyTorch exposes ROCm GPUs through torch.cuda too. Unsupported AMD
            # runtimes fail explicitly instead of silently performing CPU inference.
            if not torch.cuda.is_available():
                raise RuntimeError('No supported GPU runtime detected. CUDA or compatible ROCm PyTorch is required; CPU fallback is disabled.')
            models = Path(args.models).resolve()
            prompt_cache={}
            if args.engine=='qwen3':
                from qwen_tts import Qwen3TTSModel
                model=Qwen3TTSModel.from_pretrained(str(models),device_map='cuda:0',dtype=torch.float16,attn_implementation='sdpa',local_files_only=True)
            else:
                from indextts.infer_v2 import IndexTTS2
                models = Path(args.models).resolve()
                auxiliary=models/'hf_cache'
                aux_paths=dict(w2v_bert=str(auxiliary/'w2v-bert-2.0'),
                    semantic_codec=str(auxiliary/'semantic_codec_model.safetensors'),
                    campplus=str(auxiliary/'campplus_cn_common.bin'),bigvgan=str(auxiliary/'bigvgan'))
                model = IndexTTS2(cfg_path=str(models / 'config.yaml'), model_dir=str(models),
                                  use_fp16=True, device='cuda', use_cuda_kernel=False, use_deepspeed=False,
                                  use_qwen_emo=False, aux_paths=aux_paths)
        emit({'ready': True, 'backend': 'rocm' if torch.version.hip else 'cuda',
              'device': torch.cuda.get_device_name(0)})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    if args.engine=='qwen3':
                        import soundfile as sf
                        reference=request['reference_audio'];text=request.get('reference_text') or None
                        key=(reference,text)
                        if key not in prompt_cache:
                            prompt_cache[key]=model.create_voice_clone_prompt(ref_audio=reference,ref_text=text,x_vector_only_mode=text is None)
                        wavs,rate=model.generate_voice_clone(text=request['text'],language='Chinese',voice_clone_prompt=prompt_cache[key])
                        if len(wavs)!=1:raise RuntimeError('Unexpected TTS output count')
                        sf.write(request['output_path'],wavs[0],rate,subtype='PCM_16')
                    else:
                        model.infer(spk_audio_prompt=request['reference_audio'], text=request['text'],
                                    output_path=request['output_path'], verbose=False)
                emit({'ok': True})
            except Exception as exc:
                emit({'error': str(exc)})
    except Exception as exc:
        emit({'error': str(exc)})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
