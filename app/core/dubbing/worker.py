"""Isolated persistent TTS worker. Local engines never connect to the network.

Edge TTS is explicitly online and sends only the requested text/voice.
"""
import argparse
import contextlib
import json
import os
import socket
from pathlib import Path
import sys

ENGINES = ('indextts2', 'qwen3', 'qwen3-custom', 'qwen3-design', 'dots', 'cosyvoice3', 'gpt-sovits', 'edge-tts')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--engine', choices=ENGINES, default='indextts2')
    args = parser.parse_args()
    protocol = sys.stdout
    def emit(value):
        protocol.write('DUBBING_JSON:' + json.dumps(value, ensure_ascii=False) + '\n')
        protocol.flush()
    try:
        if args.engine != 'edge-tts':
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
            original_connect = socket.socket.connect
            def offline_connect(sock, address):
                if sock.family in (socket.AF_INET, socket.AF_INET6):
                    raise RuntimeError('Offline dubbing: required model assets are missing; download them before running.')
                return original_connect(sock, address)
            socket.socket.connect = offline_connect
            socket.socket.connect_ex = offline_connect
        if args.engine == 'cosyvoice3':
            for key in ('onnx_path', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE'):
                os.environ.pop(key, None)
        # Windows embeddable Python's ._pth omits the executed script directory.
        # Explicitly expose only our worker's sibling adapters.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        repo = Path(args.repo).resolve()
        if args.engine in ('indextts2', 'gpt-sovits'):
            os.environ['NLTK_DATA'] = str(repo / '.nltk_data')
        sys.path[:0] = [str(repo), str(repo/'src')]
        with contextlib.redirect_stdout(sys.stderr):
            if args.engine != 'edge-tts':
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('No supported GPU runtime detected. CUDA or compatible ROCm PyTorch is required; CPU fallback is disabled.')
            if args.engine in ('cosyvoice3', 'gpt-sovits'):
                from adapter_cosy_gpt import create_adapter
                adapter = create_adapter(args.engine, args.models, args.repo)
            else:
                from adapter_standard import StandardAdapter
                adapter = StandardAdapter(args.engine, args.models, args.repo)
        info = {'ready': True, 'backend': 'online', 'device': None}
        if args.engine != 'edge-tts':
            info.update(backend='rocm' if torch.version.hip else 'cuda', device=torch.cuda.get_device_name(0))
        emit(info)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    adapter.synthesize(request)
                emit({'ok': True})
            except Exception as exc:
                emit({'error': str(exc)})
    except Exception as exc:
        emit({'error': str(exc)})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
