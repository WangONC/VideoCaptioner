"""Small, dependency-free GPU admission probe shared by setup and inference.

A successful tensor probe is not certification of any model or driver. In
particular, invalid HIP kernels must remain errors, never silent CPU fallback.
"""
import json
import platform
import subprocess


def _drivers():
    if platform.system() != 'Windows':
        return []
    try:
        result = subprocess.run(['powershell', '-NoProfile', '-Command',
            'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress'],
            capture_output=True, text=True, timeout=15, creationflags=0x08000000)
        value = json.loads(result.stdout)
        return value if isinstance(value, list) else [value]
    except Exception:
        return []


def select_gpu(torch_module=None):
    """Select and probe a discrete GPU, retaining its current-device binding.

    ROCm admission is intentionally limited to this application's RX 9070
    recipe, not an assertion that other AMD devices are intrinsically unusable.
    No kernels are launched on unselected/unsupported integrated devices.
    """
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    hip = getattr(torch.version, 'hip', None)
    report = dict(torch=str(torch.__version__), hip=hip,
                  cuda=getattr(torch.version, 'cuda', None),
                  backend='rocm' if hip else 'cuda', devices=[], drivers=_drivers())
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('没有可用 GPU 运行时；不会回退 CPU')
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            arch = str(getattr(prop, 'gcnArchName', ''))
            name = str(prop.name)
            supported = not hip or (arch.split(':')[0] == 'gfx1201' and '9070' in name)
            report['devices'].append(dict(index=index, name=name, architecture=arch,
                memory_bytes=int(getattr(prop, 'total_memory', 0)), supported=supported))
        candidates = [d for d in report['devices'] if d['supported']]
        if not candidates:
            raise RuntimeError('未找到当前配方支持的独立 GPU（AMD 配方要求 RX 9070 / gfx1201）')
        chosen = max(candidates, key=lambda d: d['memory_bytes'])
        index = chosen['index']
        report.update(index=index, device='cuda:%d' % index, name=chosen['name'],
                      architecture=chosen['architecture'])
        torch.cuda.set_device(index)
        # Validate execution, not merely enumeration/import. On failure stop;
        # an asynchronous GPU fault can poison this process's device context.
        x = torch.ones(8, device=report['device'])
        result = (x + x).sum().item()
        torch.cuda.synchronize(index)
        if result != 16:
            raise RuntimeError('GPU 张量结果校验失败')
        report['probe'] = 'passed'
        return report
    except Exception as exc:
        report['error'] = str(exc)
        raise RuntimeError('GPU 检查失败；未回退 CPU。请核对所选设备、架构、PyTorch/HIP 与驱动配方。\n'
                           + json.dumps(report, ensure_ascii=False)) from exc


def select_device(requested='gpu', torch_module=None):
    """Explicit CPU admission; never called as recovery from a GPU failure."""
    if requested == 'gpu':
        return select_gpu(torch_module)
    if requested != 'cpu':
        raise ValueError('未知运行设备：' + str(requested))
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    x = torch.ones(8, device='cpu')
    if (x + x).sum().item() != 16:
        raise RuntimeError('CPU 张量结果校验失败')
    return dict(backend='cpu', device='cpu', name='CPU', index=None,
                architecture=platform.machine(), torch=str(torch.__version__),
                hip=getattr(torch.version, 'hip', None), devices=[], probe='passed')
