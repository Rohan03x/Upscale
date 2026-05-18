import sys, os

path = '/workspace/Real-ESRGAN/inference_realesrgan_video.py'
with open(path) as f:
    src = f.read()

# Fix 1: CPU deepcopy for ONNX export (never mutate live model dtype)
OLD1 = """                if not os.path.exists(_onnx_path):
                    print(f'[opt] Exporting ONNX model \u2192 {_onnx_path}')
                    # Export as float32 \u2014 TRT will optimise to fp16 internally
                    upsampler.model.eval().float()
                    _dummy = torch.zeros(
                        _bs, 3, _H_pad, _W_pad,
                        device=upsampler.device)          # float32
                    with torch.no_grad():
                        torch.onnx.export(
                            upsampler.model, _dummy, _onnx_path,
                            input_names=['input'], output_names=['output'],
                            opset_version=17,
                            dynamic_axes={
                                'input':  {0: 'batch'},
                                'output': {0: 'batch'},
                            },
                        )
                    upsampler.model.half()    # restore fp16 for PyTorch fallback path
                    print('[opt] ONNX export done (float32 graph, TRT fp16 internally)')"""

NEW1 = """                if not os.path.exists(_onnx_path):
                    print(f'[opt] Exporting ONNX model \u2192 {_onnx_path}')
                    # CPU deepcopy: live model dtype is never mutated
                    import copy as _cp
                    _export_model = _cp.deepcopy(upsampler.model).float().eval().cpu()
                    _dummy = torch.zeros(_bs, 3, _H_pad, _W_pad)  # float32, CPU
                    with torch.no_grad():
                        torch.onnx.export(
                            _export_model, _dummy, _onnx_path,
                            input_names=['input'], output_names=['output'],
                            opset_version=17,
                            dynamic_axes={
                                'input':  {0: 'batch'},
                                'output': {0: 'batch'},
                            },
                        )
                    del _export_model
                    torch.cuda.empty_cache()
                    print('[opt] ONNX export done (float32 graph, TRT fp16 internally)')"""

if OLD1 not in src:
    print('FIX1 MISSING - searching for anchor...')
    idx = src.find('upsampler.model.eval().float()')
    if idx >= 0:
        print(repr(src[max(0,idx-300):idx+300]))
    else:
        print('anchor string not found at all')
    sys.exit(1)
src = src.replace(OLD1, NEW1, 1)
print('Fix 1 OK')

# Fix 2: reduce TRT workspace from 32 GB to 6 GiB
OLD2 = "                    'trt_max_workspace_size': 32 * 1024 ** 3,   # 32 GB"
NEW2 = "                    'trt_max_workspace_size': 6 * 1024 ** 3,    # 6 GiB"
if OLD2 not in src:
    print('FIX2 MISSING')
    sys.exit(1)
src = src.replace(OLD2, NEW2, 1)
print('Fix 2 OK')

# Fix 3: torch.cuda.empty_cache() before ort.InferenceSession to free PyTorch cache for TRT
OLD3 = ("                print('[opt] Building TRT engine (one-time compile, cached to disk) ...')\n"
        "                _ort_session = ort.InferenceSession(")
NEW3 = ("                print('[opt] Building TRT engine (one-time compile, cached to disk) ...')\n"
        "                torch.cuda.empty_cache()   # free PyTorch CUDA cache so TRT has room\n"
        "                _ort_session = ort.InferenceSession(")
if OLD3 not in src:
    print('FIX3 MISSING')
    sys.exit(1)
src = src.replace(OLD3, NEW3, 1)
print('Fix 3 OK')

with open(path, 'w') as f:
    f.write(src)
print('ALL FIXES APPLIED')
