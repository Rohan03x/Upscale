import torch, time, sys, os
sys.path.insert(0, r'C:\VideoUpscale\tools\Real-ESRGAN')
from basicsr.archs.hat_arch import HAT

import shutil, subprocess, glob, pathlib
if shutil.which('cl') is None:
    bats = glob.glob(r'C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools\VC\Auxiliary\Build\vcvars64.bat')
    if bats:
        out = subprocess.run(f'cmd /c "{bats[0]}" && set', shell=True, capture_output=True, text=True).stdout
        for line in out.splitlines():
            if '=' in line:
                k, _, v = line.partition('=')
                if k.upper() in ('PATH','INCLUDE','LIB','LIBPATH','VCINSTALLDIR'):
                    os.environ[k] = v
pylibs = str(pathlib.Path(sys.base_prefix) / 'libs')
if pylibs.lower() not in os.environ.get('LIB','').lower():
    os.environ['LIB'] = pylibs + ';' + os.environ.get('LIB','')
print('cl.exe:', shutil.which('cl'))

model = HAT(upscale=4, in_chans=3, img_size=64, window_size=16,
            compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
            overlap_ratio=0.5, img_range=1., depths=[6,6,6,6,6,6],
            embed_dim=180, num_heads=[6,6,6,6,6,6], mlp_ratio=2,
            upsampler='pixelshuffle', resi_connection='1conv')
model.load_state_dict(torch.load(r'C:\VideoUpscale\tools\Real-ESRGAN\weights\Real_HAT_GAN_SRx4_sharper.pth', map_location='cpu')['params_ema'])
model = model.half().eval().cuda()
if hasattr(model, 'mean'):
    model.mean = model.mean.to(dtype=torch.float16, device='cuda')

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()
model_c = torch.compile(model, mode='default')
tile = torch.randn(1, 3, 512, 512, dtype=torch.float16, device='cuda')
print('Warmup (compiling kernels, ~1-3 min)...', flush=True)
with torch.no_grad():
    _ = model_c(tile)
torch.cuda.synchronize()
print('Timing...', flush=True)
t0 = time.time()
for _ in range(3):
    with torch.no_grad():
        _ = model_c(tile)
torch.cuda.synchronize()
t = (time.time()-t0)/3
print(f'compile(default) 512x512: {t:.2f}s  (eager: 28.54s)  speedup: {28.54/t:.1f}x')
print(f'Estimated total: {t*12*1098/3600:.1f} hours for 1098 frames')
