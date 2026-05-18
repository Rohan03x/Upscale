import os, time, shutil, sys
src = '/dev/shm/upscale_temp/Mortal Kombat II/Mortal Kombat II_s2_downscaled_esrgan.mp4'
dst = '/workspace/s3_esrgan_backup.mp4'
print('[watcher] waiting for stage 3 output...', flush=True)
while True:
    if os.path.exists(src):
        prev = 0
        while True:
            cur = os.path.getsize(src) if os.path.exists(src) else 0
            if cur == prev and cur > 10**8:
                print(f'[watcher] stable at {cur} bytes, backing up...', flush=True)
                shutil.copy2(src, dst)
                print('[watcher] BACKUP DONE', flush=True)
                sys.exit(0)
            prev = cur
            time.sleep(15)
    time.sleep(15)
