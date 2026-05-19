import subprocess

def log_step(step, msg):
    print(f"\n[{step}] {msg}")
    print("=" * 50)

def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    result=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
    return float(result.stdout.strip())

