def log_step(step, msg):
    print(f"\n[{step}] {msg}")
    print("=" * 50)

def extract_audio_from_mp4(video_path, output_path):
	import subprocess
	# ffmpeg command: extract mono WAV (16-bit PCM)
	cmd = [
		"ffmpeg", "-y",            # overwrite if exists
		"-i", video_path,          # input video
		"-vn",                     # no video
		"-ac", "1",                # mono (1 channel)
		"-acodec", "pcm_s16le",    # uncompressed 16-bit PCM
		output_path
	]
	subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
	return print('extracted audio for ', video_path)

#TRANSCRIPTION FUNCTIONS
def load_whisperx_model(model_name='large_v3', device=None, compute_type=None):
    import torch
    import whisperx
    if device is None:
          device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if compute_type is None:
          compute_type = 'float16' if device == 'cuda' else 'int8'

    model = whisperx.load_model(
          model_name, device=device, compute_type=compute_type
    )
    return model, device, compute_type

def generate_text_info_timestamps_whisperx(
    audio_path,
    json_out_path,
    model,
    device=None,
    batch_size=2,
    compute_type="float16"
):
    import whisperx
    import json
    log_step(1, f'Starting transcription: {audio_path}')

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    language = result.get("language", None)
    log_step(2, f"Detected language: {language}")

    # Attempt alignment
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device
        )
    except ValueError:
        log_step('Error: ', f"No alignment model for language={language}. skipping file...")
        return None, language

    aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    output = {
        "language": language,
        "segments": aligned["segments"]
    }

    with open(json_out_path, "w") as f:
        json.dump(output, f, indent=2)

    log_step(3, f"Saved JSON: {json_out_path}")
    return json_out_path, language

def json_to_csv_absolute(json_path, csv_path):
    import json
    import numpy as np
    import syllables
    import os 
    import pandas as pd 

    with open(json_path) as f:
        data = json.load(f)

    language = data.get("language", "unknown")
    rows = []

    for seg in data.get("segments", []):
        words = seg.get("words", [])

        for w in words:
            text = w.get("word") or w.get("text", "")
            onset = w.get("start", np.nan)
            offset = w.get("end", np.nan)
            confidence = w.get("score", np.nan)

            try:
                syll = syllables.estimate(text)
            except Exception:
                syll = np.nan

            rows.append({
                "filename": os.path.basename(json_path),
                "onset": onset,
                "offset": offset,
                "text": text,
                "confidence": confidence,
                "language": language,
                "syllable": syll
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    log_step('', f"Saved CSV: {csv_path}")






