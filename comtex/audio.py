import audio2numpy as a2n
import math
import numpy as np
import subprocess
import torch
import whisperx
import json
import syllables
import os 
import pandas as pd 
from scipy import signal, stats
from scipy.stats import entropy, levene
from scipy.signal import find_peaks
from scipy.special import rel_entr
from scipy.signal import butter, filtfilt, hilbert, chirp
from scipy.fftpack import fft, rfft
from scipy.io import wavfile as io
from comtex.utils import *

def extract_audio_from_mp4(video_path, output_path):
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

#spectral flux functions
def normalize(sig, rms_level=20):
    """
    Normalize the signal given a certain technique (peak or rms).
    Args:
        - infile    (str) : input filename/path.
        - rms_level (int) : rms level in dB.
    """
    r = 10**(rms_level / 10.0)
    a = np.sqrt( (len(sig) * r**2) / np.sum(sig**2) )

    # normalize
    y = sig * a

    return y

def compute_spectrogram(x,rate): ##
    NFFT=len(x)
    b = x 
    f, B = signal.periodogram(b, rate,nfft =NFFT,scaling = 'density')
    return(f,B)

def get_spectral_flux(audio_path, output_path, ws=0.2, hs=0.02):
    import audio2numpy as a2n
    import math
    import numpy as np

    x, sr = a2n.audio_from_file(audio_path)
    duration = len(x) / sr

	# x_total, rate = librosa.load(filename ,sr = 44100) 
	#normalise each audio file
    x_total = normalize(x,20)

    N = len(x_total)
    rate = sr
    #compute number of windows based on lenght, ws and hs.
    nb_window = math.floor((N - (ws*rate))/(hs*rate)) + 1
    print('nb window = ', nb_window)
    SF = np.zeros(nb_window)
    SF_norm = np.zeros(nb_window)
    #set maximum frequency of interest to 3000Hz
    freq_max_of_interest = math.floor(3000 * ws)
    for window in range(nb_window):
        # print(str(window) + ' / ' + str(nb_window))
        start_sample = math.floor(window * hs * rate)
        end_sample = start_sample + math.floor(ws*rate)
        
        #select first window
        x_sub = x_total[start_sample:end_sample]
        #compute spectrogram of that time window
        f,B_sub_previous = compute_spectrogram(x_sub,rate)
        
        start_sample = start_sample + math.floor(hs*rate)
        end_sample = end_sample + math.floor(hs*rate)
        #select second window
        x_sub = x_total[start_sample:end_sample]
        #compute spectrogram of that time window
        f,B_sub_next = compute_spectrogram(x_sub,rate)

        ### plot consecutive spectrogram in same plot
        # plt.plot(f[0:freq_max_of_interest],B_sub_previous[0:freq_max_of_interest],label = 'fft previous')
        # plt.plot(f[0:freq_max_of_interest],B_sub_next[0:freq_max_of_interest],label = 'fft next')
        # plt.legend()
        # plt.show()
        
        # compute the spectral flux as the sum of the absolute differences in amplitude across frequency
        SF[window] = sum(abs(B_sub_previous[1:freq_max_of_interest] - B_sub_next[1:freq_max_of_interest]))
        SF_norm[window] = sum(abs((B_sub_previous[1:freq_max_of_interest])/np.sum((B_sub_previous[1:freq_max_of_interest])) - (B_sub_next[1:freq_max_of_interest])/np.sum(B_sub_next[1:freq_max_of_interest])))

    time_s = np.arange(len(SF)) * hs

    if '.npz' in output_path:
        np.savez_compressed(output_path, spectral_flux=SF, time_s=time_s)
    else:
        print(f"Couldn't save file: must be .npz format. Returning SF series still")

    return SF, SF_norm, time_s
