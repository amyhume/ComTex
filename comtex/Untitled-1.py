

def get_processing_progress():
    import os 
    from pathlib import Path
    from comtex.utils import *
    import pandas as pd

    home = Path.home()

    if 'mario' in str(home):
        pcloud = 'pCloudDrive'
    else:
        pcloud = 'pCloud Drive'
    
    #listing all files
    videos = ([f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/corpus_13/videos') if f.endswith('.mp4')]
              + [f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/corpus_68/videos') if f.endswith('.mp4')])
    audios = ([f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/scripts/audio_13') if f.endswith('.wav')]
                +[f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/scripts/audio_68') if f.endswith('.wav')])

    trans = ([f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/scripts/transcriptions_13_v3') if f.endswith('.csv')]
            + [f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/scripts/transcriptions_68_v3') if f.endswith('.csv')])

    ling_sem = (
        [f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/scripts/transcriptions_13_v3/output/EleutherAI_pythia-1b') if f.endswith('.csv')]
        + [f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/scripts/transcriptions_13_v3/output/EleutherAI_pythia-1b_2') if f.endswith('.csv')]
        + [f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/scripts/transcriptions_68_v3/output/EleutherAI_pythia-1b_2') if f.endswith('.csv')]
    )     

    captions = ([f for f in os.listdir(f'{home}/{pcloud}/Lev2/Pierre/Blip_captions') if f.endswith('.csv') and f.startswith('1')]
                + [f for f in os.listdir(f'{home}/{pcloud}/Lev2/Pierre/Blip_captions') if f.endswith('.csv') and f.startswith('6')])

    vis_sem = ([f for f in os.listdir(f'{home}/{pcloud}/Lev2/Pierre/Blip_captions_semantic') if f.endswith('.csv') and f.startswith('1')]
                + [f for f in os.listdir(f'{home}/{pcloud}/Lev2/Pierre/Blip_captions_semantic') if f.endswith('.csv') and f.startswith('6')])

    flicker = ([f for f in os.listdir(f'{home}/{pcloud}/Lev2/Pierre/flicker_timeseries') if f.endswith('.npz') and f.startswith('1')]
                + [f for f in os.listdir(f'{home}/{pcloud}/Lev2/Pierre/flicker_timeseries') if f.endswith('.npz') and f.startswith('6')])
    sf = ([f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/scripts/sf_timeseries') if f.endswith('.npz') and f.startswith('1')]
            + [f for f in os.listdir(f'{home}/{pcloud}/Lev2/corpus_analysis/scripts/sf_timeseries') if f.endswith('.npz') and f.startswith('6')])

    #making long df 
    long_df = []

    print('CHECKING PROCESSING STATUS FOR EACH VIDEO. May take a while...')
    print('===============================================================')
    for i, v in enumerate(videos):
        print(f"Video {i}/{len(videos)}")
        group = '1-4' if v.startswith('1') else '6-8'
        vid = v[:5]

        group_path = 'corpus_13' if group == '1-4' else 'corpus_68'
        dur = get_video_duration(f'{home}/{pcloud}/Lev2/corpus_analysis/{group_path}/videos/{v}')

        long_df.append({
            'video_id': vid, 'group': group, 'duration': dur,
            'audio': 1 if f"{vid}.wav" in audios else 0,
            'transcription': 1 if f"{vid}.csv" in trans else 0, 'linguistic_semantics': 1 if f"{vid}.csv" in ling_sem else 0,
            'frame_captions': 1 if f"{vid}_captions.csv" in captions else 0, 'visual_semantics': 1 if f"{vid}_captions.csv" in vis_sem else 0,
            'flicker': 1 if f"{vid}_flicker.npz" in flicker else 0, 'spectral_flux': 1 if f"{vid}_sf.npz" else 0
        })
        
    long_df = pd.DataFrame(long_df)

    #summary_df 
    summary_df = pd.DataFrame({
        'file_type': ['videos', 'audios', 'transcriptions', 'linguistic_semantics', 'frame_captions', 'visual_semantics', 'spectral_flux', 'flicker'],
        'n_1_4': [],
        'n_6_8': 

    })


