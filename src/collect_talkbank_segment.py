import os

import pandas as pd
from tqdm import tqdm
from src.dataset_utils import TalkbankDataset

if __name__ == '__main__':
    available_languages = ['en', 'zh', 'ja', 'de', 'fr', 'es']
    output_dir = "predictions"

    data = {
        'file': [],
        'transcription': [],
        'language': [],
        'subset': [],
        'full_language': [],
        'switch_id': [],
        'segment_id': [],
        'transcript_filename': [],
        'audio_len_sec': [],
        'orig_file_start': [],
        'orig_file_end': [],
        'channel': [],
        'whisper': [],
        'wav2vec2-large-960h': [],
        'canary': [],
        'wav2vec256': []
    }

    for lang_code in available_languages:
        # Load dataset
        dataset = TalkbankDataset(lang_code, 'segment')
        for i in tqdm(range(len(dataset)), total=len(dataset)):
            audio, metadata = dataset[i]
            segment_id = metadata['segment_id']

            data['file'].append(metadata['segment_id'] + 'mp3')
            data['transcription'].append(metadata['transcript'])
            data['language'].append(metadata['language_code'])
            data['subset'].append(metadata['subset'])
            data['full_language'].append(metadata['full_language'])
            data['switch_id'].append(metadata['switch_id'])
            data['segment_id'].append(metadata['segment_id'])
            data['transcript_filename'].append(metadata['transcript_filename'])
            data['audio_len_sec'].append(metadata['audio_len_sec'])
            data['orig_file_start'].append(metadata['orig_file_start'])
            data['orig_file_end'].append(metadata['orig_file_end'])
            data['channel'].append(metadata['channel'])

            whisper_file = os.path.join(output_dir, "Whisper",
                                        metadata['segment_id'] + '.txt')
            if os.path.isfile(whisper_file):
                data['whisper'].append(open(whisper_file, 'r').read())
            else:
                data['whisper'].append(None)

            wav2vec2_file = os.path.join(output_dir, "Wav2vec2",
                                         metadata['segment_id'] + '.txt')
            if os.path.isfile(wav2vec2_file):
                data['wav2vec256'].append(open(wav2vec2_file, 'r').read())
            else:
                data['wav2vec256'].append(None)

            canary_file = os.path.join(output_dir, "Canary",
                                       metadata['segment_id'] + '.txt')
            if os.path.isfile(canary_file):
                data['canary'].append(open(canary_file, 'r').read())
            else:
                data['canary'].append(None)

            wav2vec2m_file = os.path.join(output_dir, "Wav2vec2Multi",
                                          metadata['segment_id'] + '.txt')
            if os.path.isfile(wav2vec2m_file):
                data['wav2vec2-large-960h'].append(
                    open(wav2vec2m_file, 'r').read())
            else:
                data['wav2vec2-large-960h'].append(None)

    df = pd.DataFrame(data)
    df.to_csv('talkbank_df_segments.csv', index=False)
