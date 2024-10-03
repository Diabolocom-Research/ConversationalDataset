import os
from tqdm import tqdm
import traceback
from pathlib import Path
import soundfile as sf
import uuid
from src.engines import Engine
from src.dataset_utils import TalkbankDataset

if __name__ == '__main__':
    cfg = {"whisper_size": "large-v3", "device": "cuda"}
    engine_name = "Whisper"
    available_languages = ['en', 'zh', 'ja', 'de', 'fr', 'es']
    output_dir = "predictions"

    # Create engine
    model = Engine.create(engine_name, **cfg)

    # Create dir if not exists
    Path(os.path.join(output_dir, engine_name)).mkdir(parents=True,
                                                      exist_ok=True)

    # Transcribe all available languages
    for lang_code in available_languages:
        # Load dataset
        dataset = TalkbankDataset(lang_code, 'segment')
        for i in tqdm(range(len(dataset)), total=len(dataset)):
            audio, metadata = dataset[i]
            segment_id = metadata['segment_id']
            print(segment_id)
            output_file = os.path.join(output_dir, engine_name,
                                       segment_id + ".txt")
            path = f'{uuid.uuid4()}.wav'

            # Transcribe if file does not exist (so you can resume transcription at any time)
            if not os.path.isfile(output_file):
                try:
                    # Write waveform to local file and then delete
                    sf.write(path, audio, 16000)
                    transcription = model.process(path, lang_code)
                    with open(output_file, 'w') as f:
                        f.write(transcription)

                except KeyboardInterrupt:
                    break
                except:
                    print(traceback.format_exc())
                finally:
                    os.remove(path)
