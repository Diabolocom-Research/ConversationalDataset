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
    engine_name = "Canary"
    available_languages = ["en", "de", "es", "fr"]
    output_dir = "predictions"

    # Create engine
    model = Engine.create(engine_name, **cfg)

    # Create dir if not exists
    Path(os.path.join(output_dir, engine_name)).mkdir(parents=True,
                                                      exist_ok=True)

    # Transcribe all available languages
    for lang_code in available_languages:
        # Load dataset
        dataset = TalkbankDataset(lang_code, 'switch')
        for i in tqdm(range(len(dataset)), total=len(dataset)):
            audio, metadata = dataset[i]
            switch_id = metadata['switch_id']
            output_file = os.path.join(output_dir, engine_name,
                                       switch_id + ".txt")
            path = f'{uuid.uuid4()}.wav'
            print(switch_id)

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
