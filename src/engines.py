import os
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import json
import whisper
from nemo.collections.asr.models import EncDecMultiTaskModel


class Engine(object):

    @staticmethod
    def create(engine, **kwargs):
        if engine == "Whisper":
            return WhisperEngine(**kwargs)
        elif engine == "Wav2vec2":
            return Wav2vec2Engine(**kwargs)
        elif engine == "Canary":
            return CanaryEngine(**kwargs)
        elif engine == "Wav2vec2Multi":
            return Wav2vec2MultilingualEngine(**kwargs)
        else:
            ValueError("cannot create engine of type", engine)


class WhisperEngine(object):

    def __init__(self, whisper_size="large-v3", device="cuda"):
        """
		Args references:
			https://github.com/openai/whisper/blob/main/whisper/__init__.py#L99 # load_model
			https://github.com/openai/whisper/blob/main/whisper/decoding.py#L81 # transcribe_args
			https://github.com/openai/whisper/blob/main/whisper/transcribe.py#L38 # transcribe
		"""
        self.model = whisper.load_model(whisper_size).to(device)

    def process(self, audio_file, language=None):
        """
        Transcribe file
        :param language: lang code
        :param audio_file: path for file
        :return: transcription
        """
        result = self.model.transcribe(audio_file, language=language)
        return result['text']

    def __str__(self):
        return 'WHISPER'


class Wav2vec2Engine(object):

    def __init__(self, size="large", device="cuda"):
        """
		https://huggingface.co/facebook/wav2vec2-base-960h
		Params:
			size: large - base
		"""
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(
            f"facebook/wav2vec2-{size}-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained(
            f"facebook/wav2vec2-{size}-960h")
        self.model.to(self.device)

    def _process_seg(self, audio):
        input_values = self.processor(audio,
                                      return_tensors="pt",
                                      padding="longest",
                                      sampling_rate=16000).input_values
        with torch.no_grad():
            input_values = input_values.to(self.device)
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription

    def process(self, audio):
        if isinstance(audio, str) and os.path.isfile(audio):
            audio, sr = librosa.load(audio, sr=16000)
            assert sr == 16000
        return self._process_seg(audio)

    def __str__(self):
        return 'Wav2vec2'


class Wav2vec2MultilingualEngine(object):

    def __init__(self, device="cuda"):
        """
		https://huggingface.co/facebook/wav2vec2-base-960h
		Params:
			size: large - base
		"""
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(
            f"voidful/wav2vec2-xlsr-multilingual-56")
        self.model = Wav2Vec2ForCTC.from_pretrained(
            f"voidful/wav2vec2-xlsr-multilingual-56")
        self.model.to(self.device)

    def _process_seg(self, audio):
        input_values = self.processor(audio,
                                      return_tensors="pt",
                                      padding="longest",
                                      sampling_rate=16000).input_values
        with torch.no_grad():
            input_values = input_values.to(self.device)
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription

    def process(self, audio):
        if isinstance(audio, str) and os.path.isfile(audio):
            audio, sr = librosa.load(audio, sr=16000)
            assert sr == 16000
        return self._process_seg(audio)

    def __str__(self):
        return 'Wav2vec2'


class CanaryEngine(object):

    def __init__(self,
                 batch_size=1,
                 language="en",
                 beam_size=1,
                 device="cuda",
                 **kwargs):
        """
		https://huggingface.co/nvidia/canary-1b
		"""
        # load model
        self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
        self.model.to(device)
        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = beam_size
        self.model.change_decoding_strategy(decode_cfg)
        self.batch_size = batch_size

    def process(self, audio_path, language=None):
        manifest = {
            "audio_filepath": audio_path,  # path to the audio file
            "duration":
            100,  # duration of the audio, can be set to `None` if using NeMo main branch
            "taskname":
            "asr",  # use "s2t_translation" for speech-to-text translation with r1.23, or "ast" if using the NeMo main branch
            "source_lang":
            language,  # language of the audio input, set `source_lang`==`target_lang` for ASR, choices=['en','de','es','fr']
            "target_lang":
            language,  # language of the text output, choices=['en','de','es','fr']
            "pnc": "yes",  # whether to have PnC output, choices=['yes', 'no']
            "answer": "na",
        }
        with open('input_manifest.json', 'w') as f:
            f.write(json.dumps(manifest))
        predicted_text = self.model.transcribe('input_manifest.json',
                                               self.batch_size)
        return predicted_text[0]

    def __str__(self):
        return 'Canary'
