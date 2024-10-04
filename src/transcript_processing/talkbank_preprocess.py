import os
import json
import pandas as pd
from pydub import AudioSegment
from rich.progress import track
import numpy as np

from .CHAFile.ChaFile import *
from .preprocess_talkbank_text import preprocess_talkbank_text


def get_files_path(directory):
	files_path = []
	for f in os.listdir(directory):
		if os.path.isdir(os.path.join(directory, f)):
			files_path.extend(get_files_path(os.path.join(directory, f)))
		else:
			files_path.append(os.path.join(directory, f))
	return files_path


def match_audio_transcription(files_path):
	data = {}
	for f in files_path:
		if f.endswith(".cha"):
			name = os.path.splitext(os.path.basename(f))[0]
			if name not in data:
				data[name] = {"audio": None, "transcript": f}
			else:
				data[name]["transcript"] = f
		# elif f.endswith(".mp3") or f.endswith(".mp4") or f.endswith(".wav"):
		else:
			name = os.path.splitext(os.path.basename(f))[0]
			if name not in data:
				data[name] = {"audio": f, "transcript": None}
			else:
				data[name]["audio"] = f
	# fill a csv with validated audio + transcript
	data_df = {"name": [], "audio": [], "transcript": []}
	for k, v in data.items():
		if v["audio"] is not None and v["transcript"] is not None:
			data_df["name"].append(k)
			data_df["audio"].append(v["audio"])
			data_df["transcript"].append(v["transcript"])
	df = pd.DataFrame(data_df)
	return df


def preprocess_data(name, audio_path, transcript_path, output_dir=None, sampling_rate=16000, timestamps_pad=100, max_silence=1000, min_segment=300, min_audio=3000, overwrite=True):
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	audio = AudioSegment.from_file(audio_path)
	audio = audio.set_frame_rate(sampling_rate)
	if audio.channels == 2:
		split_audios = audio.split_to_mono()
		audios = [split_audios[0], split_audios[1]]
	elif audio.channels == 1:
		audios = [audio]
	else:
		raise ValueError(f"More than 2 channels audio not implemented yet.\nName: {name}\nNb channels: {audio.channels}")

	cha = ChaFile(transcript_path)
	lines = cha.getLines()

	# split by speaker
	speaker_data = {}
	speaker_channels = {}
	for line in lines:
		if "bullet" not in line:
			continue
		speaker = line["hablante"]
		text = line["emisión"]
		timestamps = line["bullet"]
		start, end = timestamps[0], timestamps[1]
		if speaker not in speaker_data:
			speaker_data[speaker] = []
			speaker_channels[speaker] = []
		if len(speaker_data[speaker]) > 0 and start - speaker_data[speaker][-1]["end"] <= timestamps_pad * 3:
			speaker_data[speaker][-1]["end"] = end
			speaker_data[speaker][-1]["text"] += "\n" + text
		else:
			speaker_data[speaker].append({"start": start, "end": end, "text": text})

		# detect the good channel for each speaker
		values = []
		for a in audios:
			values.append(a[start:end].dBFS)
		speaker_channels[speaker].append(np.argmax(values))

	for k in speaker_channels:
		speaker_channels[k] = round(np.mean(speaker_channels[k]))

	# generate new .wav files and .csv
	for speaker, turns in speaker_data.items():
		wav_path = os.path.join(output_dir, name + "_" + speaker + ".wav")
		csv_path = os.path.join(output_dir, name + "_" + speaker + ".csv")
		speaker_audio = AudioSegment.empty()
		df_speaker = []
		prev_end = 0
		cut = 0
		for row in turns:
			start = int(max(0, row["start"] - timestamps_pad))
			end = int(min(audio.duration_seconds * 1000, row["end"] + timestamps_pad))
			text = row["text"]
			if end - start < min_segment:
				continue
			if len(df_speaker) > 0:
				silence_duration = min(max_silence, start - prev_end)
				speaker_audio += AudioSegment.silent(duration=silence_duration)
				cut -= silence_duration
			speaker_audio += audios[speaker_channels[speaker]][start:end]
			cut += start - prev_end
			prev_end = end
			df_speaker.append({"start_ms": start - cut, "end_ms": end - cut, "text": text})
		if speaker_audio.duration_seconds < min_audio / 1000:
			continue
		if overwrite or not os.path.isfile(wav_path):
			speaker_audio.export(wav_path, format="wav")
		df_speaker = pd.DataFrame(data=df_speaker)
		df_speaker["preprocess_text"] = df_speaker["text"].apply(lambda x: preprocess_talkbank_text(x, remove_tags=True))
		if overwrite or not os.path.isfile(csv_path):
			df_speaker.to_csv(csv_path, index=False, encoding="utf-8")


def talkbank_preprocess(original_dir, output_dir=None, sampling_rate=16000, overwrite=True):
	"""
	Select a directory containing .cha and .mp3 files downloaded on https://www.talkbank.org/
	It will preprocess audio:
		- split audio by speaker by file (1 .wav file / speaker / file)
		- split transcription by speaker by file (1 .csv file / speaker / file)
		- set sampling rate
		- set mono channel
		- replace everything else than speech by silence
	"""
	if output_dir is None:
		output_dir = "preprocess_" + original_dir
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	print("Saving preprocess data into directory:", output_dir)
	files_path = get_files_path(original_dir)
	df = match_audio_transcription(files_path)
	for i in track(df.index, description=f"[cyan]Preprocessing {original_dir}"):
		preprocess_data(df["name"][i], df["audio"][i], df["transcript"][i], output_dir=output_dir,
			sampling_rate=sampling_rate, overwrite=overwrite)
	return output_dir


if __name__ == '__main__':
	talkbank_preprocess("ca/CallFriend/French - Quebecois", overwrite=True)
