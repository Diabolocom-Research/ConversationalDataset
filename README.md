# ConversationalDataset: Benchmarking Conversations

This repository will host **benchmarks** and **datasets** related to conversational AI tasks.
---

## 🗂 Processed TalkBank Dataset for ASR benchmarking

Pre-processed dataset with train-test splits for all languages is avaialble at [Hugging Face](https://huggingface.co/datasets/diabolocom/talkbank_4_stt) as described in this [Paper](https://arxiv.org/abs/2409.12042).


## 🚀 Setting Up the Environment

To set up the environment and install necessary dependencies, run:

```bash
conda env create -f environment.yml
```



## 📊 Benchmarking

To generate transcripts using various ASR systems on the TalkBank dataset, use the following scripts:

#### Segment

```bash
python src/run_canary_prediction_segment.py #  Canary 1b
python src/run_whisper_prediction_segment.py # Whisper large-v3
python src/run_wav2vec2_prediction_segment.py # Wav2vec2
python src/run_wav2vec2multi_prediction_segment.py # Wav2vec2 multilingual
```

#### Switch

```bash
python src/run_canary_prediction_switch.py # Canary 1b
python src/run_whisper_prediction_switch.py # Whisper large-v3
```

After generating the transcripts, consolidate them into a CSV file for further analysis:

```bash
python src/collect_talkbank_segment.py # Produces talkbank_df_segments.csv
python src/collect_talkbank_switch.py # Produces talkbank_df_switch.csv
```


## Work in Progress 🚧

- **Pre-processing code**: Coming soon! We will upload scripts for cleaning, formatting, and preparing TalkBannk dataset subset.

Stay tuned for updates!


