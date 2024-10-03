import os
import uuid

import librosa
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset


def get_talkbank_dataset(language='en', dataset_type='segment'):
    dataset_train = load_dataset('diabolocom/talkbank_4_stt',
                                 split=f"{language}_{dataset_type}_train")
    dataset_test = load_dataset('diabolocom/talkbank_4_stt',
                                split=f"{language}_{dataset_type}_test")
    return concatenate_datasets([dataset_train, dataset_test])


class TalkbankDataset(Dataset):

    def __init__(self, language, dataset_type='segment'):
        self.dataset = get_talkbank_dataset(language=language,
                                            dataset_type=dataset_type)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx].copy()
        path = str(uuid.uuid4()) + '.mp3'

        with open(path, 'wb') as f:
            f.write(sample['audio']['bytes'])
        audio, _ = librosa.load(path, sr=16000, mono=True)
        os.remove(path)
        del sample['audio']

        return audio, sample
