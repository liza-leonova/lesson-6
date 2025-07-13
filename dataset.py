from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from torch.nn.utils.rnn import pad_sequence
import torch
import re


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [s.strip() for s in sentences if s]


class TextDataset(Dataset):
    def __init__(self, data_type="train"):
        self.prepare_dataset(data_type)
        try:
            self.text_tokenizer = Tokenizer.from_file("data/tokenizer.json")
        except Exception:
            self.text_tokenizer = self.initialize_tokenizer()

    def initialize_tokenizer(self):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=30000,
            special_tokens=["<pad>", "<s>", "</s>"]
        )
        tokenizer.train_from_iterator(self.sentences, trainer)
        tokenizer.save('data/tokenizer.json')
        return tokenizer

    def prepare_dataset(self, data_type):
        with open(f'data/Zaratustra_{data_type}.txt', mode="r", encoding="utf-8") as f:
            text_content = f.read()
            self.sentences = split_into_sentences(text_content)

    def get_pad_token_id(self):
        return self.text_tokenizer.token_to_id('<pad>')

    def get_bos_token_id(self):
        return self.text_tokenizer.token_to_id('<s>')

    def get_eos_token_id(self):
        return self.text_tokenizer.token_to_id('</s>')

    def __len__(self):
        return len(self.sentences)

    def get_vocab_size(self):
        return self.text_tokenizer.get_vocab_size()

    def __getitem__(self, idx):
        bos_id = self.text_tokenizer.token_to_id('<s>')
        eos_id = self.text_tokenizer.token_to_id('</s>')
        sentence = self.sentences[idx]
        tokens = [bos_id] + self.text_tokenizer.encode(sentence).ids + [eos_id]
        input_seq = tokens[:-1]
        target_seq = tokens[1:]
        return {
            "src_ids": torch.tensor(input_seq, dtype=torch.long),
            "trgt_ids": torch.tensor(target_seq, dtype=torch.long)
        }


class BatchCollator:
    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, batch):
        input_seqs = [item['src_ids'] for item in batch if item is not None]
        target_seqs = [item['trgt_ids'] for item in batch if item is not None]
        padded_inputs = pad_sequence(input_seqs, batch_first=True, padding_value=self.pad_token)
        padded_targets = pad_sequence(target_seqs, batch_first=True, padding_value=self.pad_token)
        return {
            'src_ids': padded_inputs,
            'trgt_ids': padded_targets
        }


def create_dataloaders(
        batch_size=1,
        max_seq_len=128,
        max_train_samples=None,
        max_val_samples=None,
        workers=0
):
    train_set = TextDataset(data_type="train")
    val_set = TextDataset(data_type="val")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=BatchCollator(train_set.text_tokenizer.token_to_id('<pad>')),
        num_workers=workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchCollator(val_set.text_tokenizer.token_to_id('<pad>')),
        num_workers=workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_set