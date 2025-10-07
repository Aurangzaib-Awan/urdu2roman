
import torch

def build_vocab(text_series):
    chars = set()
    for line in text_series:
        chars.update(list(line))
    return sorted(list(chars))

def build_tokenizers(train_df):
    urdu_vocab_list = ['<pad>', '<sos>', '<eos>'] + build_vocab(train_df['urdu'])
    roman_vocab_list = ['<pad>', '<sos>', '<eos>'] + build_vocab(train_df['roman'])

    urdu2idx = {ch: i for i, ch in enumerate(urdu_vocab_list)}
    idx2urdu = {i: ch for i, ch in enumerate(urdu_vocab_list)}
    roman2idx = {ch: i for i, ch in enumerate(roman_vocab_list)}
    idx2roman = {i: ch for i, ch in enumerate(roman_vocab_list)}

    MAX_LEN = 50

    return urdu2idx, idx2urdu, roman2idx, idx2roman, MAX_LEN