from transformers import PreTrainedTokenizer
import collections
import torch
import os

file = __file__

class ChessTokenizer:
    def __init__(self, **kwargs):
        vocab_file = os.path.join(os.path.dirname(file), "vocab.txt")
        self.vocab_file = vocab_file
        vocab = self.load_vocab(vocab_file)

        self.max_length = kwargs.get("max_length", 256)

        # Define special tokens
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"

        self.vocab = [self.pad_token, self.eos_token] + vocab
        self.vocab_size = len(self.vocab)

        self.ids_to_tokens = {idx: token for idx, token in enumerate(self.vocab)}
        self.tokens_to_ids = {token: idx for idx, token in enumerate(self.vocab)}
        
        
        
        # Assign IDs to special tokens
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = f.read().splitlines()
        return vocab
    
    def encode(self, text: str | list[str]):
        if isinstance(text, str):

            text_list = self._tokenize(text)
            text_list = text_list

            ids = self._convert_tokens_to_ids(text_list)

            return ids

        elif isinstance(text, list):
            text_lists = [self._tokenize(t) for t in text]
            text_lists = [
                text_list + [self.eos_token] for text_list in text_lists
            ]

            idss = [self._convert_tokens_to_ids(text_list) for text_list in text_lists]

            return idss

    def _tokenize(self, text: str):
        return text.split(" ")
    
    def _convert_tokens_to_ids(self, tokens: list[str]):
        return [self.tokens_to_ids[token] for token in tokens]
    
    def __call__(self, text: str | list[str], padding: bool = True, max_length: int = None, return_tensors: bool = False):
        if isinstance(text, str):
            text = [text]

        if return_tensors:
            padding = True

        ids = self.encode(text)

        attention_mask = [
            [1] * len(ids_) for ids_ in ids
        ]

        ids = [ids_ + [self.eos_token_id] for ids_ in ids]
        attention_mask = [attention_mask_ + [0] for attention_mask_ in attention_mask]

        if padding:
            maximum_length = max([len(ids_) for ids_ in ids])
            if maximum_length > self.max_length:
                max_length = self.max_length
            else:
                max_length = maximum_length

            ids = [
                ids_ + [self.pad_token_id] * (max_length - len(ids_)) if len(ids_) < max_length else ids_[:max_length] for ids_ in ids 
            ]

            attention_mask = [
                attention_mask_ + [0] * (max_length - len(attention_mask_)) if len(attention_mask_) < max_length else attention_mask_[:max_length] for attention_mask_ in attention_mask
            ]

        if return_tensors:
            ids = torch.tensor(ids)
            attention_mask = torch.tensor(attention_mask)

        return {"input_ids": ids, "attention_mask": attention_mask}
            
            
        



