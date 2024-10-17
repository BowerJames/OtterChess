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
        self.eos_token = "[EOS]"
        self.eom_token = "[EOM]"

        self.vocab = [self.eos_token, self.eom_token] + vocab
        self.vocab_size = len(self.vocab)

        self.ids_to_tokens = {idx: token for idx, token in enumerate(self.vocab)}
        self.tokens_to_ids = {token: idx for idx, token in enumerate(self.vocab)}
        
        
        
        # Assign IDs to special tokens
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.eom_token_id = 1
    
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = f.read().splitlines()
        return vocab
    
    def encode(self, text: str | list[str]):
        if isinstance(text, str):
            text = text.strip()

            text_list = self._tokenize(text)

            ids = self._convert_tokens_to_ids(text_list)

            return ids

        elif isinstance(text, list):
            text_lists = [self._tokenize(t.strip()) for t in text]

            idss = [self._convert_tokens_to_ids(text_list) for text_list in text_lists]

            return idss

    def _tokenize(self, text: str) -> list[str]:
        tokens = []
        moves = text.split(" ")
        for move in moves:
            assert len(move) in [4, 5]
            square_1 = move[:2]
            square_2 = move[2:4]
            tokens.append(square_1)
            tokens.append(square_2)
            if len(move) ==5:
                piece = move[4]
                tokens.append(piece)
            tokens.append("[EOM]")
        tokens.append("[EOS]")
        return tokens
            
        
    
    def _convert_tokens_to_ids(self, tokens: list[str]):
        return [self.tokens_to_ids[token] for token in tokens]
    
    def __call__(self, text: str | list[str], padding: bool = True, max_length: int | None = None, return_tensors: str | None = None):
        if isinstance(text, str):
            text = [text]
            batched=False
        else:
            batched=True
        
        if return_tensors:
            padding = True

        ids = self.encode(text)

        attention_mask = [
            [1] * len(ids_) for ids_ in ids
        ]

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

        if not batched:
            ids = ids[0]
            attention_mask = attention_mask[0]

        if return_tensors=="pt":
            ids = torch.tensor(ids)
            attention_mask = torch.tensor(attention_mask)

        return {"input_ids": ids, "attention_mask": attention_mask}
            
    def pad(self,encoded_inputs, return_tensors: str | None = None, **kwargs):
        input_ids = [
            encoded_input["input_ids"] for encoded_input in encoded_inputs
        ]
        attention_mask = [
            encoded_input["attention_mask"] for encoded_input in encoded_inputs
        ]

        maximum_length = max(
            [len(input_ids_) for input_ids_ in input_ids]
        )

        max_length = min(maximum_length, self.max_length)



        maximum_length = max([len(input_ids_) for input_ids_ in input_ids])
        if maximum_length > self.max_length:
            max_length = self.max_length
        else:
            max_length = maximum_length

        input_ids = [
            input_ids_ + [self.pad_token_id] * (max_length - len(input_ids_)) if len(input_ids_) < max_length else input_ids_[:max_length] for input_ids_ in input_ids
        ]

        attention_mask = [
            attention_mask_ + [0] * (max_length - len(attention_mask_)) if len(attention_mask_) < max_length else attention_mask_[:max_length] for attention_mask_ in attention_mask
        ]

        if return_tensors=="pt":
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
            
            
        



