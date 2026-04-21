import json
from typing import Iterable, Iterator
import regex as re

ByteToken = bytes
PreToken = tuple[ByteToken, ...]

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = special_tokens

        self._token_to_id = {token: id for id, token in self._vocab.items()}
        self._merge_to_order = {merge: idx for idx, merge in enumerate(self._merges)}
        self._merge_count = len(self._merges)

        self._delimiter = None
        if special_tokens:
            self._special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
            self._delimiter = f"({'|'.join(re.escape(token) for token in self._special_tokens)})"
        self._pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None) -> "BPETokenizer":
        with open(vocab_filepath, 'r', encoding="utf-8") as f:
            vocab = json.loads(f)
        
        merges = []
        with open (merges_filepath, "r", encoding="utf-8")as f:
            lines = f.readlines()
            for line in lines:
                merges.append(tuple(line.split()))
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encoding = []
        if self._delimiter:
            for chunk in re.split(self._delimiter, text):
                if chunk in self._special_tokens:
                    encoding.append(self._token_to_id[chunk.encode('utf-8')])
                    continue
                encoding.extend(self._encode_chunk(chunk))
        else:
            encoding = self._encode_chunk(text)
        return encoding

    def _encode_chunk(self, chunk: str) -> list[int]:
        chunk_encoding = []
        for pretoken in re.finditer(self._pat, chunk):
            encoded_token = pretoken.group().encode("utf-8")
            byte_pre_token: PreToken = tuple([bytes([elem]) for elem in encoded_token])
            tokens = self._merge_pretoken(byte_pre_token)
            chunk_encoding.extend([self._token_to_id[token] for token in tokens])
        return chunk_encoding

    def _merge_pretoken(self, pre_token: PreToken):
        if len(pre_token) == 1:
            return pre_token
        first_merge_idx_pair = (0, 1)
        first_merge_pair = (pre_token[0], pre_token[1])
        first_merge_order = self._merge_to_order.get(first_merge_pair, self._merge_count+1)
        for i in range(1, len(pre_token)-1):
            new_merge_idx_pair = (i, i+1)
            new_merge_pair = (pre_token[i], pre_token[i+1])
            new_merge_order = self._merge_to_order.get(new_merge_pair, self._merge_count+1)
            if new_merge_order < first_merge_order:
                first_merge_idx_pair = new_merge_idx_pair
                first_merge_pair = new_merge_pair
                first_merge_order = new_merge_order

        if first_merge_order == self._merge_count+1:
            return pre_token
        else:
            pre_token = (pre_token[:first_merge_idx_pair[0]] + (first_merge_pair[0]+first_merge_pair[1],) + pre_token[first_merge_idx_pair[1]+1:])
            return self._merge_pretoken(pre_token)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        binary_text = b''
        for id in ids:
            binary_text += self._vocab.get(id, "\ufffd".encode("utf-8"))
        
        return binary_text.decode("utf-8", errors='replace')


if __name__ == "__main__":
    tokenizer = BPETokenizer(
        vocab = {
            0: b' ',
            1: b'a',
            2: b'c',
            3: b'e',
            4: b'h',
            5: b't',
            6: b'th',
            7: b' c',
            8: b' a',
            9: b'the',
            10: b' at'
        },
        merges= [
            (b't', b'h'),
            (b' ', b'c'),
            (b' ', b'a'),
            (b'th', b'e'),
            (b' a', b't'),
        ],
    )
    # print(tokenizer.encode("the cat ate"))
    print(tokenizer.decode([9,20, 7,1,5,10,3]))