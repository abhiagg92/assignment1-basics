from collections import defaultdict
import regex as re
from copy import deepcopy


# from cs336_basics.pretokenization_example import find_chunk_boundaries

class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=["<|endoftext|>"]):
        self._vocab_size = vocab_size
        self._special_tokens = special_tokens
        self._delimiter = "|".join(re.escape(token) for token in special_tokens)
        self._pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self._vocab = {idx:token.encode("utf-8") for idx, token in enumerate(special_tokens)}
        self._id_to_token = {i+len(special_tokens): bytes([i]) for i in range(256)}

        self._vocab.update(self._id_to_token)
         
        self._counts = defaultdict(int)
        self._pairwise_count = defaultdict(int)
        self._merges = []
    
    @property
    def merges(self):
        return self._merges
    
    @property
    def vocab(self):
        return self._vocab
    
    def fit(self, input_path: str):
        with open(input_path, "r") as f:
            data = f.read()

        for chunk in re.split(self._delimiter, data):
            for token in re.finditer(self._pat, chunk):
                encoded_token = token.group().encode("utf-8")
                byte_list = tuple([bytes([elem]) for elem in encoded_token])
                self._counts[byte_list] += 1
                for i in range(len(byte_list)-1):
                    self._pairwise_count[(byte_list[i], byte_list[i+1])] += 1

        while len(self._vocab) < self._vocab_size:     
            to_merge_pair = self._find_merge_pair()
            self._merge_pair(to_merge_pair)

    def _find_merge_pair(self):
        sorted_pairs = sorted(self._pairwise_count.items(), key=lambda x: x[1], reverse=True)
        most_common_pair = sorted_pairs[0]
        for pair in sorted_pairs[1:]:
            if pair[1] == most_common_pair[1] and max(most_common_pair[0], pair[0]) == pair[0]:
                most_common_pair = pair
            if pair[1] < most_common_pair[1]:
                break
        return most_common_pair[0]
        
    def _merge_pair(self, pair):
        vocab_size = len(self._vocab)
        merged_pair = pair[0] + pair[1]

        self._merges.append((pair))
        self._vocab[vocab_size] = merged_pair
        self._pairwise_count.pop(pair)

        # pre_token_counts = deepcopy(self._counts)
        to_update = []
        for pre_token, count in self._counts.items():

            match_locations = []
            last_match = False
            for i in range(len(pre_token) - 1):
                if last_match: # Avoid consecutive matches
                    last_match = False
                    continue
                if pre_token[i] == pair[0] and pre_token[i+1] == pair[1]:
                    match_locations.append(i)
                    last_match = True

            if not match_locations:
                continue
            
            for idx in range(len(match_locations)):
                match_location = match_locations[idx]
                if match_location < len(pre_token)-len(pair):
                    elem_after_match = pre_token[match_location+len(pair)]
                    self._pairwise_count[(merged_pair, elem_after_match)] += count

                    old_key = (pair[1], elem_after_match)
                    self._pairwise_count[old_key] -= count
                    if self._pairwise_count[old_key] <= 0:
                        self._pairwise_count.pop(old_key)

                if match_location > 0:
                    elem_before_match = pre_token[match_location-1]
                    self._pairwise_count[(elem_before_match, merged_pair)] += count

                    old_key = (elem_before_match, pair[0])
                    self._pairwise_count[old_key] -= count
                    if self._pairwise_count[old_key] <= 0:
                        self._pairwise_count.pop(old_key)
                
            update_token = self._update_pre_token(pre_token, match_locations)
            to_update.append((pre_token, update_token))


        for old, new in to_update:
            self._counts[new] += self._counts.pop(old)

    def _update_pre_token(self, pre_token, match_locations):
        updated_token = deepcopy(pre_token)
        for idx, location in enumerate(match_locations):
            match_location = location - idx
            updated_token = updated_token[:match_location] + (updated_token[match_location] + updated_token[match_location+1],) + updated_token[match_location+2:]
        # token_count = self._counts[pre_token]
        # self._counts.pop(pre_token)
        # self._counts[updated_token] = token_count
        return updated_token


def tokenize(input_path: str, vocab_size, special_tokens=["<|endoftext|>"]):
    bpe_tokenizer = BPETokenizer(vocab_size, special_tokens)
    bpe_tokenizer.fit(input_path)
    return bpe_tokenizer.vocab, bpe_tokenizer.merges


if __name__ == "__main__":
    import cProfile
    input_path = "/home/gulgul/Downloads/cs336/assignment1-basics/tests/fixtures/corpus.en"
    cProfile.run('tokenize(input_path, 500)', 'profile_output_v1.prof')
