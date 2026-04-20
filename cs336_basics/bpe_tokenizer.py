from collections import defaultdict
import regex as re
from copy import deepcopy
from multiprocessing import Pool
import json
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries

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
        with open(input_path, "rb") as f:
            num_processes = 8
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            process_chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                process_chunks.append(chunk)

        with Pool(processes=num_processes) as pool:
            results = pool.map(self._pre_tokenize, process_chunks)
        
        self._combine_pool_results(results)

        num_merges = self._vocab_size - len(self._vocab)
        with tqdm(total=num_merges, desc="BPE Merges", leave=True) as pbar:
            while len(self._vocab) < self._vocab_size:
                to_merge_pair = self._find_merge_pair()
                self._merge_pair(to_merge_pair)
                pbar.update(1)

    def _pre_tokenize(self, process_chunk):
        counts = defaultdict(int)
        pairwise_counts = defaultdict(int)
        for chunk in tqdm(re.split(self._delimiter, process_chunk), desc="Pre-tokenizing"):
            for token in re.finditer(self._pat, chunk):
                encoded_token = token.group().encode("utf-8")
                byte_list = tuple([bytes([elem]) for elem in encoded_token])
                counts[byte_list] += 1
                for i in range(len(byte_list)-1):
                    pairwise_counts[(byte_list[i], byte_list[i+1])] += 1
        return counts, pairwise_counts
    
    def _combine_pool_results(self, pool_results):
        for counts, pairwise_counts in pool_results:
            for pre_token, count in counts.items():
                self._counts[pre_token] += count
            
            for pair, pairwise_count in pairwise_counts.items():
                self._pairwise_count[pair] += pairwise_count
 
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
        for idx, location in enumerate(match_locations):
            match_location = location - idx
            updated_token = pre_token[:match_location] + (pre_token[match_location] + pre_token[match_location+1],) + pre_token[match_location+2:]
        return updated_token


def tokenize(input_path: str, vocab_size, special_tokens=["<|endoftext|>"]):
    bpe_tokenizer = BPETokenizer(vocab_size, special_tokens)
    bpe_tokenizer.fit(input_path)
    vocab = bpe_tokenizer.vocab
    merges = bpe_tokenizer.merges
    vocab_str = {str(k): v.decode('utf-8', 'replace') for k, v in vocab.items()}
    with open('vocab.json', 'w') as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)
    
    with open('merges.txt', 'w', encoding='utf-8') as f:
        for a, b in merges:
            f.write(f"{a.decode('utf-8', 'replace')} {b.decode('utf-8', 'replace')}\n")
    return vocab, merges


if __name__ == "__main__":
    # import cProfile
    import time
    input_path = "/home/gulgul/Downloads/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # cProfile.run('tokenize(input_path, 10000)', 'profile_output_v1.prof')
    start_time = time.time()
    tokenize(input_path, 10000)
    print(f"Runtime: {time.time()-start_time}")
