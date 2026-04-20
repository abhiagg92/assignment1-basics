from collections import defaultdict
import regex as re
from multiprocessing import Pool
import json
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries

# Type aliases for readability
ByteToken = bytes
PreToken = tuple[ByteToken, ...]
BytePair = tuple[ByteToken, ByteToken]

class BPETrainer:
    """Byte Pair Encoding trainer that learns subword units from text.
    
    This trainer implements the BPE algorithm which iteratively merges
    the most frequent adjacent byte pairs to build a vocabulary of subword units.
    """

    def __init__(self, vocab_size: int, special_tokens: list[str] = ["<|endoftext|>"]) -> None:
        """Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size including special tokens and base bytes.
            special_tokens: List of special tokens to include in the vocabulary.
        """
        self._vocab_size = vocab_size
        self._special_tokens = special_tokens
        self._delimiter = "|".join(re.escape(token) for token in special_tokens)
        self._pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self._vocab: dict[int, bytes] = {idx: token.encode("utf-8") for idx, token in enumerate(special_tokens)}
        self._id_to_token: dict[int, bytes] = {i + len(special_tokens): bytes([i]) for i in range(256)}

        self._vocab.update(self._id_to_token)

        self._counts: defaultdict[PreToken, int] = defaultdict(int)
        self._pairwise_count: defaultdict[BytePair, int] = defaultdict(int)
        self._merges: list[BytePair] = []
    
    @property
    def merges(self) -> list[BytePair]:
        """Return the list of learned merge operations."""
        return self._merges

    @property
    def vocab(self) -> dict[int, bytes]:
        """Return the vocabulary mapping token IDs to byte sequences."""
        return self._vocab

    def fit(self, input_path: str) -> None:
        """Train the BPE tokenizer on the given input file.
        
        Reads the input file in chunks, pre-tokenizes using GPT-2 style regex,
        then iteratively merges the most frequent byte pairs until reaching
        the target vocabulary size.
        
        Args:
            input_path: Path to the training text file.
        """
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

    def _pre_tokenize(self, process_chunk: str) -> tuple[defaultdict[PreToken, int], defaultdict[BytePair, int]]:
        """Pre-tokenize a text chunk into byte sequences and count pair frequencies.
        
        Splits text using GPT-2 style regex pattern, converts each token to bytes,
        and counts both individual pre-tokens and adjacent byte pairs.
        
        Args:
            process_chunk: A chunk of text to pre-tokenize.
            
        Returns:
            A tuple of (pre-token counts, byte pair counts).
        """
        counts: defaultdict[PreToken, int] = defaultdict(int)
        pairwise_counts: defaultdict[BytePair, int] = defaultdict(int)
        for chunk in tqdm(re.split(self._delimiter, process_chunk), desc="Pre-tokenizing"):
            for token in re.finditer(self._pat, chunk):
                encoded_token = token.group().encode("utf-8")
                byte_list: PreToken = tuple([bytes([elem]) for elem in encoded_token])
                counts[byte_list] += 1
                for i in range(len(byte_list) - 1):
                    pairwise_counts[(byte_list[i], byte_list[i + 1])] += 1
        return counts, pairwise_counts
    
    def _combine_pool_results(
        self, pool_results: list[tuple[defaultdict[PreToken, int], defaultdict[BytePair, int]]]
    ) -> None:
        """Combine pre-tokenization results from multiple worker processes.
        
        Aggregates pre-token counts and byte pair counts from all workers
        into the tokenizer's internal state.
        
        Args:
            pool_results: List of (pre-token counts, pair counts) from each worker.
        """
        for counts, pairwise_counts in pool_results:
            for pre_token, count in counts.items():
                self._counts[pre_token] += count

            for pair, pairwise_count in pairwise_counts.items():
                self._pairwise_count[pair] += pairwise_count
 
    def _find_merge_pair(self) -> BytePair:
        """Find the most frequent byte pair to merge next.
        
        In case of frequency ties, selects the lexicographically largest pair
        for deterministic behavior.
        
        Returns:
            The byte pair with the highest frequency.
        """
        sorted_pairs = sorted(self._pairwise_count.items(), key=lambda x: x[1], reverse=True)
        most_common_pair = sorted_pairs[0]
        for pair in sorted_pairs[1:]:
            if pair[1] == most_common_pair[1] and max(most_common_pair[0], pair[0]) == pair[0]:
                most_common_pair = pair
            if pair[1] < most_common_pair[1]:
                break
        return most_common_pair[0]
        
    def _find_match_locations(self, pre_token: PreToken, pair: BytePair) -> list[int]:
        """Find all non-overlapping locations where a byte pair occurs in a pre-token.
        
        Scans through the pre-token and records indices where the pair matches,
        skipping the next position after each match to avoid overlapping matches.
        
        Args:
            pre_token: A tuple of byte tokens to search within.
            pair: The byte pair to search for.
            
        Returns:
            List of indices where the pair starts in pre_token.
        """
        match_locations: list[int] = []
        last_match = False
        for i in range(len(pre_token) - 1):
            if last_match:  # Avoid consecutive matches
                last_match = False
                continue
            if pre_token[i] == pair[0] and pre_token[i + 1] == pair[1]:
                match_locations.append(i)
                last_match = True
        return match_locations

    def _update_pairwise_counts(
        self,
        pre_token: PreToken,
        pair: BytePair,
        merged_pair: ByteToken,
        match_locations: list[int],
        count: int,
    ) -> None:
        """Update byte pair frequency counts after merging a pair.
        
        For each merge location, updates counts for neighboring pairs:
        - Decrements old pairs formed with the original pair elements
        - Increments new pairs formed with the merged token
        
        Args:
            pre_token: The pre-token containing the pair to merge.
            pair: The byte pair being merged.
            merged_pair: The new merged byte token.
            match_locations: Indices where the pair occurs in pre_token.
            count: Frequency count of this pre-token in the corpus.
        """
        for match_location in match_locations:
            # Update counts for element after the match
            if match_location < len(pre_token) - len(pair):
                elem_after_match = pre_token[match_location + len(pair)]
                self._pairwise_count[(merged_pair, elem_after_match)] += count

                old_key = (pair[1], elem_after_match)
                self._pairwise_count[old_key] -= count
                if self._pairwise_count[old_key] <= 0:
                    self._pairwise_count.pop(old_key)

            # Update counts for element before the match
            if match_location > 0:
                elem_before_match = pre_token[match_location - 1]
                self._pairwise_count[(elem_before_match, merged_pair)] += count

                old_key = (elem_before_match, pair[0])
                self._pairwise_count[old_key] -= count
                if self._pairwise_count[old_key] <= 0:
                    self._pairwise_count.pop(old_key)

    def _merge_pair(self, pair: BytePair) -> None:
        """Execute a single BPE merge operation.
        
        Merges the given byte pair into a new token, updates the vocabulary,
        and adjusts all internal counts to reflect the merge.
        
        Args:
            pair: The byte pair to merge into a single token.
        """
        vocab_size = len(self._vocab)
        merged_pair: ByteToken = pair[0] + pair[1]

        self._merges.append(pair)
        self._vocab[vocab_size] = merged_pair
        self._pairwise_count.pop(pair)

        to_update: list[tuple[PreToken, PreToken]] = []
        for pre_token, count in self._counts.items():
            match_locations = self._find_match_locations(pre_token, pair)

            if not match_locations:
                continue

            self._update_pairwise_counts(pre_token, pair, merged_pair, match_locations, count)
            update_token = self._update_pre_token(pre_token, match_locations)
            to_update.append((pre_token, update_token))

        for old, new in to_update:
            self._counts[new] += self._counts.pop(old)

    def _update_pre_token(self, pre_token: PreToken, match_locations: list[int]) -> PreToken:
        """Create a new pre-token with merged pairs at the given locations.
        
        Replaces each pair at the match locations with a single merged token.
        Adjusts indices as merges reduce the tuple length.
        
        Args:
            pre_token: The original pre-token tuple.
            match_locations: Indices where pairs should be merged.
            
        Returns:
            A new pre-token with the pairs merged.
        """
        updated_token = pre_token
        for idx, location in enumerate(match_locations):
            match_location = location - idx
            updated_token = (
                updated_token[:match_location]
                + (updated_token[match_location] + updated_token[match_location + 1],)
                + updated_token[match_location + 2:]
            )
        return updated_token


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str] = ["<|endoftext|>"]
) -> tuple[dict[int, bytes], list[BytePair]]:
    """Train a BPE tokenizer and return the vocabulary and merge operations.
    
    Args:
        input_path: Path to the training text file.
        vocab_size: Target vocabulary size.
        special_tokens: List of special tokens to include.
        
    Returns:
        A tuple of (vocabulary dict, list of merge operations).
    """
    bpe_trainer = BPETrainer(vocab_size, special_tokens)
    bpe_trainer.fit(input_path)
    vocab = bpe_trainer.vocab
    merges = bpe_trainer.merges
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
    train_bpe(input_path, 10000)
    print(f"Runtime: {time.time()-start_time}")
