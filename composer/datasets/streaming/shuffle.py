# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.typing import NDArray

from composer.datasets.streaming.format import StreamingDatasetIndex

__all__ = ['BlockCipherShuffler']


def _encrypt_round(key: int, round_num: int, plaintext: int, block_size: int) -> int:
    """Performs round_num rounds of Feistel network encryption, recursively calling _encrypt_round.

    Args:
        key (int): cipher key
        round_num (int): number of rounds to perform
        plaintext (int): plaintext message
        block_size (int): number of bits in cipher block

    Returns:
        int: ciphertext output
    """
    if round_num == 0:
        return plaintext

    half_block_size = (block_size + 1) // 2
    N = 1 << half_block_size
    upper, lower = plaintext >> half_block_size, plaintext % (N)
    gen = np.random.default_rng(key + round_num + upper)
    lower = lower ^ gen.integers(N)
    upper, lower = lower, upper
    return _encrypt_round(key, round_num - 1, (upper << half_block_size) ^ lower, block_size)


def _decrypt_round(key: int, round_num: int, ciphertext: int, block_size: int, num_rounds: int) -> int:
    """Performs round_num rounds of Feistel network decryption, recursively calling _decrypt_round.

    Args:
        key (int): cipher key
        round_num (int): index of current round
        ciphertext (int): encrypted block
        block_size (int): number of bits in cipher block
        num_rounds (int): total number of rounds in encryption

    Returns:
        int: plaintext output
    """
    if round_num > num_rounds:
        return ciphertext

    half_block_size = (block_size + 1) // 2
    N = 1 << half_block_size
    upper, lower = ciphertext >> half_block_size, ciphertext % (N)
    gen = np.random.default_rng(key + round_num + lower)
    upper = upper ^ gen.integers(N)
    upper, lower = lower, upper
    return _decrypt_round(key, round_num + 1, (upper << half_block_size) ^ lower, block_size, num_rounds)


def encrypt(key: int, value: int, num_possible_values: int) -> int:
    """Permutes the set [0, num_possible_values) ∈ Z using a four-round Feistel network
    and Numpy's random number generator for round functions. Warning: likely not cryptographically secure,
    designed to give sufficient pseudorandomness to dataset shuffling scheme.

    Args:
        key (int): Cipher key
        value (int): Message to encrypt. must be in [0, num_possible_values).
        num_possible_values (int): Size of the set of the plaintext/ciphertext space.

    """
    num_rounds = 4
    block_size = int(np.ceil(np.log2(num_possible_values)))
    ciphertext = _encrypt_round(key, num_rounds, value, block_size)
    if ciphertext < num_possible_values:
        return ciphertext
    return encrypt(key, ciphertext, num_possible_values)


def decrypt(key: int, value: int, num_possible_values: int) -> int:
    """Un-permutes the set [0, num_possible_values) ∈ Z using a four-round Feistel network
    and Numpy's random number generator for round functions. Warning: likely not cryptographically secure,
    designed to give sufficient pseudorandomness to dataset shuffling scheme.

    Args:
        key (int): Cipher key
        value (int): Message to decrypt. must be in [0, num_possible_values).
        num_possible_values (int): Size of the set of the plaintext/ciphertext space.
    """
    num_rounds = 4
    block_size = int(np.ceil(np.log2(num_possible_values)))
    plaintext = _decrypt_round(key, 1, value, block_size, num_rounds)
    if plaintext < num_possible_values:
        return plaintext
    return decrypt(key, plaintext, num_possible_values)


class BlockCipherShuffler:

    def __init__(self, cipher_key: int, index: StreamingDatasetIndex):
        """Block Cipher Shuffler takes a `cipher_key` and a `StreamingDatasetIndex` and handles utility functions
        for shuffling within shards and samples.

        Args:
            cipher_key (int): RNG seed for reproducibility
            index (StreamingDatasetIndex): Index of underlying dataset
        """
        self._cipher_key = cipher_key
        self.index = index

    def shuffle_shards(self, num_nodes: int, global_rank: int) -> NDArray[np.int64]:
        """Returns an ordering of shards such that shards are distributed evenly between nodes and shuffled randomly.

        Args:
            num_nodes (int): Number of nodes
            global_rank (int): Rank of current node

        Returns:
            NDArray[np.int64]: Shuffled shard ordering
        """
        num_shards = self.index.num_shards
        return np.array([
            encrypt(self._cipher_key, idx, num_shards)
            for idx in np.arange(num_shards)
            if idx % num_nodes == global_rank
        ])

    def shuffle_sample(self, idx: int, num_workers: int, rank: int, shuffle_buffer_size: int) -> int:
        """
        Shuffles the samples as much as possible while maintaining the shuffle_buffer_size invariant of shards
        required on the disk at once.

        Args:
            idx (int): index of sample to be shuffled
            num_workers (int): number of worker threads
            rank (int): rank of current worker thread
            shuffle_buffer_size (int): number of shards needed at once

        Returns:
            int: id of shuffled sample
        """
        if self._cipher_key is None:
            raise ValueError('shuffling is on but no seed was specified')
        idx = idx * num_workers + rank

        ### at a high level, this function does the following things:
        ### 1. calculates the group of shards that need to be present on the disk at a time
        ### 2. calculates the index of the sample relative to this group
        ### 3. maps the relative groupwise index to a global sample index

        # calculate the index of the shard group
        shard_id = self.index.sample_shards[idx]
        shard_index = decrypt(self._cipher_key, shard_id, self.index.num_shards)
        first_shard_group_index = shard_index - (shard_index % shuffle_buffer_size)
        last_shard_group_index = min(int(first_shard_group_index + shuffle_buffer_size), int(self.index.num_shards))

        # calculate the number of samples in the shard group
        samples_in_group = np.sum(self.index.shard_samples[np.arange(first_shard_group_index, last_shard_group_index)])

        # calculate how far into our shard group the shuffled sample lies
        group_key = int(self._cipher_key + first_shard_group_index)
        group_relative_sample_id = encrypt(group_key, idx % samples_in_group, samples_in_group)

        # map the shard group relative offset to a global sample offset
        relative_id_to_shard_index = []
        relative_id_to_shard_offset = []
        for shard_member_index in np.arange(first_shard_group_index, last_shard_group_index):
            relative_id_to_shard_index += [shard_member_index] * self.index.shard_samples[shard_member_index]
            relative_id_to_shard_offset += list(range(self.index.shard_samples[shard_member_index]))

        target_shard_index = relative_id_to_shard_index[group_relative_sample_id]
        shard_offset = relative_id_to_shard_offset[group_relative_sample_id]
        shard_base_offset = np.sum(self.index.shard_samples[:target_shard_index])

        return shard_base_offset + shard_offset

    def get_shard_index(self, shard_id: int) -> int:
        """Gets the index (relative position in an epoch) of a shard with a given ID

        Args:
            shard_id (int): ID of shard

        Returns:
            int: relative position in an epoch of shard
        """
        return decrypt(self._cipher_key, shard_id, self.index.num_shards)