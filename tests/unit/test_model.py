"""Unit tests for src/model.py"""

import jax.numpy as jnp
import pytest

from src.model import NanoLLM

# Test constants - generic sequence lengths
SEQ_LENGTH_SMALL = 4
SEQ_LENGTH_MEDIUM = 8
SEQ_LENGTH_LARGE = 16


class TestCausalAttentionMask:
    """Test suite for NanoLLM.causal_attention_mask() method"""

    def test_returns_lower_triangular_matrix(self):
        """Test that mask is lower triangular (1s below/on diagonal, 0s above)"""
        model = NanoLLM()
        mask = model.causal_attention_mask(SEQ_LENGTH_SMALL)

        # Verify it's lower triangular by checking upper triangle is all zeros
        for i in range(SEQ_LENGTH_SMALL):
            for j in range(SEQ_LENGTH_SMALL):
                if j > i:  # Upper triangle
                    assert mask[i, j] == 0, f"Expected 0 at position ({i}, {j}), got {mask[i, j]}"
                else:  # Lower triangle + diagonal
                    assert mask[i, j] == 1, f"Expected 1 at position ({i}, {j}), got {mask[i, j]}"

    def test_correct_shape_for_various_sequence_lengths(self):
        """Test that mask shape matches (seq_len, seq_len) for different lengths"""
        model = NanoLLM()

        test_lengths = [SEQ_LENGTH_SMALL, SEQ_LENGTH_MEDIUM, SEQ_LENGTH_LARGE]

        for seq_len in test_lengths:
            mask = model.causal_attention_mask(seq_len)
            assert mask.shape == (seq_len, seq_len), \
                f"Expected shape ({seq_len}, {seq_len}), got {mask.shape}"

    def test_mask_values_are_binary(self):
        """Test that mask contains only 0s and 1s"""
        model = NanoLLM()
        mask = model.causal_attention_mask(SEQ_LENGTH_MEDIUM)

        # Check all values are either 0 or 1
        unique_values = jnp.unique(mask)
        assert len(unique_values) <= 2, f"Expected only 0s and 1s, got {unique_values}"
        assert all(val in [0, 1] for val in unique_values), \
            f"Expected only 0s and 1s, got {unique_values}"

    def test_diagonal_elements_are_ones(self):
        """Test that diagonal elements are 1 (tokens can attend to themselves)"""
        model = NanoLLM()
        mask = model.causal_attention_mask(SEQ_LENGTH_SMALL)

        # Check diagonal is all 1s
        diagonal = jnp.diag(mask)
        assert jnp.all(diagonal == 1), \
            f"Expected all diagonal elements to be 1, got {diagonal}"
