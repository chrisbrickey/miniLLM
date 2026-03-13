"""Unit tests for src/dataset.py"""

from unittest.mock import MagicMock, patch
import pytest

from src.dataset import StoryDataset

# Test constants - generic, reusable values
SAMPLE_TEXT_001 = "sample text 001"
SAMPLE_TEXT_002 = "sample text 002 with more content"
SAMPLE_TEXT_003 = "sample text 003"
TEST_DELIMITER = "<|delimiter|>"
TEST_MAXLEN = 10


@pytest.fixture
def mock_tokenizer():
    """Fixture providing a mock tokenizer with predictable behavior."""
    tokenizer = MagicMock()

    # Mock encode to return token lists based on input length
    def mock_encode(text, allowed_special=None):
        # Return list of integers representing tokens
        # Use text length as a proxy for token count for testing
        return list(range(len(text)))

    tokenizer.encode.side_effect = mock_encode
    return tokenizer


@pytest.fixture
def sample_text_list():
    """Fixture providing a list of sample text entries."""
    return [SAMPLE_TEXT_001, SAMPLE_TEXT_002, SAMPLE_TEXT_003]


@pytest.fixture
def dataset_with_mock_tokenizer(mock_tokenizer, sample_text_list):
    """Fixture providing a StoryDataset instance with mocked tokenizer."""
    with patch("src.dataset.tiktoken.get_encoding", return_value=mock_tokenizer):
        # Mock the delimiter token encoding
        mock_tokenizer.encode.return_value = [999]
        delimiter_token = 999

        # Reset to normal behavior after getting delimiter
        def mock_encode(text, allowed_special=None):
            return list(range(len(text)))
        mock_tokenizer.encode.side_effect = mock_encode

        dataset = StoryDataset(
            stories=sample_text_list,
            maxlen=TEST_MAXLEN,
            delimiter=TEST_DELIMITER,
            tokenizer_name="test-tokenizer"
        )
        # Override delimiter_token since our mock changes behavior
        dataset.delimiter_token = delimiter_token

        yield dataset


class TestStoryDataset:
    """Test suite for StoryDataset class"""

    def test_len_returns_story_count(self, sample_text_list):
        """Test that __len__() returns the number of stories in the dataset"""
        with patch("src.dataset.tiktoken.get_encoding"):
            dataset = StoryDataset(
                stories=sample_text_list,
                maxlen=TEST_MAXLEN,
                delimiter=TEST_DELIMITER
            )

        assert len(dataset) == 3

    def test_getitem_tokenizes_story(self, dataset_with_mock_tokenizer):
        """Test that __getitem__() calls tokenizer and returns token list"""
        # Get first item
        result = dataset_with_mock_tokenizer[0]

        # Should return a list
        assert isinstance(result, list)
        # Should contain integers (token IDs)
        assert all(isinstance(token, int) for token in result)
        # Should be exactly maxlen long
        assert len(result) == TEST_MAXLEN

    def test_getitem_truncates_long_sequences(self, mock_tokenizer):
        """Test that sequences longer than maxlen are truncated"""
        long_text = "x" * 50  # Will produce 50 tokens with our mock

        with patch("src.dataset.tiktoken.get_encoding", return_value=mock_tokenizer):
            mock_tokenizer.encode.return_value = [999]  # For delimiter
            dataset = StoryDataset(
                stories=[long_text],
                maxlen=TEST_MAXLEN,
                delimiter=TEST_DELIMITER
            )
            # Reset mock behavior
            mock_tokenizer.encode.side_effect = lambda text, allowed_special=None: list(range(len(text)))

            result = dataset[0]

        # Should be truncated to maxlen
        assert len(result) == TEST_MAXLEN
        # Should contain the first maxlen tokens (0, 1, 2, ..., maxlen-1)
        assert result == list(range(TEST_MAXLEN))

    def test_getitem_pads_short_sequences(self, mock_tokenizer):
        """Test that sequences shorter than maxlen are padded with zeros"""
        short_text = "abc"  # Will produce 3 tokens with our mock

        with patch("src.dataset.tiktoken.get_encoding", return_value=mock_tokenizer):
            mock_tokenizer.encode.return_value = [999]  # For delimiter
            dataset = StoryDataset(
                stories=[short_text],
                maxlen=TEST_MAXLEN,
                delimiter=TEST_DELIMITER
            )
            # Reset mock behavior
            mock_tokenizer.encode.side_effect = lambda text, allowed_special=None: list(range(len(text)))

            result = dataset[0]

        # Should be padded to maxlen
        assert len(result) == TEST_MAXLEN
        # First 3 tokens should be [0, 1, 2]
        assert result[:3] == [0, 1, 2]
        # Remaining should be padding zeros
        assert result[3:] == [0] * (TEST_MAXLEN - 3)
