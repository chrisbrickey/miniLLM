"""Unit tests for src/data_loader.py"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch
import pytest

from src.data_loader import load_text_from_file
from src.config import PROJECT_ROOT

TEST_FILE_PATH = "data/test_file.txt"  # Relative to project root
DELIMITER = "<|endoftext|>"
SAMPLE_STORY_1 = "Once upon a time in a test."
SAMPLE_STORY_2 = "Another sample story for testing."
SAMPLE_STORY_3 = "Third test story goes here."


class TestLoadStoriesFromFile:
    """Test suite for load_stories_from_file utility function"""

    def test_load_single_story_with_delimiter(self, capsys):
        """Test loading a single story that ends with delimiter"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n"
        expected_path = (PROJECT_ROOT / TEST_FILE_PATH).resolve()

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_text_from_file(TEST_FILE_PATH, DELIMITER)

        assert len(stories) == 1
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"

        # Verify print output
        captured = capsys.readouterr()
        assert "Loading data" in captured.out
        assert "Loaded 1 paragraphs" in captured.out

    def test_load_multiple_stories(self, capsys):
        """Test loading multiple stories separated by delimiters"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n{SAMPLE_STORY_2}{DELIMITER}\n{SAMPLE_STORY_3}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_text_from_file(TEST_FILE_PATH, DELIMITER)

        assert len(stories) == 3
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"
        assert stories[1] == f"{SAMPLE_STORY_2}{DELIMITER}"
        assert stories[2] == f"{SAMPLE_STORY_3}{DELIMITER}"

        captured = capsys.readouterr()
        assert "Loaded 3 paragraphs" in captured.out

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised when file doesn't exist"""
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Data file not found"):
                load_text_from_file(TEST_FILE_PATH, DELIMITER)

    def test_empty_file(self, capsys):
        """Test loading from an empty file returns empty list"""
        file_content = ""

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_text_from_file(TEST_FILE_PATH, DELIMITER)

        assert stories == []

        captured = capsys.readouterr()
        assert "Loaded 0 paragraphs" in captured.out

    def test_max_stories_limit(self, capsys):
        """Test that max_paragraphs parameter limits the number of paragraphs loaded"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n{SAMPLE_STORY_2}{DELIMITER}\n{SAMPLE_STORY_3}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_text_from_file(TEST_FILE_PATH, DELIMITER, max_paragraphs=2)

        assert len(stories) == 2
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"
        assert stories[1] == f"{SAMPLE_STORY_2}{DELIMITER}"

        captured = capsys.readouterr()
        assert "Loaded 2 paragraphs" in captured.out

    def test_multiple_delimiters_on_same_line(self, capsys):
        """Test handling multiple delimiters on a single line"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}{SAMPLE_STORY_2}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_text_from_file(TEST_FILE_PATH, DELIMITER)

        assert len(stories) == 2
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"
        assert stories[1] == f"{SAMPLE_STORY_2}{DELIMITER}"

    def test_path_outside_project_raises_error(self):
        """Test that paths outside project root are rejected"""
        with pytest.raises(ValueError, match="outside the project root"):
            load_text_from_file("../../etc/passwd", DELIMITER)

    def test_absolute_path_outside_project_raises_error(self):
        """Test that absolute paths outside project are rejected"""
        with pytest.raises(ValueError, match="outside the project root"):
            load_text_from_file("/etc/passwd", DELIMITER)

    def test_print_output_shows_relative_path(self, capsys):
        """Test that printed output shows path relative to project root, not absolute"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                load_text_from_file(TEST_FILE_PATH, DELIMITER)

        captured = capsys.readouterr()
        # Should show relative path like "data/test_file.txt"
        assert TEST_FILE_PATH in captured.out
        # Should NOT show absolute path with /Users/...
        assert "/Users" not in captured.out
        assert "Loading data from data/test_file.txt" in captured.out


# Test constants for preprocess_data tests
TEST_BATCH_SIZE = 4
TEST_MAXLEN = 20
NUM_SAMPLE_PARAGRAPHS = 12


@pytest.fixture
def sample_paragraph_list():
    """Fixture providing a list of sample paragraphs for preprocessing"""
    return [f"paragraph_{i:03d}" for i in range(NUM_SAMPLE_PARAGRAPHS)]


class TestPreprocessData:
    """Test suite for preprocess_data function"""

    def test_creates_dataloader_with_valid_data(self, sample_paragraph_list, capsys):
        """Test that preprocess_data creates a dataloader with correct batch estimation"""
        from src.data_loader import preprocess_data

        with patch("src.data_loader.StoryDataset") as mock_dataset_class:
            with patch("src.data_loader.pygrain.IndexSampler") as mock_sampler:
                with patch("src.data_loader.pygrain.DataLoader") as mock_dataloader:
                    # Mock dataset to return correct length
                    mock_dataset = MagicMock()
                    mock_dataset.__len__ = MagicMock(return_value=NUM_SAMPLE_PARAGRAPHS)
                    mock_dataset_class.return_value = mock_dataset

                    dataloader, batches_per_epoch = preprocess_data(
                        list_of_paragraphs=sample_paragraph_list,
                        batch_size=TEST_BATCH_SIZE,
                        maxlen=TEST_MAXLEN,
                        delimiter=DELIMITER,
                        num_epochs=1,
                        shuffle=False,
                        seed=42
                    )

        # Verify batch calculation: 12 paragraphs / 4 batch_size = 3 batches
        expected_batches = NUM_SAMPLE_PARAGRAPHS // TEST_BATCH_SIZE
        assert batches_per_epoch == expected_batches

        # Verify dataset was created with correct parameters
        mock_dataset_class.assert_called_once_with(
            sample_paragraph_list,
            TEST_MAXLEN,
            DELIMITER
        )

        # Verify sampler was configured correctly
        mock_sampler.assert_called_once()

        # Verify print output
        captured = capsys.readouterr()
        assert "Estimated batches per epoch" in captured.out
        assert "Created DataLoader" in captured.out

    def test_empty_dataset_raises_error(self):
        """Test that preprocess_data raises ValueError for empty paragraph list"""
        from src.data_loader import preprocess_data

        empty_list = []

        with pytest.raises(ValueError, match="No valid stories found in the dataset"):
            preprocess_data(
                list_of_paragraphs=empty_list,
                batch_size=TEST_BATCH_SIZE,
                maxlen=TEST_MAXLEN,
                delimiter=DELIMITER
            )

    def test_batch_estimation_with_shuffle_enabled(self, sample_paragraph_list, capsys):
        """Test that batch estimation works correctly when shuffle is enabled"""
        from src.data_loader import preprocess_data

        with patch("src.data_loader.StoryDataset") as mock_dataset_class:
            with patch("src.data_loader.pygrain.IndexSampler") as mock_sampler:
                with patch("src.data_loader.pygrain.DataLoader") as mock_dataloader:
                    mock_dataset = MagicMock()
                    mock_dataset.__len__ = MagicMock(return_value=NUM_SAMPLE_PARAGRAPHS)
                    mock_dataset_class.return_value = mock_dataset

                    dataloader, batches_per_epoch = preprocess_data(
                        list_of_paragraphs=sample_paragraph_list,
                        batch_size=TEST_BATCH_SIZE,
                        maxlen=TEST_MAXLEN,
                        delimiter=DELIMITER,
                        shuffle=True,
                        seed=99
                    )

        # Batch estimation should be same regardless of shuffle
        assert batches_per_epoch == NUM_SAMPLE_PARAGRAPHS // TEST_BATCH_SIZE

        # Verify sampler was called with shuffle=True
        call_kwargs = mock_sampler.call_args.kwargs
        assert call_kwargs["shuffle"] is True
        assert call_kwargs["seed"] == 99
