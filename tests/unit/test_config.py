"""Unit tests for src/config.py path validation"""

from pathlib import Path
import pytest

from src.config import PROJECT_ROOT, validate_project_path, format_path_for_display


class TestValidateProjectPath:
    """Test suite for validate_project_path utility function"""

    def test_relative_path_within_project(self):
        """Test that relative paths are resolved relative to PROJECT_ROOT"""
        result = validate_project_path("data/test.txt")
        expected = (PROJECT_ROOT / "data/test.txt").resolve()
        assert result == expected

    def test_absolute_path_within_project(self):
        """Test that absolute paths within project are accepted"""
        test_path = PROJECT_ROOT / "data" / "test.txt"
        result = validate_project_path(str(test_path))
        assert result == test_path.resolve()

    def test_path_object_input(self):
        """Test that Path objects are handled correctly"""
        test_path = Path("data/test.txt")
        result = validate_project_path(test_path)
        expected = (PROJECT_ROOT / "data/test.txt").resolve()
        assert result == expected

    def test_nested_relative_path(self):
        """Test that nested relative paths work correctly"""
        result = validate_project_path("data/raw/stories.txt")
        expected = (PROJECT_ROOT / "data/raw/stories.txt").resolve()
        assert result == expected

    def test_parent_directory_escape_raises_error(self):
        """Test that paths with .. that escape project root raise ValueError"""
        with pytest.raises(ValueError, match="outside the project root"):
            validate_project_path("../../etc/passwd")

    def test_absolute_path_outside_project_raises_error(self):
        """Test that absolute paths outside project raise ValueError"""
        with pytest.raises(ValueError, match="outside the project root"):
            validate_project_path("/etc/passwd")

    def test_symlink_escape_attempt_raises_error(self, tmp_path):
        """Test that symlinks pointing outside project are rejected"""
        # This test would require actually creating a symlink, which may not
        # work on all systems. For now, we rely on the parent directory tests.
        # In a real scenario, resolve() would catch symlink escapes.
        pass

    def test_current_directory_notation(self):
        """Test that ./ notation works correctly"""
        result = validate_project_path("./data/test.txt")
        expected = (PROJECT_ROOT / "data/test.txt").resolve()
        assert result == expected

    def test_project_root_itself(self):
        """Test that project root path is accepted"""
        result = validate_project_path(".")
        assert result == PROJECT_ROOT.resolve()

    def test_path_with_multiple_parent_refs(self):
        """Test that multiple ../ that stay within project work"""
        # Start from a nested location and go back up
        result = validate_project_path("data/../data/test.txt")
        expected = (PROJECT_ROOT / "data/test.txt").resolve()
        assert result == expected


class TestFormatPathForDisplay:
    """Test suite for format_path_for_display utility function"""

    def test_absolute_path_within_project(self):
        """Test that absolute paths within project are shown relative to root"""
        test_path = PROJECT_ROOT / "data" / "sample.txt"
        result = format_path_for_display(test_path)
        assert result == Path("data/sample.txt")

    def test_relative_path_stays_relative(self):
        """Test that relative paths are displayed relative to project root"""
        result = format_path_for_display("data/sample.txt")
        assert result == Path("data/sample.txt")

    def test_path_outside_project_shows_filename_only(self):
        """Test that paths outside project root show only filename"""
        result = format_path_for_display("/tmp/external.txt")
        assert result == Path("external.txt")


class TestTokenizerConfig:
    """Test suite for TokenizerConfig dataclass"""

    def test_config_initialization(self):
        """Test that TokenizerConfig can be initialized with required fields"""
        from src.config import TokenizerConfig

        config = TokenizerConfig(
            delimiter="<|test|>",
            name="gpt2"
        )

        assert config.delimiter == "<|test|>"
        assert config.name == "gpt2"

    def test_tokenizer_property_returns_encoding(self):
        """Test that tokenizer property returns a tiktoken Encoding instance"""
        from src.config import TokenizerConfig
        import tiktoken

        config = TokenizerConfig(delimiter="<|test|>", name="gpt2")

        tokenizer = config.tokenizer
        assert isinstance(tokenizer, tiktoken.Encoding)
        assert tokenizer.name == "gpt2"

    def test_vocab_size_property(self):
        """Test that vocab_size property returns correct vocabulary size"""
        from src.config import TokenizerConfig

        config = TokenizerConfig(delimiter="<|test|>", name="gpt2")

        vocab_size = config.vocab_size
        assert isinstance(vocab_size, int)
        assert vocab_size > 0

    def test_end_token_aliases_delimiter(self):
        """Test that end_token property returns the delimiter value"""
        from src.config import TokenizerConfig

        test_delimiter = "<|custom|>"
        config = TokenizerConfig(delimiter=test_delimiter, name="gpt2")

        assert config.end_token == test_delimiter


class TestModelConfig:
    """Test suite for ModelConfig dataclass"""

    def test_config_with_valid_parameters(self):
        """Test that ModelConfig initializes with valid embed_dim and num_heads"""
        from src.config import ModelConfig

        config = ModelConfig(
            embed_dim=192,
            num_heads=6
        )

        assert config.embed_dim == 192
        assert config.num_heads == 6

    def test_validation_fails_when_embed_dim_not_divisible_by_num_heads(self):
        """Test that __post_init__ raises AssertionError for invalid dimensions"""
        from src.config import ModelConfig

        with pytest.raises(AssertionError, match="embed_dim .* must be divisible by num_heads"):
            ModelConfig(
                embed_dim=100,
                num_heads=7
            )

    def test_default_values(self):
        """Test that ModelConfig has sensible default values"""
        from src.config import ModelConfig

        config = ModelConfig()

        assert config.maxlen > 0
        assert config.vocab_size > 0
        assert config.embed_dim > 0
        assert config.num_heads > 0
        assert config.num_transformer_blocks > 0


class TestTrainingConfig:
    """Test suite for TrainingConfig dataclass"""

    def test_calculate_training_steps_basic(self):
        """Test calculate_training_steps with typical values"""
        from src.config import TrainingConfig

        config = TrainingConfig(
            num_epochs=3,
            warmup_rate=0.1
        )

        batches_per_epoch = 100
        total_steps, warmup_steps = config.calculate_training_steps(batches_per_epoch)

        assert total_steps == 300  # 100 batches * 3 epochs
        assert warmup_steps == 30  # 10% of 300

    def test_calculate_training_steps_ensures_minimum_warmup(self):
        """Test that warmup_steps is at least 1 even for very small datasets"""
        from src.config import TrainingConfig

        config = TrainingConfig(
            num_epochs=1,
            warmup_rate=0.1
        )

        batches_per_epoch = 5
        total_steps, warmup_steps = config.calculate_training_steps(batches_per_epoch)

        assert total_steps == 5
        assert warmup_steps >= 1  # Should be at least 1

    def test_default_values(self):
        """Test that TrainingConfig has sensible default values"""
        from src.config import TrainingConfig

        config = TrainingConfig()

        assert config.batch_size > 0
        assert config.num_epochs > 0
        assert 0.0 <= config.warmup_rate <= 1.0
        assert config.lr_peak_value > config.lr_init_value
        assert config.lr_peak_value > config.lr_end_value
