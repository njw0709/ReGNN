import pytest
from regnn.data.base import DatasetConfig


def test_dataset_config_valid():
    """Test valid dataset configurations"""
    # Test with string list moderators
    config = DatasetConfig(
        focal_predictor="focal",
        controlled_predictors=["control1", "control2"],
        moderators=["mod1", "mod2"],
        outcome="outcome",
        df_dtypes={},
        rename_dict={},
    )
    assert config.focal_predictor == "focal"

    # Test with nested list moderators
    config = DatasetConfig(
        focal_predictor="focal",
        controlled_predictors=["control1"],
        moderators=[["mod1", "mod2"], ["mod3", "mod4"]],
        outcome="outcome",
        df_dtypes={},
        rename_dict={},
    )
    assert config.focal_predictor == "focal"


def test_dataset_config_invalid_moderators():
    """Test invalid moderator configurations"""
    # Test with single moderator
    with pytest.raises(ValueError, match="Must have at least 2 moderators"):
        DatasetConfig(
            focal_predictor="focal",
            controlled_predictors=["control1"],
            moderators=["mod1"],
            outcome="outcome",
        )

    # Test with empty nested moderators
    with pytest.raises(ValueError, match="Must have at least 2 moderators"):
        DatasetConfig(
            focal_predictor="focal",
            controlled_predictors=["control1"],
            moderators=[["mod1"], []],
            outcome="outcome",
        )


def test_dataset_config_invalid_strings():
    """Test invalid string configurations"""
    # Test empty focal predictor
    with pytest.raises(ValueError, match="Cannot be empty or whitespace"):
        DatasetConfig(
            focal_predictor="",
            controlled_predictors=["control1"],
            moderators=["mod1", "mod2"],
            outcome="outcome",
        )

    # Test whitespace focal predictor
    with pytest.raises(ValueError, match="Cannot be empty or whitespace"):
        DatasetConfig(
            focal_predictor="   ",
            controlled_predictors=["control1"],
            moderators=["mod1", "mod2"],
            outcome="outcome",
        )

    # Test empty outcome
    with pytest.raises(ValueError, match="Cannot be empty or whitespace"):
        DatasetConfig(
            focal_predictor="focal",
            controlled_predictors=["control1"],
            moderators=["mod1", "mod2"],
            outcome="",
        )


def test_dataset_config_invalid_controlled_predictors():
    """Test invalid controlled predictors configurations"""
    # Test empty controlled predictors list
    with pytest.raises(ValueError, match="Must have at least one controlled predictor"):
        DatasetConfig(
            focal_predictor="focal",
            controlled_predictors=[],
            moderators=["mod1", "mod2"],
            outcome="outcome",
        )

    # Test controlled predictors with empty string
    with pytest.raises(
        ValueError, match="Controlled predictors cannot contain empty strings"
    ):
        DatasetConfig(
            focal_predictor="focal",
            controlled_predictors=["control1", ""],
            moderators=["mod1", "mod2"],
            outcome="outcome",
        )

    # Test controlled predictors with whitespace
    with pytest.raises(
        ValueError, match="Controlled predictors cannot contain empty strings"
    ):
        DatasetConfig(
            focal_predictor="focal",
            controlled_predictors=["control1", "   "],
            moderators=["mod1", "mod2"],
            outcome="outcome",
        )
