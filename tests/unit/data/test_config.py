import pytest
from regnn.data.base import ReGNNDatasetConfig, PreprocessStep
from regnn.data.preprocess_fns import standardize_cols, multi_cat_to_one_hot


def test_dataset_config_valid():
    """Test valid dataset configurations"""
    # Test with string list moderators
    config = ReGNNDatasetConfig(
        focal_predictor="focal",
        controlled_predictors=["control1", "control2"],
        moderators=["mod1", "mod2"],
        outcome="outcome",
        df_dtypes={},
        rename_dict={},
        preprocess_steps=[],
    )
    assert config.focal_predictor == "focal"

    # Test with nested list moderators
    config = ReGNNDatasetConfig(
        focal_predictor="focal",
        controlled_predictors=["control1"],
        moderators=[["mod1", "mod2"], ["mod3", "mod4"]],
        outcome="outcome",
        df_dtypes={},
        rename_dict={},
        preprocess_steps=[],
    )
    assert config.focal_predictor == "focal"


def test_dataset_config_with_preprocessing():
    """Test config with preprocessing steps"""
    preprocess_steps = [
        PreprocessStep(
            columns=["col1", "col2"],
            function=standardize_cols,
        ),
        PreprocessStep(
            columns=["cat1", "cat2"],
            function=multi_cat_to_one_hot,
        ),
    ]

    config = ReGNNDatasetConfig(
        focal_predictor="focal",
        controlled_predictors=["control1", "control2"],
        moderators=["mod1", "mod2"],
        outcome="outcome",
        df_dtypes={},
        rename_dict={},
        preprocess_steps=preprocess_steps,
    )

    assert len(config.preprocess_steps) == 2
    assert config.preprocess_steps[0].columns == ["col1", "col2"]
    assert config.preprocess_steps[1].columns == ["cat1", "cat2"]
    assert config.preprocess_steps[0].function == standardize_cols
    assert config.preprocess_steps[1].function == multi_cat_to_one_hot
    # Check that reverse functions were automatically set
    assert (
        config.preprocess_steps[0].reverse_function
        == standardize_cols._reverse_transform
    )
    assert (
        config.preprocess_steps[1].reverse_function
        == multi_cat_to_one_hot._reverse_transform
    )
    # Check that reverse_transform_info was initialized as empty dict
    assert config.preprocess_steps[0].reverse_transform_info == {}
    assert config.preprocess_steps[1].reverse_transform_info == {}


def test_dataset_config_invalid_moderators():
    """Test invalid moderator configurations"""
    # Test with single moderator
    with pytest.raises(ValueError, match="Must have at least 2 moderators"):
        ReGNNDatasetConfig(
            focal_predictor="focal",
            controlled_predictors=["control1"],
            moderators=["mod1"],
            outcome="outcome",
            df_dtypes={},
            rename_dict={},
            preprocess_steps=[],
        )

    # Test with empty nested moderators
    with pytest.raises(ValueError, match="Must have at least 2 moderators"):
        ReGNNDatasetConfig(
            focal_predictor="focal",
            controlled_predictors=["control1"],
            moderators=[["mod1"], []],
            outcome="outcome",
            df_dtypes={},
            rename_dict={},
            preprocess_steps=[],
        )


def test_dataset_config_invalid_strings():
    """Test invalid string configurations"""
    # Test empty focal predictor
    with pytest.raises(ValueError, match="Cannot be empty or whitespace"):
        ReGNNDatasetConfig(
            focal_predictor="",
            controlled_predictors=["control1"],
            moderators=["mod1", "mod2"],
            outcome="outcome",
            df_dtypes={},
            rename_dict={},
            preprocess_steps=[],
        )

    # Test whitespace focal predictor
    with pytest.raises(ValueError, match="Cannot be empty or whitespace"):
        ReGNNDatasetConfig(
            focal_predictor="   ",
            controlled_predictors=["control1"],
            moderators=["mod1", "mod2"],
            outcome="outcome",
            df_dtypes={},
            rename_dict={},
            preprocess_steps=[],
        )

    # Test empty outcome
    with pytest.raises(ValueError, match="Cannot be empty or whitespace"):
        ReGNNDatasetConfig(
            focal_predictor="focal",
            controlled_predictors=["control1"],
            moderators=["mod1", "mod2"],
            outcome="",
            df_dtypes={},
            rename_dict={},
            preprocess_steps=[],
        )


def test_dataset_config_invalid_controlled_predictors():
    """Test invalid controlled predictors configurations"""
    # Test empty controlled predictors list
    with pytest.raises(ValueError, match="Must have at least one controlled predictor"):
        ReGNNDatasetConfig(
            focal_predictor="focal",
            controlled_predictors=[],
            moderators=["mod1", "mod2"],
            outcome="outcome",
            df_dtypes={},
            rename_dict={},
            preprocess_steps=[],
        )

    # Test controlled predictors with empty string
    with pytest.raises(
        ValueError, match="Controlled predictors cannot contain empty strings"
    ):
        ReGNNDatasetConfig(
            focal_predictor="focal",
            controlled_predictors=["control1", ""],
            moderators=["mod1", "mod2"],
            outcome="outcome",
            df_dtypes={},
            rename_dict={},
            preprocess_steps=[],
        )

    # Test controlled predictors with whitespace
    with pytest.raises(
        ValueError, match="Controlled predictors cannot contain empty strings"
    ):
        ReGNNDatasetConfig(
            focal_predictor="focal",
            controlled_predictors=["control1", "   "],
            moderators=["mod1", "mod2"],
            outcome="outcome",
            df_dtypes={},
            rename_dict={},
            preprocess_steps=[],
        )
