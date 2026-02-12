"""Tests for the statsmodels-based regression evaluation functions.

Verifies that ``build_regression_design_matrix``, ``OLS_statsmodel_from_config``,
and ``VIF_statsmodel_from_config`` produce correct results with continuous,
categorical, and weighted regressions.
"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from regnn.data.base import DataFrameReadInConfig
from regnn.eval.eval import (
    OLS_statsmodel,
    OLS_statsmodel_from_config,
    VIF_statsmodel_from_config,
    build_regression_design_matrix,
)
from regnn.probe.dataclass.regression import ModeratedRegressionConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

np.random.seed(42)
_N = 200


def _make_continuous_data() -> pd.DataFrame:
    """Simple dataset with all-continuous columns plus a generated index."""
    df = pd.DataFrame(
        {
            "outcome": np.random.randn(_N),
            "focal": np.random.randn(_N),
            "mod1": np.random.randn(_N),
            "mod2": np.random.randn(_N),
            "ctrl1": np.random.randn(_N),
            "regnn_index": np.random.randn(_N),
        }
    )
    # Create a real linear relationship so R² is meaningful
    df["outcome"] = (
        2.0 * df["focal"]
        + 1.5 * df["focal"] * df["regnn_index"]
        + 0.5 * df["ctrl1"]
        + 0.3 * df["mod1"]
        + 0.1 * df["mod2"]
        + np.random.randn(_N) * 0.5
    )
    return df


def _make_categorical_data() -> pd.DataFrame:
    """Dataset with a binary column and a categorical column."""
    df = _make_continuous_data()
    df["binary_col"] = np.random.choice([0, 1], size=_N)
    df["cat_col"] = np.random.choice(["a", "b", "c"], size=_N)
    return df


def _make_weighted_data() -> pd.DataFrame:
    """Dataset with survey weights."""
    df = _make_continuous_data()
    df["weight"] = np.random.uniform(0.5, 2.0, size=_N)
    return df


@pytest.fixture
def continuous_df():
    return _make_continuous_data()


@pytest.fixture
def categorical_df():
    return _make_categorical_data()


@pytest.fixture
def weighted_df():
    return _make_weighted_data()


# --- Config fixtures (all-continuous) ---


@pytest.fixture
def continuous_read_config():
    return DataFrameReadInConfig(
        data_path="dummy.csv",
        read_cols=["outcome", "focal", "mod1", "mod2", "ctrl1", "regnn_index"],
        rename_dict={},
        binary_cols=[],
        categorical_cols=[],
        ordinal_cols=[],
        continuous_cols=["outcome", "focal", "mod1", "mod2", "ctrl1", "regnn_index"],
        survey_weight_col=None,
    )


@pytest.fixture
def continuous_regression_config():
    return ModeratedRegressionConfig(
        focal_predictor="focal",
        outcome_col="outcome",
        controlled_cols=["ctrl1"],
        moderators=["mod1", "mod2"],
        control_moderators=True,
        index_column_name="regnn_index",
    )


# --- Config fixtures (with categorical) ---


@pytest.fixture
def categorical_read_config():
    return DataFrameReadInConfig(
        data_path="dummy.csv",
        read_cols=[
            "outcome",
            "focal",
            "mod1",
            "mod2",
            "ctrl1",
            "regnn_index",
            "binary_col",
            "cat_col",
        ],
        rename_dict={},
        binary_cols=["binary_col"],
        categorical_cols=["cat_col"],
        ordinal_cols=[],
        continuous_cols=["outcome", "focal", "mod1", "mod2", "ctrl1", "regnn_index"],
        survey_weight_col=None,
    )


@pytest.fixture
def categorical_regression_config():
    return ModeratedRegressionConfig(
        focal_predictor="focal",
        outcome_col="outcome",
        controlled_cols=["ctrl1", "binary_col", "cat_col"],
        moderators=["mod1", "mod2"],
        control_moderators=True,
        index_column_name="regnn_index",
    )


# --- Config fixtures (weighted) ---


@pytest.fixture
def weighted_read_config():
    return DataFrameReadInConfig(
        data_path="dummy.csv",
        read_cols=[
            "outcome",
            "focal",
            "mod1",
            "mod2",
            "ctrl1",
            "regnn_index",
            "weight",
        ],
        rename_dict={},
        binary_cols=[],
        categorical_cols=[],
        ordinal_cols=[],
        continuous_cols=[
            "outcome",
            "focal",
            "mod1",
            "mod2",
            "ctrl1",
            "regnn_index",
            "weight",
        ],
        survey_weight_col="weight",
    )


@pytest.fixture
def weighted_regression_config():
    return ModeratedRegressionConfig(
        focal_predictor="focal",
        outcome_col="outcome",
        controlled_cols=["ctrl1"],
        moderators=["mod1", "mod2"],
        control_moderators=True,
        index_column_name="regnn_index",
    )


# ---------------------------------------------------------------------------
# Tests: build_regression_design_matrix
# ---------------------------------------------------------------------------


class TestBuildDesignMatrix:
    def test_continuous_shape(
        self, continuous_df, continuous_regression_config, continuous_read_config
    ):
        X, y, inter_cols, weights = build_regression_design_matrix(
            continuous_df, continuous_regression_config, continuous_read_config
        )
        # X should have: const + focal + interaction + ctrl1 + mod1 + mod2 = 6 cols
        assert X.shape == (_N, 6)
        assert y.shape == (_N,)
        assert weights is None
        assert len(inter_cols) == 1
        assert "const" in X.columns

    def test_continuous_column_order(
        self, continuous_df, continuous_regression_config, continuous_read_config
    ):
        X, _, inter_cols, _ = build_regression_design_matrix(
            continuous_df, continuous_regression_config, continuous_read_config
        )
        cols = list(X.columns)
        # const should be first
        assert cols[0] == "const"
        # focal should be second
        assert cols[1] == "focal"
        # interaction third
        assert cols[2] == inter_cols[0]
        assert "xregnn_index" in inter_cols[0]

    def test_interaction_values(
        self, continuous_df, continuous_regression_config, continuous_read_config
    ):
        X, _, inter_cols, _ = build_regression_design_matrix(
            continuous_df, continuous_regression_config, continuous_read_config
        )
        expected = continuous_df["focal"] * continuous_df["regnn_index"]
        np.testing.assert_array_almost_equal(
            X[inter_cols[0]].values, expected.values
        )

    def test_categorical_dummy_encoding(
        self, categorical_df, categorical_regression_config, categorical_read_config
    ):
        X, _, _, _ = build_regression_design_matrix(
            categorical_df, categorical_regression_config, categorical_read_config
        )
        cols = list(X.columns)
        # binary_col should be dummy-encoded (1 dummy for binary, drop_first)
        binary_dummies = [c for c in cols if c.startswith("binary_col")]
        assert len(binary_dummies) == 1

        # cat_col should be dummy-encoded (2 dummies for 3 categories, drop_first)
        cat_dummies = [c for c in cols if c.startswith("cat_col")]
        assert len(cat_dummies) == 2

    def test_weighted_returns_weights(
        self, weighted_df, weighted_regression_config, weighted_read_config
    ):
        _, _, _, weights = build_regression_design_matrix(
            weighted_df, weighted_regression_config, weighted_read_config
        )
        assert weights is not None
        assert len(weights) == _N
        np.testing.assert_array_almost_equal(
            weights, weighted_df["weight"].values
        )

    def test_rename_dict_resolution(self, continuous_df):
        """Column names in configs use renamed names; df has original names."""
        # Simulate: original df has 'my_outcome', config uses 'outcome'
        df = continuous_df.rename(columns={"outcome": "my_outcome"})
        read_cfg = DataFrameReadInConfig(
            data_path="dummy.csv",
            read_cols=["my_outcome", "focal", "mod1", "mod2", "ctrl1", "regnn_index"],
            rename_dict={"my_outcome": "outcome"},  # original -> renamed
            binary_cols=[],
            categorical_cols=[],
            ordinal_cols=[],
            continuous_cols=[
                "my_outcome",
                "focal",
                "mod1",
                "mod2",
                "ctrl1",
                "regnn_index",
            ],
            survey_weight_col=None,
        )
        reg_cfg = ModeratedRegressionConfig(
            focal_predictor="focal",
            outcome_col="outcome",  # uses *renamed* name
            controlled_cols=["ctrl1"],
            moderators=["mod1", "mod2"],
            control_moderators=True,
            index_column_name="regnn_index",
        )
        X, y, _, _ = build_regression_design_matrix(df, reg_cfg, read_cfg)
        assert len(y) == _N
        assert X.shape[0] == _N

    def test_multi_index_columns(self, continuous_df, continuous_read_config):
        """Test with multiple index columns."""
        df = continuous_df.copy()
        df["regnn_index2"] = np.random.randn(_N)
        reg_cfg = ModeratedRegressionConfig(
            focal_predictor="focal",
            outcome_col="outcome",
            controlled_cols=["ctrl1"],
            moderators=["mod1", "mod2"],
            control_moderators=True,
            index_column_name=["regnn_index", "regnn_index2"],
        )
        read_cfg = DataFrameReadInConfig(
            data_path="dummy.csv",
            read_cols=[
                "outcome",
                "focal",
                "mod1",
                "mod2",
                "ctrl1",
                "regnn_index",
                "regnn_index2",
            ],
            rename_dict={},
            binary_cols=[],
            categorical_cols=[],
            ordinal_cols=[],
            continuous_cols=[
                "outcome",
                "focal",
                "mod1",
                "mod2",
                "ctrl1",
                "regnn_index",
                "regnn_index2",
            ],
            survey_weight_col=None,
        )
        X, _, inter_cols, _ = build_regression_design_matrix(df, reg_cfg, read_cfg)
        assert len(inter_cols) == 2
        assert "focalxregnn_index" in inter_cols[0]
        assert "focalxregnn_index2" in inter_cols[1]


# ---------------------------------------------------------------------------
# Tests: OLS_statsmodel_from_config
# ---------------------------------------------------------------------------


class TestOLSStatsmodelFromConfig:
    def test_continuous_basic(
        self, continuous_df, continuous_regression_config, continuous_read_config
    ):
        result = OLS_statsmodel_from_config(
            continuous_df,
            continuous_regression_config,
            continuous_read_config,
            data_source="TEST",
        )
        assert result.status == "success"
        assert result.rsquared is not None
        assert 0.0 <= result.rsquared <= 1.0
        assert result.adjusted_rsquared is not None
        assert result.rmse is not None and result.rmse > 0
        assert result.n_observations == _N
        assert not math.isnan(result.interaction_pval)
        assert 0.0 <= result.interaction_pval <= 1.0

    def test_continuous_coefficients_populated(
        self, continuous_df, continuous_regression_config, continuous_read_config
    ):
        result = OLS_statsmodel_from_config(
            continuous_df,
            continuous_regression_config,
            continuous_read_config,
        )
        assert result.coefficients is not None
        assert result.standard_errors is not None
        assert result.p_values is not None
        # 6 columns in X (const + focal + interaction + ctrl1 + mod1 + mod2)
        assert len(result.coefficients) == 6
        assert len(result.standard_errors) == 6
        assert len(result.p_values) == 6

    def test_continuous_raw_summary(
        self, continuous_df, continuous_regression_config, continuous_read_config
    ):
        result = OLS_statsmodel_from_config(
            continuous_df,
            continuous_regression_config,
            continuous_read_config,
        )
        assert result.raw_summary is not None
        assert "R-squared" in result.raw_summary

    def test_matches_manual_statsmodels(
        self, continuous_df, continuous_regression_config, continuous_read_config
    ):
        """Verify results match a manually constructed statsmodels regression."""
        result = OLS_statsmodel_from_config(
            continuous_df,
            continuous_regression_config,
            continuous_read_config,
        )
        # Build manually
        df = continuous_df
        X = pd.DataFrame(
            {
                "focal": df["focal"],
                "focalxregnn_index": df["focal"] * df["regnn_index"],
                "ctrl1": df["ctrl1"],
                "mod1": df["mod1"],
                "mod2": df["mod2"],
            }
        )
        X = sm.add_constant(X)
        y = df["outcome"]
        manual = sm.OLS(y, X).fit()

        assert result.rsquared == pytest.approx(manual.rsquared, rel=1e-6)
        assert result.adjusted_rsquared == pytest.approx(manual.rsquared_adj, rel=1e-6)
        assert result.interaction_pval == pytest.approx(
            manual.pvalues["focalxregnn_index"], rel=1e-6
        )

    def test_categorical_regression(
        self, categorical_df, categorical_regression_config, categorical_read_config
    ):
        result = OLS_statsmodel_from_config(
            categorical_df,
            categorical_regression_config,
            categorical_read_config,
            data_source="TRAIN",
        )
        assert result.status == "success"
        assert result.n_observations == _N
        assert 0.0 <= result.interaction_pval <= 1.0

    def test_weighted_regression(
        self, weighted_df, weighted_regression_config, weighted_read_config
    ):
        result = OLS_statsmodel_from_config(
            weighted_df,
            weighted_regression_config,
            weighted_read_config,
        )
        assert result.status == "success"
        assert result.n_observations == _N
        assert "WLS" in result.message

    def test_summary_fallback_on_patsy_error(
        self, continuous_df, continuous_regression_config, continuous_read_config
    ):
        """When model.summary() triggers a patsy AssertionError, we should
        still get a valid result with a fallback raw_summary string."""
        from unittest.mock import patch

        result = OLS_statsmodel_from_config(
            continuous_df,
            continuous_regression_config,
            continuous_read_config,
        )
        # The normal path produces a summary — let's verify the fallback by
        # monkey-patching summary to raise the same error seen in production.
        with patch(
            "statsmodels.regression.linear_model.OLSResults.summary",
            side_effect=AssertionError("patsy DesignInfo coverage error"),
        ):
            result_fallback = OLS_statsmodel_from_config(
                continuous_df,
                continuous_regression_config,
                continuous_read_config,
            )
        assert result_fallback.status == "success"
        assert result_fallback.raw_summary is not None
        assert "R-squared" in result_fallback.raw_summary
        assert "Coef" in result_fallback.raw_summary
        # Numerical results should be identical
        assert result_fallback.rsquared == pytest.approx(result.rsquared, rel=1e-6)
        assert result_fallback.interaction_pval == pytest.approx(
            result.interaction_pval, rel=1e-6
        )

    def test_high_r_squared_with_known_data(self):
        """With a perfect linear model (no noise), R² should be ~1."""
        df = pd.DataFrame(
            {
                "focal": np.random.randn(100),
                "mod1": np.random.randn(100),
                "ctrl1": np.random.randn(100),
                "regnn_index": np.random.randn(100),
            }
        )
        # Perfect linear relationship
        df["outcome"] = (
            3.0 * df["focal"] + 2.0 * df["focal"] * df["regnn_index"] + df["ctrl1"]
        )
        read_cfg = DataFrameReadInConfig(
            data_path="dummy.csv",
            read_cols=["outcome", "focal", "mod1", "ctrl1", "regnn_index"],
            rename_dict={},
            binary_cols=[],
            categorical_cols=[],
            ordinal_cols=[],
            continuous_cols=["outcome", "focal", "mod1", "ctrl1", "regnn_index"],
            survey_weight_col=None,
        )
        reg_cfg = ModeratedRegressionConfig(
            focal_predictor="focal",
            outcome_col="outcome",
            controlled_cols=["ctrl1"],
            moderators=["mod1"],
            control_moderators=True,
            index_column_name="regnn_index",
        )
        result = OLS_statsmodel_from_config(df, reg_cfg, read_cfg)
        assert result.rsquared == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: VIF_statsmodel_from_config
# ---------------------------------------------------------------------------


class TestVIFStatsmodelFromConfig:
    def test_basic_vif(
        self, continuous_df, continuous_regression_config, continuous_read_config
    ):
        result = VIF_statsmodel_from_config(
            continuous_df,
            continuous_regression_config,
            continuous_read_config,
        )
        assert result.vif_main > 0
        assert result.vif_interaction > 0

    def test_vif_reasonable_range(
        self, continuous_df, continuous_regression_config, continuous_read_config
    ):
        """VIF for independent random variables should be low (near 1)."""
        result = VIF_statsmodel_from_config(
            continuous_df,
            continuous_regression_config,
            continuous_read_config,
        )
        # For random data, VIF should generally be < 10
        assert result.vif_main < 10
        assert result.vif_interaction < 10


# ---------------------------------------------------------------------------
# Tests: Legacy OLS_statsmodel (command-string based)
# ---------------------------------------------------------------------------


class TestLegacyOLSStatsmodel:
    def test_basic_regression(self, continuous_df):
        """Legacy function should still work with simple continuous commands."""
        result = OLS_statsmodel(
            continuous_df,
            regress_cmd="regress outcome c.focal c.ctrl1 c.mod1 c.mod2",
            data_source="TEST",
        )
        assert result.rsquared is not None
        assert 0.0 <= result.rsquared <= 1.0
        assert result.coefficients is not None
        assert result.raw_summary is not None
