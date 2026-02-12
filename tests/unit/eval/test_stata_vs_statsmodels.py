"""Cross-validation tests: Stata vs statsmodels regression results.

Runs the **same** regression on the **same** data through both the Stata and
statsmodels backends and asserts that the numerical results agree within a
tight tolerance.

All tests are **skipped** automatically when Stata is not available (e.g. in
CI environments without a Stata licence).
"""

import numpy as np
import pandas as pd
import pytest

from regnn.data.base import DataFrameReadInConfig
from regnn.probe.dataclass.regression import ModeratedRegressionConfig

# Import both backend runners used by the probe
from regnn.probe.functions.regression_eval_probes import (
    _run_stata_regression,
    _run_statsmodels_regression,
    generate_stata_command,
)


# ---------------------------------------------------------------------------
# Stata availability check
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stata():
    """Initialise Stata; skip the entire module when unavailable."""
    try:
        from regnn.eval.utils import init_stata
        return init_stata()
    except Exception as exc:
        pytest.skip(f"Stata not available: {exc}")


# ---------------------------------------------------------------------------
# Shared data factories
# ---------------------------------------------------------------------------

_N = 500
_SEED = 42


def _make_continuous_data(seed: int = _SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "outcome": rng.randn(_N),
        "focal": rng.randn(_N),
        "mod1": rng.randn(_N),
        "mod2": rng.randn(_N),
        "ctrl1": rng.randn(_N),
        "regnn_index": rng.randn(_N),
    })
    # Deterministic relationship so R² is non-trivial
    df["outcome"] = (
        2.0 * df["focal"]
        + 1.5 * df["focal"] * df["regnn_index"]
        + 0.5 * df["ctrl1"]
        + 0.3 * df["mod1"]
        + 0.1 * df["mod2"]
        + rng.randn(_N) * 0.5
    )
    return df


def _make_binary_data(seed: int = _SEED) -> pd.DataFrame:
    """Continuous + one binary predictor."""
    rng = np.random.RandomState(seed)
    df = _make_continuous_data(seed)
    df["binary_x"] = rng.choice([0, 1], size=_N).astype(float)
    return df


def _make_weighted_data(seed: int = _SEED) -> pd.DataFrame:
    """Continuous data with probability weights."""
    rng = np.random.RandomState(seed)
    df = _make_continuous_data(seed)
    df["weight"] = rng.uniform(0.5, 2.0, size=_N)
    return df


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------


def _continuous_configs():
    read_cfg = DataFrameReadInConfig(
        data_path="dummy.csv",
        read_cols=["outcome", "focal", "mod1", "mod2", "ctrl1", "regnn_index"],
        rename_dict={},
        binary_cols=[],
        categorical_cols=[],
        ordinal_cols=[],
        continuous_cols=["outcome", "focal", "mod1", "mod2", "ctrl1", "regnn_index"],
        survey_weight_col=None,
    )
    reg_cfg = ModeratedRegressionConfig(
        focal_predictor="focal",
        outcome_col="outcome",
        controlled_cols=["ctrl1"],
        moderators=["mod1", "mod2"],
        control_moderators=True,
        index_column_name="regnn_index",
    )
    return read_cfg, reg_cfg


def _binary_configs():
    read_cfg = DataFrameReadInConfig(
        data_path="dummy.csv",
        read_cols=[
            "outcome", "focal", "mod1", "mod2", "ctrl1",
            "regnn_index", "binary_x",
        ],
        rename_dict={},
        binary_cols=["binary_x"],
        categorical_cols=[],
        ordinal_cols=[],
        continuous_cols=[
            "outcome", "focal", "mod1", "mod2", "ctrl1", "regnn_index",
        ],
        survey_weight_col=None,
    )
    reg_cfg = ModeratedRegressionConfig(
        focal_predictor="focal",
        outcome_col="outcome",
        controlled_cols=["ctrl1", "binary_x"],
        moderators=["mod1", "mod2"],
        control_moderators=True,
        index_column_name="regnn_index",
    )
    return read_cfg, reg_cfg


def _weighted_configs():
    read_cfg = DataFrameReadInConfig(
        data_path="dummy.csv",
        read_cols=[
            "outcome", "focal", "mod1", "mod2", "ctrl1",
            "regnn_index", "weight",
        ],
        rename_dict={},
        binary_cols=[],
        categorical_cols=[],
        ordinal_cols=[],
        continuous_cols=[
            "outcome", "focal", "mod1", "mod2", "ctrl1",
            "regnn_index", "weight",
        ],
        survey_weight_col="weight",
    )
    reg_cfg = ModeratedRegressionConfig(
        focal_predictor="focal",
        outcome_col="outcome",
        controlled_cols=["ctrl1"],
        moderators=["mod1", "mod2"],
        control_moderators=True,
        index_column_name="regnn_index",
    )
    return read_cfg, reg_cfg


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _reorder_stata_to_statsmodels(stata_vec: list) -> list:
    """Reorder a Stata coefficient/SE/pvalue vector to match statsmodels order.

    Stata orders variables as they appear in the command, with ``_cons`` last.
    statsmodels (via ``add_constant``) places ``const`` first.

    Stata:       [focal, interaction, ctrl1, mod1, mod2, ..., _cons]
    statsmodels: [const, focal, interaction, ctrl1, mod1, mod2, ...]
    """
    return [stata_vec[-1]] + stata_vec[:-1]


def _assert_results_close(
    stata_result,
    sm_result,
    *,
    rtol_scalar: float = 1e-4,
    rtol_coefs: float = 1e-4,
    rtol_pvals: float = 1e-3,
):
    """Assert that Stata and statsmodels probe results agree numerically."""

    # --- Both must have succeeded ---
    assert stata_result.status == "success", f"Stata failed: {stata_result.message}"
    assert sm_result.status == "success", f"statsmodels failed: {sm_result.message}"

    # --- Scalar statistics ---
    assert sm_result.rsquared == pytest.approx(
        stata_result.rsquared, rel=rtol_scalar
    ), f"R² mismatch: SM={sm_result.rsquared} vs Stata={stata_result.rsquared}"

    assert sm_result.adjusted_rsquared == pytest.approx(
        stata_result.adjusted_rsquared, rel=rtol_scalar
    ), f"Adj R² mismatch: SM={sm_result.adjusted_rsquared} vs Stata={stata_result.adjusted_rsquared}"

    assert sm_result.rmse == pytest.approx(
        stata_result.rmse, rel=rtol_scalar
    ), f"RMSE mismatch: SM={sm_result.rmse} vs Stata={stata_result.rmse}"

    assert sm_result.n_observations == stata_result.n_observations, (
        f"N mismatch: SM={sm_result.n_observations} vs Stata={stata_result.n_observations}"
    )

    # --- Interaction p-value (the key metric the probe tracks) ---
    assert sm_result.interaction_pval == pytest.approx(
        stata_result.interaction_pval, rel=rtol_pvals
    ), f"Interaction p-val mismatch: SM={sm_result.interaction_pval} vs Stata={stata_result.interaction_pval}"

    # --- Coefficient vectors (reorder Stata → statsmodels order) ---
    if stata_result.coefficients is not None and sm_result.coefficients is not None:
        stata_coefs = _reorder_stata_to_statsmodels(list(stata_result.coefficients))
        sm_coefs = list(sm_result.coefficients)
        assert len(sm_coefs) == len(stata_coefs), (
            f"Coefficient count mismatch: SM={len(sm_coefs)} vs Stata={len(stata_coefs)}"
        )
        np.testing.assert_allclose(
            sm_coefs, stata_coefs, rtol=rtol_coefs,
            err_msg="Coefficient vectors differ",
        )

    # --- Standard errors (reorder Stata → statsmodels order) ---
    if stata_result.standard_errors is not None and sm_result.standard_errors is not None:
        stata_se = _reorder_stata_to_statsmodels(list(stata_result.standard_errors))
        sm_se = list(sm_result.standard_errors)
        np.testing.assert_allclose(
            sm_se, stata_se, rtol=rtol_coefs,
            err_msg="Standard error vectors differ",
        )

    # --- P-values (reorder Stata → statsmodels order) ---
    if stata_result.p_values is not None and sm_result.p_values is not None:
        stata_pv = _reorder_stata_to_statsmodels(list(stata_result.p_values))
        sm_pv = list(sm_result.p_values)
        np.testing.assert_allclose(
            sm_pv, stata_pv, rtol=rtol_pvals,
            err_msg="P-value vectors differ",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStataVsStatsmodels:
    """Compare Stata and statsmodels regression results on identical data."""

    # -- Unweighted OLS, all continuous -----------------------------------

    def test_ols_continuous(self, stata):
        """Unweighted OLS with all-continuous predictors — should match
        to very high precision since both solve the same normal equations."""
        df = _make_continuous_data()
        read_cfg, reg_cfg = _continuous_configs()

        stata_result = _run_stata_regression(
            df_eval=df,
            regress_cmd=None,  # auto-generate
            data_readin_config=read_cfg,
            regression_config=reg_cfg,
            data_source_name="TEST",
        )
        sm_result = _run_statsmodels_regression(
            df_eval=df,
            regression_config=reg_cfg,
            data_readin_config=read_cfg,
            data_source_name="TEST",
        )

        _assert_results_close(
            stata_result, sm_result,
            rtol_scalar=1e-6,
            rtol_coefs=1e-6,
            rtol_pvals=1e-4,
        )

    # -- Unweighted OLS, with a binary predictor --------------------------

    def test_ols_with_binary(self, stata):
        """Unweighted OLS with a binary control variable (Stata ``i.`` dummy
        vs pandas ``get_dummies``)."""
        df = _make_binary_data()
        read_cfg, reg_cfg = _binary_configs()

        stata_result = _run_stata_regression(
            df_eval=df,
            regress_cmd=None,
            data_readin_config=read_cfg,
            regression_config=reg_cfg,
            data_source_name="TEST",
        )
        sm_result = _run_statsmodels_regression(
            df_eval=df,
            regression_config=reg_cfg,
            data_readin_config=read_cfg,
            data_source_name="TEST",
        )

        # Scalar stats should still match tightly
        assert stata_result.status == "success"
        assert sm_result.status == "success"
        assert sm_result.rsquared == pytest.approx(stata_result.rsquared, rel=1e-6)
        assert sm_result.adjusted_rsquared == pytest.approx(
            stata_result.adjusted_rsquared, rel=1e-6
        )
        assert sm_result.rmse == pytest.approx(stata_result.rmse, rel=1e-6)
        assert sm_result.n_observations == stata_result.n_observations
        assert sm_result.interaction_pval == pytest.approx(
            stata_result.interaction_pval, rel=1e-4
        )

        # NOTE: coefficient vector ordering comparison is skipped here because
        # Stata's i.binary_x may produce extra base-level rows in r(table)
        # that differ from pandas get_dummies.  Scalar stats already confirm
        # numerical equivalence.

    # -- Weighted regression (pweight) ------------------------------------

    def test_wls_weighted(self, stata):
        """WLS with probability weights — Stata ``[pweight=…]`` uses the
        sandwich (robust) estimator, matched by statsmodels WLS + HC1.

        Point estimates (coefficients, R²) should match tightly.
        Standard errors / p-values may diverge slightly due to different
        robust-SE implementations, so we use wider tolerances.
        """
        df = _make_weighted_data()
        read_cfg, reg_cfg = _weighted_configs()

        stata_result = _run_stata_regression(
            df_eval=df,
            regress_cmd=None,
            data_readin_config=read_cfg,
            regression_config=reg_cfg,
            data_source_name="TEST",
        )
        sm_result = _run_statsmodels_regression(
            df_eval=df,
            regression_config=reg_cfg,
            data_readin_config=read_cfg,
            data_source_name="TEST",
        )

        assert stata_result.status == "success"
        assert sm_result.status == "success"

        # Point estimates should still match well
        assert sm_result.rsquared == pytest.approx(stata_result.rsquared, rel=1e-4)
        assert sm_result.adjusted_rsquared == pytest.approx(
            stata_result.adjusted_rsquared, rel=1e-4
        )
        assert sm_result.n_observations == stata_result.n_observations

        # Coefficients should match (point estimates identical for WLS)
        if stata_result.coefficients and sm_result.coefficients:
            stata_coefs = _reorder_stata_to_statsmodels(
                list(stata_result.coefficients)
            )
            sm_coefs = list(sm_result.coefficients)
            np.testing.assert_allclose(
                sm_coefs, stata_coefs, rtol=1e-4,
                err_msg="Weighted coefficients differ",
            )

        # Robust SEs / p-values: wider tolerance (different sandwich implementations)
        assert sm_result.interaction_pval == pytest.approx(
            stata_result.interaction_pval, rel=5e-2
        ), (
            f"Weighted interaction p-value mismatch: "
            f"SM={sm_result.interaction_pval:.6f} vs Stata={stata_result.interaction_pval:.6f}"
        )

    # -- Regression command auto-generation --------------------------------

    def test_auto_generated_command_matches(self, stata):
        """Verify that generate_stata_command produces a runnable command
        that Stata accepts and produces valid output."""
        df = _make_continuous_data()
        read_cfg, reg_cfg = _continuous_configs()
        cmd = generate_stata_command(read_cfg, reg_cfg)

        assert cmd.startswith("regress outcome")
        assert "c.focal" in cmd
        assert "#" in cmd  # interaction term

        # Verify it runs in Stata without error
        stata.pdataframe_to_data(df, force=True)
        stata.run(cmd, quietly=True)
        ereturns = stata.get_ereturn()
        assert ereturns is not None
        assert "e(r2)" in ereturns

    # -- Deterministic data (known coefficients) ---------------------------

    def test_known_coefficients(self, stata):
        """Fit a model with known DGP coefficients and verify both backends
        recover them accurately."""
        rng = np.random.RandomState(99)
        n = 1000
        TRUE_BETA_FOCAL = 3.0
        TRUE_BETA_INTERACTION = 2.0
        TRUE_BETA_CTRL = 1.0

        df = pd.DataFrame({
            "focal": rng.randn(n),
            "ctrl1": rng.randn(n),
            "mod1": rng.randn(n),
            "regnn_index": rng.randn(n),
        })
        noise = rng.randn(n) * 0.01  # very small noise
        df["outcome"] = (
            TRUE_BETA_FOCAL * df["focal"]
            + TRUE_BETA_INTERACTION * df["focal"] * df["regnn_index"]
            + TRUE_BETA_CTRL * df["ctrl1"]
            + noise
        )

        read_cfg = DataFrameReadInConfig(
            data_path="dummy.csv",
            read_cols=["outcome", "focal", "ctrl1", "mod1", "regnn_index"],
            rename_dict={},
            binary_cols=[],
            categorical_cols=[],
            ordinal_cols=[],
            continuous_cols=["outcome", "focal", "ctrl1", "mod1", "regnn_index"],
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

        stata_result = _run_stata_regression(
            df_eval=df, regress_cmd=None,
            data_readin_config=read_cfg, regression_config=reg_cfg,
            data_source_name="TEST",
        )
        sm_result = _run_statsmodels_regression(
            df_eval=df, regression_config=reg_cfg,
            data_readin_config=read_cfg, data_source_name="TEST",
        )

        assert stata_result.status == "success"
        assert sm_result.status == "success"

        # Both should recover the true coefficients closely
        # statsmodels order: [const, focal, focalxregnn_index, ctrl1, mod1]
        sm_coefs = sm_result.coefficients
        assert sm_coefs[1] == pytest.approx(TRUE_BETA_FOCAL, abs=0.05)
        assert sm_coefs[2] == pytest.approx(TRUE_BETA_INTERACTION, abs=0.05)
        assert sm_coefs[3] == pytest.approx(TRUE_BETA_CTRL, abs=0.05)
        assert sm_coefs[0] == pytest.approx(0.0, abs=0.05)  # intercept ≈ 0

        # Stata order: [focal, focalxindex, ctrl1, mod1, _cons]
        st_coefs = stata_result.coefficients
        assert st_coefs[0] == pytest.approx(TRUE_BETA_FOCAL, abs=0.05)
        assert st_coefs[1] == pytest.approx(TRUE_BETA_INTERACTION, abs=0.05)
        assert st_coefs[2] == pytest.approx(TRUE_BETA_CTRL, abs=0.05)
        assert st_coefs[-1] == pytest.approx(0.0, abs=0.05)  # _cons ≈ 0

        # Cross-check: both backends agree with each other
        _assert_results_close(
            stata_result, sm_result,
            rtol_scalar=1e-6,
            rtol_coefs=1e-4,
            rtol_pvals=1e-3,
        )

        # R² should be near 1 (very little noise)
        assert sm_result.rsquared > 0.999
        assert stata_result.rsquared > 0.999
