"""Sanity test verifying repository configuration and package imports."""

from pathlib import Path
import yaml
import src


def test_package_version():
    assert src.__version__ == "0.1.0"


def test_config_structure():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    assert config_path.exists(), "config.yaml not found"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["project"]["name"] == "pharmacy-demand-expiry-risk"
    assert config["forecasting"]["horizon_days"] == 28
    assert config["expiry_risk"]["scoring_window_days"] == 90
    assert config["splits"]["gap_train_val_days"] == 28
    assert config["splits"]["gap_val_test_days"] == 28
