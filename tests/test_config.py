import pytest

from simulator.config import check_config, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg["world"]["n_ips"] == 2000
    assert cfg["databases"]["detection_rates"]["apility"] == 0.73


def test_bad_share_is_rejected():
    cfg = load_config()
    cfg["world"]["malicious_share"] = 1.5
    with pytest.raises(ValueError):
        check_config(cfg)


def test_bad_user_range_is_rejected():
    cfg = load_config()
    cfg["world"]["users_per_shared_ip"] = [50, 5]
    with pytest.raises(ValueError):
        check_config(cfg)