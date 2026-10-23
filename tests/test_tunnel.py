import base64

from deploy.tunnel import Tunnel, clean, config_text

CONF = """[Interface]
PrivateKey = aaa=
Address = 10.0.0.2/32
DNS = 1.1.1.1
PostUp = iptables -A FORWARD -j ACCEPT
Table = off

[Peer]
PublicKey = bbb=
Endpoint = 203.0.113.5:51820
AllowedIPs = 0.0.0.0/0
"""


def test_config_from_base64_env_and_kernel_keys_removed(monkeypatch, tmp_path):
    monkeypatch.setenv("WG_CONFIG_B64", base64.b64encode(CONF.encode()).decode())
    text = clean(config_text(tmp_path))
    assert "PostUp" not in text and "Table" not in text
    assert "PrivateKey = aaa=" in text and "Endpoint = 203.0.113.5:51820" in text


def test_config_from_file_on_disk(monkeypatch, tmp_path):
    monkeypatch.delenv("WG_CONFIG_B64", raising=False)
    monkeypatch.delenv("WG_CONFIG", raising=False)
    monkeypatch.delenv("WG_CONFIG_FILE", raising=False)
    (tmp_path / "wg.conf").write_text(CONF)
    assert "[Peer]" in config_text(tmp_path)


def test_no_config_means_no_tunnel(monkeypatch, tmp_path):
    for k in ("WG_CONFIG_B64", "WG_CONFIG", "WG_CONFIG_FILE"):
        monkeypatch.delenv(k, raising=False)
    status = {}
    assert Tunnel(tmp_path, status).start() is False
    assert status["tunnel"].startswith("off")
