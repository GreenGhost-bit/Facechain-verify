"""Optional Ed25519 operator signature over the ``record_hash``.

The blockchain anchor proves *when* a finding existed and that it has not
changed. A signature additionally proves *who* produced it: the operator signs
the 32-byte ``record_hash`` with an Ed25519 private key, and anyone can check it
against the public key carried in ``signature.json``.

Requires the ``cryptography`` package (``pip install -e '.[sign]'``). Every
function raises :class:`~facechain.errors.FaceChainError` with a clear message if
it is absent, so the pipeline degrades gracefully when ``--sign`` is not used.
"""

from __future__ import annotations

import binascii
import contextlib
from pathlib import Path
from typing import Any

from .errors import ConfigError

_MISSING = (
    "the 'cryptography' package is required for operator signatures; "
    "install it with: pip install -e '.[sign]'"
)


def _ed25519() -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ConfigError(_MISSING) from exc
    return ed25519


def generate_keypair() -> tuple[str, str]:
    """Return ``(private_key_hex, public_key_hex)`` for a fresh Ed25519 key."""
    ed = _ed25519()
    from cryptography.hazmat.primitives import serialization

    priv = ed.Ed25519PrivateKey.generate()
    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return priv_raw.hex(), pub_raw.hex()


def load_or_create_key(path: str | Path) -> tuple[str, str, bool]:
    """Load a hex private key from ``path`` (creating it if absent).

    Returns ``(private_hex, public_hex, created)``.
    """
    ed = _ed25519()
    p = Path(path)
    if p.is_file():
        priv_hex = p.read_text("utf-8").strip()
        try:
            priv = ed.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
        except (ValueError, binascii.Error) as exc:
            raise ConfigError(f"{p} is not a valid hex Ed25519 private key") from exc
        from cryptography.hazmat.primitives import serialization

        pub_hex = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        ).hex()
        return priv_hex, pub_hex, False

    priv_hex, pub_hex = generate_keypair()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(priv_hex + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):  # pragma: no cover - windows / restricted fs
        p.chmod(0o600)
    return priv_hex, pub_hex, True


def sign_record(record_hash: str, private_key_hex: str) -> dict[str, str]:
    """Sign the hex ``record_hash`` bytes; return a self-describing signature dict."""
    ed = _ed25519()
    try:
        msg = bytes.fromhex(record_hash)
    except ValueError as exc:
        raise ConfigError("record_hash must be hex") from exc
    priv = ed.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    sig = priv.sign(msg)
    pub_hex = _public_hex(priv)
    return {
        "alg": "Ed25519",
        "signed_field": "record_hash",
        "record_hash": record_hash,
        "public_key": pub_hex,
        "signature": sig.hex(),
    }


def verify_signature(sig_obj: dict[str, Any], record_hash: str) -> bool:
    """True iff ``sig_obj`` is a valid Ed25519 signature over ``record_hash``."""
    try:
        ed = _ed25519()
    except ConfigError:
        return False
    from cryptography.exceptions import InvalidSignature

    if sig_obj.get("alg") != "Ed25519" or sig_obj.get("record_hash") != record_hash:
        return False
    try:
        pub = ed.Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(sig_obj["public_key"])))
        pub.verify(bytes.fromhex(str(sig_obj["signature"])), bytes.fromhex(record_hash))
    except (KeyError, ValueError, InvalidSignature):
        return False
    return True


def _public_hex(priv: Any) -> str:
    from cryptography.hazmat.primitives import serialization

    return priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    ).hex()


def available() -> bool:
    try:
        _ed25519()
    except ConfigError:
        return False
    return True
