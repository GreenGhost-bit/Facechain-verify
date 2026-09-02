"""Typed exception hierarchy.

Every failure the pipeline can produce is a subclass of :class:`FaceChainError`
and carries a stable ``.code`` string so the CLI can map it to a deterministic
exit code and callers can branch without string matching.
"""

from __future__ import annotations


class FaceChainError(Exception):
    """Base class for every expected pipeline failure."""

    code: str = "error"
    exit_code: int = 1

    def __init__(self, message: str, *, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        # Only inline short scalar details; structured detail stays on ``.detail``
        # for callers that want to persist it.
        if self.detail is None or isinstance(self.detail, (dict, list, tuple)):
            return self.message
        text = str(self.detail)
        if len(text) > 200:
            return self.message
        return f"{self.message} ({text})"


# ---- input / imaging -------------------------------------------------------
class InputError(FaceChainError):
    code = "input_error"
    exit_code = 2


class ImageDecodeError(InputError):
    code = "image_decode_error"


class ImageTooLargeError(InputError):
    code = "image_too_large"


# ---- face stage ----------------------------------------------------------
class FaceError(FaceChainError):
    code = "face_error"
    exit_code = 3


class NoFaceFoundError(FaceError):
    code = "no_face_found"


class FaceEngineUnavailableError(FaceError):
    code = "face_engine_unavailable"


# ---- search stage ------------------------------------------------------
class SearchError(FaceChainError):
    code = "search_error"
    exit_code = 4


class NoMatchFoundError(SearchError):
    """A genuine search ran but nothing cleared the similarity threshold."""

    code = "no_match_found"


class ProviderError(SearchError):
    code = "provider_error"


class UnsafeURLError(SearchError):
    """A candidate URL failed the SSRF / safety policy."""

    code = "unsafe_url"


# ---- anchor stage ----------------------------------------------------
class AnchorError(FaceChainError):
    code = "anchor_error"
    exit_code = 5


class ChainIntegrityError(AnchorError):
    """The ledger failed an internal consistency check (tamper detected)."""

    code = "chain_integrity_error"


class BackendUnavailableError(AnchorError):
    code = "anchor_backend_unavailable"


# ---- verification ----------------------------------------------------
class VerificationError(FaceChainError):
    code = "verification_failed"
    exit_code = 6


class ConfigError(FaceChainError):
    code = "config_error"
    exit_code = 2
