from __future__ import annotations

import pytest

from facechain.canonical import (
    FIXED_SCALE,
    canonical_bytes,
    canonical_json,
    from_fixed,
    hash_object,
    sha256_hex,
    to_fixed,
)


def test_key_order_is_stable_regardless_of_input_order():
    a = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    b = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json(a) == '{"a":2,"b":1,"c":{"y":2,"z":1}}'


def test_no_insignificant_whitespace():
    assert b" " not in canonical_bytes({"a": [1, 2, 3], "b": "x"})


def test_floats_are_rejected_everywhere():
    with pytest.raises(TypeError):
        canonical_json({"score": 0.5})
    with pytest.raises(TypeError):
        canonical_json({"nested": [{"x": 1.0}]})
    with pytest.raises(TypeError):
        canonical_json([1, 2, 3.5])


def test_non_string_keys_rejected():
    with pytest.raises(TypeError):
        canonical_json({1: "a"})


def test_unicode_is_emitted_raw_not_escaped():
    assert canonical_json({"name": "Zoë"}) == '{"name":"Zoë"}'


def test_hash_object_matches_manual_sha256():
    obj = {"a": 1, "b": [2, 3]}
    assert hash_object(obj) == sha256_hex(canonical_bytes(obj))
    # golden value pins the canonical form
    assert hash_object({"a": 1}) == sha256_hex(b'{"a":1}')


def test_fixed_point_round_trip_and_determinism():
    assert to_fixed(0.9375) == 937500
    assert to_fixed(-0.5) == -500000
    assert from_fixed(to_fixed(0.123456)) == pytest.approx(0.123456, abs=1e-6)
    assert to_fixed(1.0, FIXED_SCALE) == FIXED_SCALE


def test_record_hash_is_insensitive_to_equivalent_representations():
    assert hash_object({"x": 1, "y": 2}) == hash_object({"y": 2, "x": 1})
