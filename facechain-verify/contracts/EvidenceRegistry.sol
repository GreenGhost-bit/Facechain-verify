// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title EvidenceRegistry
/// @notice Append-only notary for facechain-verify evidence bundles. Stores the
///         first block/timestamp at which a given 32-byte record hash was seen.
///         Re-anchoring the same hash is a no-op (idempotent) and never
///         overwrites the original attestation.
contract EvidenceRegistry {
    struct Attestation {
        uint256 blockNumber;
        uint256 timestamp;
        address submitter;
    }

    mapping(bytes32 => Attestation) private _attestations;
    uint256 public count;

    event Anchored(bytes32 indexed recordHash, address indexed submitter, uint256 timestamp);

    error AlreadyAnchored(bytes32 recordHash);
    error UnknownRecord(bytes32 recordHash);

    /// @notice Notarise `recordHash`. Reverts if it was already anchored.
    function anchor(bytes32 recordHash) external {
        if (_attestations[recordHash].timestamp != 0) revert AlreadyAnchored(recordHash);
        _attestations[recordHash] = Attestation(block.number, block.timestamp, msg.sender);
        unchecked {
            count += 1;
        }
        emit Anchored(recordHash, msg.sender, block.timestamp);
    }

    /// @notice Idempotent variant: anchors only if new, returns silently otherwise.
    function anchorIfAbsent(bytes32 recordHash) external {
        if (_attestations[recordHash].timestamp != 0) return;
        _attestations[recordHash] = Attestation(block.number, block.timestamp, msg.sender);
        unchecked {
            count += 1;
        }
        emit Anchored(recordHash, msg.sender, block.timestamp);
    }

    function isAnchored(bytes32 recordHash) external view returns (bool) {
        return _attestations[recordHash].timestamp != 0;
    }

    function recordBlock(bytes32 recordHash) external view returns (uint256) {
        Attestation memory a = _attestations[recordHash];
        if (a.timestamp == 0) revert UnknownRecord(recordHash);
        return a.blockNumber;
    }

    function recordTimestamp(bytes32 recordHash) external view returns (uint256) {
        Attestation memory a = _attestations[recordHash];
        if (a.timestamp == 0) revert UnknownRecord(recordHash);
        return a.timestamp;
    }

    function attestationOf(bytes32 recordHash)
        external
        view
        returns (uint256 blockNumber, uint256 timestamp, address submitter)
    {
        Attestation memory a = _attestations[recordHash];
        if (a.timestamp == 0) revert UnknownRecord(recordHash);
        return (a.blockNumber, a.timestamp, a.submitter);
    }
}
