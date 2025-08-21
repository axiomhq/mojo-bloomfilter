"""xxHash64."""

from memory import Span

# xxHash64 constants
alias PRIME1: UInt64 = 0x9E3779B185EBCA87
alias PRIME2: UInt64 = 0xC2B2AE3D27D4EB4F
alias PRIME3: UInt64 = 0x165667B19E3779F9
alias PRIME4: UInt64 = 0x85EBCA77C2B2AE63
alias PRIME5: UInt64 = 0x27D4EB2F165667C5

# Pre-computed operations
alias PRIME1_PLUS_2: UInt64 = 0x60EA27EEADC0B5D6  # prime1 + prime2
alias NEG_PRIME1: UInt64 = 0x61C8864E7A143579  # -prime1


@always_inline
fn rol1(x: UInt64) -> UInt64:
    return (x << 1) | (x >> 63)


@always_inline
fn rol7(x: UInt64) -> UInt64:
    return (x << 7) | (x >> 57)


@always_inline
fn rol11(x: UInt64) -> UInt64:
    return (x << 11) | (x >> 53)


@always_inline
fn rol12(x: UInt64) -> UInt64:
    return (x << 12) | (x >> 52)


@always_inline
fn rol18(x: UInt64) -> UInt64:
    return (x << 18) | (x >> 46)


@always_inline
fn rol23(x: UInt64) -> UInt64:
    return (x << 23) | (x >> 41)


@always_inline
fn rol27(x: UInt64) -> UInt64:
    return (x << 27) | (x >> 37)


@always_inline
fn rol31(x: UInt64) -> UInt64:
    return (x << 31) | (x >> 33)


@always_inline
fn avalanche(h: UInt64) -> UInt64:
    var result = h
    result ^= result >> 33
    result *= PRIME2
    result ^= result >> 29
    result *= PRIME3
    result ^= result >> 32
    return result


@always_inline
fn round(acc: UInt64, input: UInt64) -> UInt64:
    var result = acc + input * PRIME2
    result = rol31(result)
    result *= PRIME1
    return result


@always_inline
fn merge_round(acc: UInt64, val: UInt64) -> UInt64:
    var processed_val = round(0, val)
    var result = acc ^ processed_val
    result = result * PRIME1 + PRIME4
    return result


@always_inline
fn u64_from_bytes(data: Span[UInt8], offset: Int) -> UInt64:
    """Extract UInt64 from bytes in little-endian format."""
    return (
        UInt64(data[offset])
        | (UInt64(data[offset + 1]) << 8)
        | (UInt64(data[offset + 2]) << 16)
        | (UInt64(data[offset + 3]) << 24)
        | (UInt64(data[offset + 4]) << 32)
        | (UInt64(data[offset + 5]) << 40)
        | (UInt64(data[offset + 6]) << 48)
        | (UInt64(data[offset + 7]) << 56)
    )


@always_inline
fn u32_from_bytes(data: Span[UInt8], offset: Int) -> UInt32:
    """Extract UInt32 from bytes in little-endian format."""
    return (
        UInt32(data[offset])
        | (UInt32(data[offset + 1]) << 8)
        | (UInt32(data[offset + 2]) << 16)
        | (UInt32(data[offset + 3]) << 24)
    )


fn sum64(data: Span[UInt8]) -> UInt64:
    """Hash arbitrary byte data."""
    var len = len(data)
    var h: UInt64
    var p = 0  # Current position

    if len >= 32:
        var v1 = PRIME1_PLUS_2
        var v2 = PRIME2
        var v3: UInt64 = 0
        var v4 = NEG_PRIME1

        # Process 32-byte chunks
        while p <= len - 32:
            v1 = round(v1, u64_from_bytes(data, p))
            v2 = round(v2, u64_from_bytes(data, p + 8))
            v3 = round(v3, u64_from_bytes(data, p + 16))
            v4 = round(v4, u64_from_bytes(data, p + 24))
            p += 32

        h = rol1(v1) + rol7(v2) + rol12(v3) + rol18(v4)
        h = merge_round(h, v1)
        h = merge_round(h, v2)
        h = merge_round(h, v3)
        h = merge_round(h, v4)
    else:
        h = PRIME5

    h += UInt64(len)

    # Process remaining 8-byte chunks
    while p <= len - 8:
        h ^= round(0, u64_from_bytes(data, p))
        h = rol27(h) * PRIME1 + PRIME4
        p += 8

    # Process remaining 4-byte chunks
    while p <= len - 4:
        h ^= UInt64(u32_from_bytes(data, p)) * PRIME1
        h = rol23(h) * PRIME2 + PRIME3
        p += 4

    # Process remaining bytes
    while p < len:
        h ^= UInt64(data[p]) * PRIME5
        h = rol11(h) * PRIME1
        p += 1

    return avalanche(h)
