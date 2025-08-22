"""Improved Bloom filter implementation with better hash independence."""

import math
from memory import Span
import xxhash


struct StandardBloomFilter:
    """Standard Bloom filter using enhanced double hashing for efficiency."""

    var m: UInt
    var k: UInt
    var bits: List[UInt64]  # Using 64-bit words for better performance

    fn __init__(out self, m: UInt, k: UInt) raises:
        """Initialize bloom filter with given parameters."""
        if m == 0:
            raise Error("Bloom filter size (m) must be greater than 0")
        if k == 0:
            raise Error("Number of hash functions (k) must be greater than 0")
        if k > 30:
            raise Error(
                "Number of hash functions (k) exceeds reasonable limit (30)"
            )

        self.m = m
        self.k = k
        var num_words = (m + 63) // 64  # Number of 64-bit words needed
        self.bits = List[UInt64]()
        for _ in range(Int(num_words)):
            self.bits.append(0)

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.m = existing.m
        self.k = existing.k
        self.bits = existing.bits

    fn _compute_hashes(self, data: Span[UInt8]) -> Tuple[UInt64, UInt64]:
        """Compute the two base hashes needed for double hashing.

        This should be called once per add/contains operation,
        not k times as was happening before.
        """
        # Compute primary hash
        var h1 = xxhash.sum64(data)

        # Derive secondary hash from h1 for efficiency
        # Mix h1 to get independence
        var h2 = h1 ^ 0x8E3C5B2F1A0D9E74  # XOR with large prime
        h2 = h2 * 0x9E3779B97F4A7C15  # Multiply by golden ratio
        h2 ^= h2 >> 33  # Mix bits

        return (h1, h2)

    fn _get_position(self, h1: UInt64, h2: UInt64, n: Int) -> UInt:
        """Get the n-th position using precomputed hashes.

        Uses the technique from "Less Hashing, Same Performance" paper.
        Standard double hashing: h1 + n * h2
        """
        var hash_val = h1 + UInt64(n) * h2
        return UInt(hash_val % UInt64(self.m))

    fn add(mut self, data: Span[UInt8]):
        """Add element using enhanced double hashing - zero allocations."""
        # Compute hashes once!
        h1, h2 = self._compute_hashes(data)

        # Use them k times
        for i in range(Int(self.k)):
            var pos = self._get_position(h1, h2, i)
            self.bits[Int(pos // 64)] |= UInt64(1) << UInt64(pos % 64)

    fn contains(self, data: Span[UInt8]) -> Bool:
        """Check if element might be in set using enhanced double hashing."""
        # Compute hashes once!
        h1, h2 = self._compute_hashes(data)

        # Use them k times
        for i in range(Int(self.k)):
            var pos = self._get_position(h1, h2, i)
            if self.bits[Int(pos // 64)] & (UInt64(1) << UInt64(pos % 64)) == 0:
                return False
        return True

    fn serialize(self) -> List[UInt8]:
        """Serialize the Bloom filter to bytes including parameters."""
        var result = List[UInt8]()

        # Magic header (4 bytes: "BLMF")
        result.append(0x42)  # 'B'
        result.append(0x4C)  # 'L'
        result.append(0x4D)  # 'M'
        result.append(0x46)  # 'F'

        # Parameters (8 bytes each for m and k, little-endian)
        var m_val = self.m
        var k_val = self.k

        # Serialize m (8 bytes, little-endian)
        result.append(UInt8(m_val & 0xFF))
        result.append(UInt8((m_val >> 8) & 0xFF))
        result.append(UInt8((m_val >> 16) & 0xFF))
        result.append(UInt8((m_val >> 24) & 0xFF))
        result.append(UInt8((m_val >> 32) & 0xFF))
        result.append(UInt8((m_val >> 40) & 0xFF))
        result.append(UInt8((m_val >> 48) & 0xFF))
        result.append(UInt8((m_val >> 56) & 0xFF))

        # Serialize k (8 bytes, little-endian)
        result.append(UInt8(k_val & 0xFF))
        result.append(UInt8((k_val >> 8) & 0xFF))
        result.append(UInt8((k_val >> 16) & 0xFF))
        result.append(UInt8((k_val >> 24) & 0xFF))
        result.append(UInt8((k_val >> 32) & 0xFF))
        result.append(UInt8((k_val >> 40) & 0xFF))
        result.append(UInt8((k_val >> 48) & 0xFF))
        result.append(UInt8((k_val >> 56) & 0xFF))

        # Bit data - serialize UInt64 words as bytes (little-endian)
        for i in range(len(self.bits)):
            var word = self.bits[i]
            result.append(UInt8(word & 0xFF))
            result.append(UInt8((word >> 8) & 0xFF))
            result.append(UInt8((word >> 16) & 0xFF))
            result.append(UInt8((word >> 24) & 0xFF))
            result.append(UInt8((word >> 32) & 0xFF))
            result.append(UInt8((word >> 40) & 0xFF))
            result.append(UInt8((word >> 48) & 0xFF))
            result.append(UInt8((word >> 56) & 0xFF))

        return result

    @staticmethod
    fn deserialize(data: List[UInt8]) raises -> StandardBloomFilter:
        """Deserialize bytes back into a Bloom filter."""
        # Validate minimum size (header + m + k = 20 bytes)
        if len(data) < 20:
            raise Error("Invalid data size")

        # Validate magic header
        if (
            data[0] != 0x42
            or data[1] != 0x4C
            or data[2] != 0x4D
            or data[3] != 0x46
        ):
            raise Error("Invalid magic header")

        # Deserialize m (8 bytes, little-endian)
        var m: UInt = (
            UInt(data[4])
            | (UInt(data[5]) << 8)
            | (UInt(data[6]) << 16)
            | (UInt(data[7]) << 24)
            | (UInt(data[8]) << 32)
            | (UInt(data[9]) << 40)
            | (UInt(data[10]) << 48)
            | (UInt(data[11]) << 56)
        )

        # Deserialize k (8 bytes, little-endian)
        var k: UInt = (
            UInt(data[12])
            | (UInt(data[13]) << 8)
            | (UInt(data[14]) << 16)
            | (UInt(data[15]) << 24)
            | (UInt(data[16]) << 32)
            | (UInt(data[17]) << 40)
            | (UInt(data[18]) << 48)
            | (UInt(data[19]) << 56)
        )

        # Create filter (seeds will be auto-generated)
        var result = StandardBloomFilter(m, k)

        # Calculate expected bit data size
        var expected_words = (m + 63) // 64
        var expected_bytes = expected_words * 8  # Each word is 8 bytes
        var actual_data_bytes = len(data) - 20

        if actual_data_bytes < expected_bytes:
            raise Error("Invalid data")

        # Reconstruct UInt64 words from bytes (little-endian)
        for i in range(Int(expected_words)):
            var idx = 20 + i * 8
            var word: UInt64 = (
                UInt64(data[idx])
                | (UInt64(data[idx + 1]) << 8)
                | (UInt64(data[idx + 2]) << 16)
                | (UInt64(data[idx + 3]) << 24)
                | (UInt64(data[idx + 4]) << 32)
                | (UInt64(data[idx + 5]) << 40)
                | (UInt64(data[idx + 6]) << 48)
                | (UInt64(data[idx + 7]) << 56)
            )
            result.bits[i] = word

        return result

    @staticmethod
    fn create_for_bpv(n: UInt, bpv: Float64) raises -> StandardBloomFilter:
        """Calculate optimal m and k for given bits per value."""
        var m_float = math.ceil(Float64(n) * bpv)
        var m = UInt(Int(m_float))
        var k_float = bpv * math.log(2.0)
        var k = UInt(max(1, Int(k_float + 0.5)))
        return StandardBloomFilter(m, k)

    @staticmethod
    fn create_for_fpr(n: UInt, fpr: Float64) raises -> StandardBloomFilter:
        """Create optimally sized filter for target FPR."""
        var m_float = -1.0 * Float64(n) * math.log(fpr) / (math.log(2.0) ** 2)
        var m = UInt(Int(m_float))
        var k_float = (Float64(m) / Float64(n)) * math.log(2.0)
        var k = UInt(max(1, Int(k_float + 0.5)))
        return StandardBloomFilter(m, k)

    fn fpr(self, n: UInt) -> Float64:
        """Calculate theoretical FPR."""
        if n == 0:
            return 0.0
        var prob_bit_set = 1.0 - math.exp(
            -Float64(self.k * n) / Float64(self.m)
        )
        return prob_bit_set ** Float64(self.k)

    fn add_many(mut self, data: List[List[UInt8]]):
        """Add multiple elements to the filter."""
        for i in range(len(data)):
            var span = Span(data[i])
            self.add(span)

    fn contains_many(self, data: List[List[UInt8]]) -> List[Bool]:
        """Check multiple elements for membership.

        Args:
            data: List of items to check.

        Returns:
            List[Bool]: List of results, True if item might be present.
        """
        var results = List[Bool]()
        for i in range(len(data)):
            var span = Span(data[i])
            results.append(self.contains(span))
        return results

    fn merge(mut self, other: StandardBloomFilter) raises:
        """Merge two Bloom filters (must have same m and k).

        Raises:
            Error: If filters have incompatible parameters.
        """
        if self.m != other.m:
            raise Error(
                "Cannot merge: different sizes (m="
                + String(self.m)
                + " vs "
                + String(other.m)
                + ")"
            )
        if self.k != other.k:
            raise Error(
                "Cannot merge: different hash counts (k="
                + String(self.k)
                + " vs "
                + String(other.k)
                + ")"
            )

        for i in range(len(self.bits)):
            self.bits[i] |= other.bits[i]

    fn capacity(self) -> UInt:
        """Return filter capacity in bits."""
        return self.m

    fn num_hashes(self) -> UInt:
        """Return number of hash functions."""
        return self.k

    fn bits_set(self) -> UInt:
        """Count the number of bits set in the filter.

        This is useful for monitoring filter fill ratio and estimating
        the cardinality of unique items that have been added.
        """
        var count: UInt = 0
        for i in range(len(self.bits)):
            var word = self.bits[i]
            # Brian Kernighan's algorithm for counting set bits
            while word != 0:
                word &= word - 1
                count += 1
        return count

    fn fill_ratio(self) -> Float64:
        """Calculate the ratio of bits that are set (fill ratio).

        Returns:
            Float64: Fill ratio (0.0 to 1.0).

        Note:
            - < 0.5: Good, filter is operating efficiently.
            - 0.5-0.75: Warning, approaching capacity.
            - > 0.75: Critical, high false positive rate.
        """
        if self.m == 0:
            return 0.0
        return Float64(self.bits_set()) / Float64(self.m)

    fn estimated_cardinality(self) -> UInt:
        """Estimate the cardinality (number of unique items) added to the filter.

        Uses the formula: n ≈ -(m/k) * ln(1 - X/m)
        where X is the number of set bits.

        Returns:
            UInt: Estimated cardinality of unique items added.
        """
        var set_bits = self.bits_set()
        if set_bits == 0:
            return 0
        if set_bits == self.m:
            # Filter is fully saturated, return a large estimate
            return self.m  # Conservative estimate

        var X = Float64(set_bits)
        var m_float = Float64(self.m)
        var k_float = Float64(self.k)

        # n ≈ -(m/k) * ln(1 - X/m)
        var fraction_set = X / m_float
        var estimated = -(m_float / k_float) * math.log(1.0 - fraction_set)

        return UInt(max(0, Int(estimated + 0.5)))  # Round to nearest integer

    fn clear(mut self):
        """Reset all bits to zero, clearing the filter."""
        for i in range(len(self.bits)):
            self.bits[i] = 0

    fn is_empty(self) -> Bool:
        """Check if the filter is empty (no bits set).

        Returns:
            Bool: True if no bits are set, False otherwise.
        """
        for i in range(len(self.bits)):
            if self.bits[i] != 0:
                return False
        return True

    fn __eq__(self, other: StandardBloomFilter) -> Bool:
        """Check if two filters are equal.

        Two filters are equal if they have the same parameters and bit patterns.

        Args:
            other: Another Bloom filter to compare with.

        Returns:
            Bool: True if filters are identical.
        """
        if self.m != other.m or self.k != other.k:
            return False

        for i in range(len(self.bits)):
            if self.bits[i] != other.bits[i]:
                return False

        return True

    fn optimal_n(self) -> UInt:
        """Calculate the optimal number of items for this filter configuration.

        This is the n value that would achieve optimal FPR given the
        current m and k parameters.

        Returns:
            UInt: Optimal number of items for this filter.
        """
        # Optimal n = (m * ln(2)) / k
        var m_float = Float64(self.m)
        var k_float = Float64(self.k)
        var optimal = (m_float * math.log(2.0)) / k_float
        return UInt(max(1, Int(optimal + 0.5)))

    fn current_fpr(self) -> Float64:
        """Calculate the actual false positive rate based on current fill ratio.

        This gives the real FPR based on how many bits are currently set,
        which may differ from the theoretical FPR if the actual number of
        items differs from the design capacity.

        Returns:
            Float64: Current false positive probability (0.0 to 1.0).
        """
        var ratio = self.fill_ratio()
        if ratio == 0.0:
            return 0.0
        if ratio == 1.0:
            return 1.0
        # Actual FPR = (bits_set / m) ^ k
        return ratio ** Float64(self.k)

    fn should_rotate(self, target_fpr: Float64) -> Bool:
        """Check if the filter should be rotated/replaced.

        Args:
            target_fpr: Maximum acceptable false positive rate.

        Returns:
            Bool: True if current FPR exceeds target, indicating rotation needed.
        """
        return self.current_fpr() > target_fpr

    fn intersect(
        self, other: StandardBloomFilter
    ) raises -> StandardBloomFilter:
        """Create a new filter containing the intersection of two filters.

        The resulting filter contains only items that might be in both filters.
        Note: Due to the probabilistic nature, some false positives may be included.

        Args:
            other: Another Bloom filter with same parameters.

        Returns:
            StandardBloomFilter: New filter containing the intersection.

        Raises:
            Error: If filters have incompatible parameters.
        """
        if self.m != other.m:
            raise Error("Cannot intersect: different sizes")
        if self.k != other.k:
            raise Error("Cannot intersect: different hash counts")

        # Create new filter with same parameters
        var result = StandardBloomFilter(self.m, self.k)

        # Intersection is bitwise AND of both filters
        for i in range(len(self.bits)):
            result.bits[i] = self.bits[i] & other.bits[i]

        return result
