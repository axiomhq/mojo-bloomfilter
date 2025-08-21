"""Simplified Split Block Bloom Filter using SIMD operations."""

import math
from memory import Span
from sys.intrinsics import PrefetchOptions
import xxhash

# Constants
alias BITS_PER_WORD = 32
alias WORDS_PER_BLOCK = 8
alias BITS_PER_BLOCK = BITS_PER_WORD * WORDS_PER_BLOCK  # 256 bits
alias BYTES_PER_BLOCK = BITS_PER_BLOCK // 8  # 32 bytes


fn fast_hash(value: UInt64, scale: Int32) -> UInt64:
    return ((value >> 32) * UInt64(scale)) >> 32


struct Word(Copyable, Movable):
    var data: SIMD[DType.uint32, 8]

    fn __init__(out self):
        self.data = SIMD[DType.uint32, 8](0)

    fn __copyinit__(out self, existing: Self):
        self.data = existing.data

    @staticmethod
    fn make_mask(hash: UInt32) -> SIMD[DType.uint32, 8]:
        # Set eight odd constants for multiply-shift hashing
        var rehash = SIMD[DType.uint32, 8](
            0x44974D91,
            0x47B6137B,
            0xA2B7289D,
            0x8824AD5B,
            0x2DF1424B,
            0x705495C7,
            0x5C6BFB31,
            0x9EFC4947,
        )

        # Broadcast the hash to all lanes
        var hash_data = SIMD[DType.uint32, 8](hash)

        # Multiply each lane by its rehash constant
        hash_data = hash_data * rehash

        # Shift right to get 5-bit indices (0-31)
        hash_data = hash_data >> (32 - 5)

        # Create masks by shifting 1 left by the hash values
        # This is equivalent to _mm256_sllv_epi32(ones, hash_data)
        var result = SIMD[DType.uint32, 8](0)
        for i in range(8):
            var shift_amount = (
                Int(hash_data[i]) & 31
            )  # Ensure shift is in [0, 31]
            result[i] = UInt32(1) << shift_amount

        return result

    fn add(mut self, hash: UInt64):
        """Add a hash to the word."""
        var result = Word.make_mask(UInt32(hash & 0xFFFFFFFF))
        self.data = self.data | result

    fn contains(self, hash: UInt64) -> Bool:
        """Check if a hash is in the word."""
        var mask = Word.make_mask(UInt32(hash & 0xFFFFFFFF))
        var test = (self.data & mask) == mask
        # Use SIMD reduction - all lanes must be true
        # Unroll for better performance
        return (
            test[0]
            and test[1]
            and test[2]
            and test[3]
            and test[4]
            and test[5]
            and test[6]
            and test[7]
        )


struct SplitBlockBloomFilter:
    """SIMD-optimized Split Block Bloom Filter using 256-bit blocks.

    This implementation follows the C reference code using AVX2-style operations
    for efficient bit manipulation within 32-byte (256-bit) blocks.
    """

    var _num_buckets: UInt32
    var blocks: List[Word]  # Each block is 8 x 32-bit = 256 bits

    fn __init__(out self, num_buckets: UInt32) raises:
        """Initialize with specified number of 256-bit buckets."""
        if num_buckets == 0:
            raise Error("Number of buckets must be greater than 0")
        if num_buckets > 1000000:  # Reasonable upper limit
            raise Error("Number of buckets exceeds reasonable limit (1M)")

        self._num_buckets = num_buckets
        self.blocks = List[Word]()

        # Initialize all blocks to zero
        for _ in range(Int(num_buckets)):
            self.blocks.append(Word())

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self._num_buckets = existing._num_buckets
        self.blocks = existing.blocks

    fn _block_index(self, hash: UInt64) -> UInt64:
        """Get the block to access within a filter with num_buckets buckets."""
        return fast_hash(hash, Int32(self._num_buckets))

    @staticmethod
    fn num_split_blocks_of(num_values: UInt, bits_per_value: UInt) -> UInt:
        """Calculate number of blocks needed for given values and bits per value.

        This matches Go's NumSplitBlocksOf function exactly.
        """
        var num_bytes = (num_values * bits_per_value + 7) // 8
        var num_blocks = (num_bytes + UInt(BYTES_PER_BLOCK - 1)) // UInt(
            BYTES_PER_BLOCK
        )
        return num_blocks

    @staticmethod
    fn create_for_bpv(n: UInt, bpv: Float64) raises -> SplitBlockBloomFilter:
        var num_blocks = SplitBlockBloomFilter.num_split_blocks_of(
            n, UInt(Int(bpv))
        )
        return SplitBlockBloomFilter(UInt32(num_blocks))

    @staticmethod
    fn create_for_fpr(n: UInt, fpr: Float64) raises -> SplitBlockBloomFilter:
        """Create filter optimized for target false positive rate.

        Args:
            n: Expected number of items.
            fpr: Target false positive rate (0-1).

        Returns:
            SplitBlockBloomFilter configured for optimal performance.
        """
        if fpr <= 0 or fpr >= 1:
            raise Error("False positive rate must be between 0 and 1")
        if n == 0:
            raise Error("Expected number of items must be greater than 0")

        # Calculate bits per value for target FPR
        # For Split Block filters, we need ~1.44 * ln(1/fpr) bits per item
        var bpv = -1.44 * math.log(fpr) / math.log(2.0)
        bpv = max(4.0, min(bpv, 20.0))  # Clamp to reasonable range

        return SplitBlockBloomFilter.create_for_bpv(n, bpv)

    fn add_hash(mut self, hash: UInt64):
        """Add a hash to the filter."""
        # Convert hash to bytes for re-hashing (unrolled)
        var bytes = List[UInt8]()
        bytes.append(UInt8(hash & 0xFF))
        bytes.append(UInt8((hash >> 8) & 0xFF))
        bytes.append(UInt8((hash >> 16) & 0xFF))
        bytes.append(UInt8((hash >> 24) & 0xFF))
        bytes.append(UInt8((hash >> 32) & 0xFF))
        bytes.append(UInt8((hash >> 40) & 0xFF))
        bytes.append(UInt8((hash >> 48) & 0xFF))
        bytes.append(UInt8((hash >> 56) & 0xFF))
        var x = xxhash.sum64(Span(bytes))
        self._add_hash(x)

    fn _add_hash(mut self, hash: UInt64):
        """Add a hash to the filter."""
        var bucket_idx = self._block_index(hash)

        # OR the mask into the existing bucket
        self.blocks[Int(bucket_idx)].add(hash)

    fn find_hash(self, hash: UInt64) -> Bool:
        """Check if a hash is in the filter."""
        var bucket_idx = self._block_index(hash)
        return self.blocks[Int(bucket_idx)].contains(hash)

    fn add(mut self, data: Span[UInt8]):
        """Add an element to the filter."""
        var hash = xxhash.sum64(data)
        self._add_hash(hash)

    fn contains(self, data: Span[UInt8]) -> Bool:
        """Check if an element might be in the set."""
        var hash = xxhash.sum64(data)
        return self.find_hash(hash)

    fn add_many(mut self, data: List[List[UInt8]]):
        """Add multiple elements to the filter."""
        for i in range(len(data)):
            var span = Span(data[i])
            self.add(span)

    fn merge(mut self, other: SplitBlockBloomFilter) raises:
        """Merge another filter into this one."""
        if self._num_buckets != other._num_buckets:
            raise Error(
                "Cannot merge filters with different bucket counts: "
                + String(self._num_buckets)
                + " != "
                + String(other._num_buckets)
            )

        for i in range(len(self.blocks)):
            self.blocks[i].data = self.blocks[i].data | other.blocks[i].data

    fn serialize(self) -> List[UInt8]:
        """Serialize the filter to bytes."""
        var result = List[UInt8]()

        # Magic header (4 bytes: "SIMD")
        result.append(0x53)  # 'S'
        result.append(0x49)  # 'I'
        result.append(0x4D)  # 'M'
        result.append(0x44)  # 'D'

        # Version (1 byte)
        result.append(1)

        # Number of buckets (4 bytes, little-endian) - unrolled
        result.append(UInt8(self._num_buckets & 0xFF))
        result.append(UInt8((self._num_buckets >> 8) & 0xFF))
        result.append(UInt8((self._num_buckets >> 16) & 0xFF))
        result.append(UInt8((self._num_buckets >> 24) & 0xFF))

        # Block data (32 bytes per block)
        for i in range(len(self.blocks)):
            var block = self.blocks[i]
            # Unroll inner loops for better performance
            # Process all 8 words in the block
            for j in range(WORDS_PER_BLOCK):
                var value = UInt32(block.data[j])
                # Unroll byte extraction (4 bytes per word)
                result.append(UInt8(value & 0xFF))
                result.append(UInt8((value >> 8) & 0xFF))
                result.append(UInt8((value >> 16) & 0xFF))
                result.append(UInt8((value >> 24) & 0xFF))

        return result

    @staticmethod
    fn deserialize(data: List[UInt8]) raises -> SplitBlockBloomFilter:
        """Deserialize bytes back into a SIMD split block bloom filter."""
        # Validate minimum size (header + version + num_buckets = 9 bytes)
        if len(data) < 9:
            raise Error("Invalid data size")

        # Validate magic header
        if (
            data[0] != 0x53
            or data[1] != 0x49
            or data[2] != 0x4D
            or data[3] != 0x44
        ):
            raise Error("Invalid magic header")

        # Check version
        if data[4] != 1:
            raise Error("Invalid version")

        # Deserialize number of buckets (unrolled)
        var num_buckets: UInt32 = (
            UInt32(data[5])
            | (UInt32(data[6]) << 8)
            | (UInt32(data[7]) << 16)
            | (UInt32(data[8]) << 24)
        )

        # Calculate expected data size
        var expected_bytes = 9 + Int(num_buckets) * BYTES_PER_BLOCK
        if len(data) < expected_bytes:
            raise Error("Invalid data")

        # Create filter and load block data
        var result = SplitBlockBloomFilter(num_buckets)

        for block_idx in range(Int(num_buckets)):
            var block_data = SIMD[DType.uint32, WORDS_PER_BLOCK](0)
            var offset = 9 + block_idx * BYTES_PER_BLOCK

            # Unroll word reconstruction
            for j in range(WORDS_PER_BLOCK):
                var value_offset = offset + j * 4
                # Unroll byte reconstruction (4 bytes per word)
                var value: UInt32 = (
                    UInt32(data[value_offset])
                    | (UInt32(data[value_offset + 1]) << 8)
                    | (UInt32(data[value_offset + 2]) << 16)
                    | (UInt32(data[value_offset + 3]) << 24)
                )
                block_data[j] = value

            var word = Word()
            word.data = block_data
            result.blocks[block_idx] = word

        return result

    fn capacity(self) -> UInt:
        """Return the total number of bits in the filter."""
        return UInt(self._num_buckets) * 256

    fn num_buckets(self) -> UInt32:
        """Return the number of buckets."""
        return self._num_buckets

    fn fpr(self, n: UInt) -> Float64:
        """Estimate false positive rate for n inserted items using the paper's formula.
        """
        # From the paper: Split Block Bloom Filters use the formula:
        # ∑P_a(i)(1-(1-1/32)^i)^8 where P_a is Poisson with parameter a

        var num_blocks = Float64(self._num_buckets)
        var items = Float64(n)

        # Average number of items per block (parameter 'a' in the paper)
        var a = items / num_blocks

        # For practical computation, we'll use the approximation from the paper
        # that works well when a ∈ [20, 52] (corresponding to ε ∈ [0.4%, 19%])

        if a < 1.0:
            # Very low load - approximate with simple formula
            var base = 1.0 - math.exp(-a / 32.0)
            var result = base
            for _ in range(7):  # 8 total multiplications (base^8)
                result *= base
            return result

        # Use the paper's insight that split block FPR is roughly:
        # (1 - (1 - 1/32)^a)^8 for moderate to high loads
        var inner_base = 1.0 - 1.0 / 32.0
        var inner_result = inner_base
        for _ in range(Int(a) - 1):  # inner_base^a
            inner_result *= inner_base

        var prob_bit_set = 1.0 - inner_result
        var result = prob_bit_set
        for _ in range(7):  # prob_bit_set^8
            result *= prob_bit_set

        return result

    fn bits_set(self) -> Int:
        """Count number of bits set in the filter.

        Returns:
            Int: Number of bits set across all blocks.
        """
        var count = 0
        for i in range(len(self.blocks)):
            var block = self.blocks[i]
            for j in range(WORDS_PER_BLOCK):
                var word = UInt32(block.data[j])
                # Brian Kernighan's algorithm
                while word != 0:
                    word &= word - 1
                    count += 1
        return count

    fn fill_ratio(self) -> Float64:
        """Calculate the ratio of bits that are set.

        Returns:
            Float64: Fill ratio (0.0 to 1.0).
        """
        var total_bits = Int(self._num_buckets) * BITS_PER_BLOCK
        return Float64(self.bits_set()) / Float64(total_bits)

    fn estimated_cardinality(self) -> Int:
        """Estimate the number of unique items that have been added.

        Uses the formula: n ≈ -m * ln(1 - k/m) where k is bits set, m is total bits.

        Returns:
            Int: Estimated number of unique items.
        """
        var k = Float64(self.bits_set())
        var m = Float64(self._num_buckets * UInt32(BITS_PER_BLOCK))

        if k == 0:
            return 0
        if k == m:
            # Filter is full, return a large estimate
            return Int(m * 2)

        var fill = k / m
        var estimate = -m * math.log(1.0 - fill)
        return max(0, Int(estimate))

    fn clear(mut self):
        """Reset all bits to zero."""
        for i in range(len(self.blocks)):
            self.blocks[i] = Word()

    fn is_empty(self) -> Bool:
        """Check if the filter is empty (no bits set).

        Returns:
            Bool: True if no bits are set, False otherwise.
        """
        for i in range(len(self.blocks)):
            var block = self.blocks[i]
            for j in range(WORDS_PER_BLOCK):
                if block.data[j] != 0:
                    return False
        return True

    fn __eq__(self, other: Self) -> Bool:
        """Check if two filters are identical.

        Args:
            other: Another SplitBlockBloomFilter to compare with.

        Returns:
            Bool: True if filters have identical configuration and bits.
        """
        if self._num_buckets != other._num_buckets:
            return False

        for i in range(len(self.blocks)):
            var self_block = self.blocks[i]
            var other_block = other.blocks[i]
            for j in range(WORDS_PER_BLOCK):
                if self_block.data[j] != other_block.data[j]:
                    return False
        return True

    fn contains_many(self, data: List[List[UInt8]]) -> List[Bool]:
        """Check multiple items for membership efficiently.

        Args:
            data: List of items to check.

        Returns:
            List[Bool]: Results for each item (True if possibly in set).
        """
        var results = List[Bool]()
        for i in range(len(data)):
            var span = Span(data[i])
            results.append(self.contains(span))
        return results

    fn current_fpr(self) -> Float64:
        """Calculate the actual false positive rate based on current fill ratio.

        Returns:
            Float64: Current false positive rate.
        """
        var fill = self.fill_ratio()
        # For Split Block filters: FPR ≈ (fill)^8 for each hash function
        var fpr = fill
        for _ in range(7):  # fill^8
            fpr *= fill
        return min(1.0, fpr)

    fn should_rotate(self, target_fpr: Float64) -> Bool:
        """Check if the filter should be rotated based on target FPR.

        Args:
            target_fpr: Target false positive rate.

        Returns:
            Bool: True if current FPR exceeds target by 2x.
        """
        return self.current_fpr() > target_fpr * 2

    fn optimal_n(self) -> Int:
        """Get the optimal number of items for this filter configuration.

        Returns:
            Int: Optimal number of items for ~50% fill ratio.
        """
        # For Split Block filters, optimal is around 50% fill
        # Each item sets approximately 8 bits
        var total_bits = Int(self._num_buckets) * BITS_PER_BLOCK
        return total_bits // 16  # Approximation for 50% fill

    fn intersect(self, other: Self) raises -> SplitBlockBloomFilter:
        """Create a new filter representing the intersection of two filters.

        Args:
            other: Another filter to intersect with.

        Returns:
            A new filter containing the intersection.

        Raises:
            Error if filters are incompatible.
        """
        if self._num_buckets != other._num_buckets:
            raise Error(
                "Cannot intersect filters with different bucket counts: "
                + String(self._num_buckets)
                + " != "
                + String(other._num_buckets)
            )

        var result = SplitBlockBloomFilter(self._num_buckets)
        for i in range(len(self.blocks)):
            result.blocks[i].data = self.blocks[i].data & other.blocks[i].data

        return result
