"""Comprehensive unit tests for bloom filter implementations."""

import math
import random
from memory import Span, UnsafePointer
from standard import StandardBloomFilter
from splitblock import SplitBlockBloomFilter


fn test_assert(condition: Bool, message: String) raises:
    """Simple assertion helper."""
    if not condition:
        raise Error("Test failed: " + message)


fn uint8_list_from_int(value: Int) -> List[UInt8]:
    """Convert an integer to a list of bytes."""
    var result = List[UInt8]()
    var temp = value
    for _ in range(8):  # 8 bytes for Int64
        result.append(UInt8(temp & 0xFF))
        temp >>= 8
    return result


fn test_bloomfilter_basic() raises:
    """Test basic Bloom filter functionality."""
    print("Testing basic Bloom filter functionality...")

    var bf = StandardBloomFilter.create_for_fpr(1000, 0.01)

    # Test add and contains
    var test_data = List[UInt8]()
    test_data.append(42)
    test_data.append(0)
    test_data.append(0)
    test_data.append(0)
    var span = Span(test_data)

    test_assert(not bf.contains(span), "Item should not be present initially")
    bf.add(span)
    test_assert(bf.contains(span), "Item should be present after adding")

    print("✓ Basic functionality test passed")


fn test_splitblock_basic() raises:
    """Test basic Split Block Bloom filter functionality."""
    print("Testing basic Split Block Bloom filter functionality...")

    var sbf = SplitBlockBloomFilter.create_for_bpv(1000, 10.0)

    # Test add and contains
    var test_data = List[UInt8]()
    test_data.append(42)
    test_data.append(0)
    test_data.append(0)
    test_data.append(0)
    var span = Span(test_data)

    test_assert(not sbf.contains(span), "Item should not be present initially")
    sbf.add(span)
    test_assert(sbf.contains(span), "Item should be present after adding")

    print("✓ Basic Split Block functionality test passed")


fn test_fpr_comparison() raises:
    """Compare FPR across different configurations."""
    print("Comparing FPR across different bits per value...")

    var num_items = 100000  # Smaller for quick comparison
    var test_configs = List[Float64]()
    test_configs.append(8.0)
    test_configs.append(10.0)
    test_configs.append(12.0)
    test_configs.append(15.0)

    for i in range(len(test_configs)):
        var bpv = test_configs[i]
        print("\n=== Testing", bpv, "bits per value ===")

        var bf = StandardBloomFilter.create_for_bpv(num_items, bpv)
        var sbf = SplitBlockBloomFilter.create_for_bpv(num_items, bpv)

        print("Bloom filter: m =", bf.capacity(), ", k =", bf.num_hashes())
        print("Split block buckets:", sbf.num_buckets())

        # Add test items
        for j in range(num_items):
            var data = uint8_list_from_int(j)
            var span = Span(data)
            bf.add(span)
            sbf.add(span)

        # Test false positives
        var bf_fps = 0
        var sbf_fps = 0
        var test_items = 100000

        for j in range(test_items):
            var test_val = num_items + j
            var data = uint8_list_from_int(test_val)
            var span = Span(data)

            if bf.contains(span):
                bf_fps += 1
            if sbf.contains(span):
                sbf_fps += 1

        var bf_fpr = Float64(bf_fps) / Float64(test_items)
        var sbf_fpr = Float64(sbf_fps) / Float64(test_items)
        var expected_bf_fpr = bf.fpr(num_items)
        var expected_sbf_fpr = sbf.fpr(num_items)

        print("Bloom filter FPR:", bf_fpr, "(expected:", expected_bf_fpr, ")")
        print("Split Block FPR:", sbf_fpr, "(expected:", expected_sbf_fpr, ")")
        print(
            "Memory usage: BF =",
            bf.capacity() // 8,
            "bytes, SBF =",
            sbf.num_buckets() * 32,
            "bytes",
        )


fn test_false_negatives_and_fpr() raises:
    """Test with 1M random numbers - check false negatives and FPR."""
    print("Testing with 1M random numbers at 12 bits per value...")

    var num_items = 1000000
    var bpv = 12.0

    # Create both filters
    var bf = StandardBloomFilter.create_for_bpv(num_items, bpv)
    var sbf = SplitBlockBloomFilter.create_for_bpv(num_items, bpv)

    # Generate 1M random numbers and store in a simple list for tracking
    var inserted_items = List[List[UInt8]]()

    print("Generating and inserting", num_items, "random items...")
    random.seed(42)  # For reproducible results

    for i in range(num_items):
        var random_value = random.random_si64(
            0, 10000000
        )  # Large range to avoid collisions
        var int_value = Int(random_value)
        var data = uint8_list_from_int(int_value)

        inserted_items.append(data)

        var span = Span(data)
        bf.add(span)
        sbf.add(span)

        if i % 100000 == 0:
            print("Inserted", i, "items...")

    print("Checking for false negatives...")

    # Check for false negatives (should be 0)
    var bf_false_negatives = 0
    var sbf_false_negatives = 0

    for i in range(len(inserted_items)):
        var span = Span(inserted_items[i])

        if not bf.contains(span):
            bf_false_negatives += 1
        if not sbf.contains(span):
            sbf_false_negatives += 1

        if i % 100000 == 0:
            print("Checked", i, "items for false negatives...")

    test_assert(
        bf_false_negatives == 0, "Bloom filter should have no false negatives"
    )
    test_assert(
        sbf_false_negatives == 0,
        "Split Block filter should have no false negatives",
    )
    print("✓ No false negatives found")

    # Test false positive rate with 1M different items
    print("Testing false positive rate...")

    var bf_false_positives = 0
    var sbf_false_positives = 0
    var test_items = 1000000

    # Use a different range to ensure no overlap with inserted items
    for i in range(test_items):
        var test_value = 20000000 + i  # Different range from inserted items
        var data = uint8_list_from_int(test_value)
        var span = Span(data)

        if bf.contains(span):
            bf_false_positives += 1
        if sbf.contains(span):
            sbf_false_positives += 1

        if i % 100000 == 0:
            print("Tested", i, "items for false positives...")

    var bf_fpr = Float64(bf_false_positives) / Float64(test_items)
    var sbf_fpr = Float64(sbf_false_positives) / Float64(test_items)

    print("Bloom filter FPR:", bf_fpr, "False positives:", bf_false_positives)
    print(
        "Split Block filter FPR:",
        sbf_fpr,
        "False positives:",
        sbf_false_positives,
    )

    # Calculate expected FPR for 12 bits per value
    var expected_bf_fpr = bf.fpr(num_items)
    var expected_sbf_fpr = sbf.fpr(num_items)

    print("Expected Bloom filter FPR:", expected_bf_fpr)
    print("Expected Split Block filter FPR:", expected_sbf_fpr)

    # Allow some tolerance (FPR should be reasonably close to expected)
    var tolerance = 0.05  # 5% tolerance

    test_assert(
        bf_fpr <= expected_bf_fpr + tolerance, "Bloom filter FPR too high"
    )
    test_assert(
        sbf_fpr <= expected_sbf_fpr + tolerance,
        "Split Block filter FPR too high",
    )

    print("✓ False positive rates within expected bounds")


fn test_serialization() raises:
    """Test serialization and deserialization."""
    print("Testing serialization...")

    # Test Bloom filter serialization
    var bf = StandardBloomFilter.create_for_fpr(1000, 0.01)

    # Add some items
    for i in range(100):
        var data = uint8_list_from_int(i)
        var span = Span(data)
        bf.add(span)

    # Serialize
    var bf_data = bf.serialize()

    # Deserialize
    var bf_restored = StandardBloomFilter.deserialize(bf_data)

    # Check that items are still present
    for i in range(100):
        var data = uint8_list_from_int(i)
        var span = Span(data)
        test_assert(
            bf_restored.contains(span),
            "Item should be present after deserialization",
        )

    print("✓ Bloom filter serialization test passed")

    # Test Split Block filter serialization
    var sbf = SplitBlockBloomFilter.create_for_bpv(1000, 10.0)

    # Add some items
    for i in range(100):
        var data = uint8_list_from_int(i + 1000)  # Different range
        var span = Span(data)
        sbf.add(span)

    # Serialize
    var sbf_data = sbf.serialize()

    # Deserialize
    var sbf_restored = SplitBlockBloomFilter.deserialize(sbf_data)

    # Check that items are still present
    for i in range(100):
        var data = uint8_list_from_int(i + 1000)
        var span = Span(data)
        test_assert(
            sbf_restored.contains(span),
            "Item should be present after deserialization",
        )

    print("✓ Split Block filter serialization test passed")


fn test_merge_functionality() raises:
    """Test merge functionality."""
    print("Testing merge functionality...")

    # Test Bloom filter merge
    var bf1 = StandardBloomFilter.create_for_fpr(1000, 0.01)
    var bf2 = StandardBloomFilter.create_for_fpr(1000, 0.01)

    # Add different items to each filter
    for i in range(50):
        var data1 = uint8_list_from_int(i)
        var data2 = uint8_list_from_int(i + 1000)
        var span1 = Span(data1)
        var span2 = Span(data2)
        bf1.add(span1)
        bf2.add(span2)

    # Merge bf2 into bf1
    bf1.merge(bf2)

    # Check that bf1 now contains items from both filters
    for i in range(50):
        var data1 = uint8_list_from_int(i)
        var data2 = uint8_list_from_int(i + 1000)
        var span1 = Span(data1)
        var span2 = Span(data2)
        test_assert(
            bf1.contains(span1),
            "bf1 should contain its original items after merge",
        )
        test_assert(
            bf1.contains(span2), "bf1 should contain bf2's items after merge"
        )

    print("✓ Bloom filter merge test passed")

    # Test Split Block filter merge
    var sbf1 = SplitBlockBloomFilter.create_for_bpv(1000, 10.0)
    var sbf2 = SplitBlockBloomFilter.create_for_bpv(1000, 10.0)

    # Add different items to each filter
    for i in range(50):
        var data1 = uint8_list_from_int(i + 2000)
        var data2 = uint8_list_from_int(i + 3000)
        var span1 = Span(data1)
        var span2 = Span(data2)
        sbf1.add(span1)
        sbf2.add(span2)

    # Merge sbf2 into sbf1
    sbf1.merge(sbf2)

    # Check that sbf1 now contains items from both filters
    for i in range(50):
        var data1 = uint8_list_from_int(i + 2000)
        var data2 = uint8_list_from_int(i + 3000)
        var span1 = Span(data1)
        var span2 = Span(data2)
        test_assert(
            sbf1.contains(span1),
            "sbf1 should contain its original items after merge",
        )
        test_assert(
            sbf1.contains(span2), "sbf1 should contain sbf2's items after merge"
        )

    print("✓ Split Block filter merge test passed")


fn test_utility_functions() raises:
    """Test utility functions like capacity, num_buckets, etc."""
    print("Testing utility functions...")

    # Test Bloom filter utilities
    var bf = StandardBloomFilter(1000, 5)
    test_assert(bf.capacity() == 1000, "Bloom filter capacity should be 1000")
    test_assert(
        bf.num_hashes() == 5, "Bloom filter should have 5 hash functions"
    )

    # Test Split Block filter utilities
    var sbf = SplitBlockBloomFilter(10)
    test_assert(
        sbf.capacity() == 2560,
        (
            "Split Block filter with 10 buckets should have 2560 bits capacity"
            " (10 * 256)"
        ),
    )
    test_assert(
        sbf.num_buckets() == 10, "Split Block filter should have 10 buckets"
    )

    # Test parameter calculation functions
    var bf_test = StandardBloomFilter.create_for_bpv(1000, 10.0)
    test_assert(bf_test.capacity() > 0, "m should be positive")
    test_assert(bf_test.num_hashes() > 0, "k should be positive")

    var num_blocks = SplitBlockBloomFilter.num_split_blocks_of(1000, 10)
    test_assert(num_blocks > 0, "num_blocks should be positive")

    print("✓ Utility functions test passed")


fn test_add_many_functionality() raises:
    """Test add_many functionality."""
    print("Testing add_many functionality...")

    var bf = StandardBloomFilter.create_for_fpr(100, 0.01)
    var sbf = SplitBlockBloomFilter.create_for_bpv(100, 10.0)

    # Create test data
    var test_data = List[List[UInt8]]()
    for i in range(10):
        test_data.append(uint8_list_from_int(i))

    # Add many items at once
    bf.add_many(test_data)
    sbf.add_many(test_data)

    # Verify all items are present
    for i in range(10):
        var data = uint8_list_from_int(i)
        var span = Span(data)
        test_assert(bf.contains(span), "Bloom filter should contain item")
        test_assert(
            sbf.contains(span), "Split Block filter should contain item"
        )

    # Test contains_many for Bloom filter
    var check_data = List[List[UInt8]]()
    for i in range(15):  # Check 0-14, only 0-9 were added
        check_data.append(uint8_list_from_int(i))

    var results = bf.contains_many(check_data)

    # First 10 should be True
    for i in range(10):
        test_assert(results[i], "Item should be present in contains_many")

    # Items 10-14 should mostly be False (allow for false positives)
    var false_positives = 0
    for i in range(10, 15):
        if results[i]:
            false_positives += 1

    test_assert(
        false_positives < 3, "Too many false positives in contains_many"
    )

    print("✓ add_many functionality test passed")


fn test_monitoring_capabilities() raises:
    """Test the monitoring and observability features."""
    print("Testing monitoring capabilities...")

    # Create a small filter for testing
    var bf = StandardBloomFilter.create_for_fpr(1000, 0.01)

    # Test initial state
    test_assert(bf.bits_set() == 0, "Initial bits_set should be 0")
    test_assert(bf.fill_ratio() == 0.0, "Initial fill_ratio should be 0%")
    test_assert(
        bf.estimated_cardinality() == 0,
        "Initial estimated cardinality should be 0",
    )

    # Add some items and check monitoring
    for i in range(100):
        var data = uint8_list_from_int(i)
        var span = Span(data)
        bf.add(span)

    # Check bits_set increased
    var set_bits = bf.bits_set()
    test_assert(set_bits > 0, "Bits set should be > 0 after adding items")
    test_assert(set_bits < bf.capacity(), "Bits set should be < capacity")

    # Check fill_ratio is reasonable
    var ratio = bf.fill_ratio()
    test_assert(ratio > 0.0, "Fill ratio should be > 0 after adding items")
    test_assert(
        ratio < 0.5,
        "Fill ratio should be < 50% for 100 items in 1000-sized filter",
    )

    # Check estimated cardinality is close to actual
    var estimated = bf.estimated_cardinality()
    test_assert(
        estimated >= 90, "Estimated cardinality should be >= 90 (actual: 100)"
    )
    test_assert(
        estimated <= 110, "Estimated cardinality should be <= 110 (actual: 100)"
    )

    # Test clear functionality
    bf.clear()
    test_assert(bf.bits_set() == 0, "Bits set should be 0 after clear")
    test_assert(bf.fill_ratio() == 0.0, "Fill ratio should be 0 after clear")

    # Verify cleared items are gone
    var test_data = uint8_list_from_int(50)  # An item we added before
    var test_span = Span(test_data)
    test_assert(
        not bf.contains(test_span),
        "Cleared filter should not contain old items",
    )

    # Test merge with compatible and incompatible filters
    var bf2 = StandardBloomFilter.create_for_fpr(1000, 0.01)  # Same parameters
    var bf3 = StandardBloomFilter.create_for_fpr(
        1000, 0.001
    )  # Different FPR = different k

    # Test merge error handling for incompatible filters
    var merge_failed = False
    try:
        bf.merge(bf3)  # Should fail - incompatible
    except:
        merge_failed = True

    test_assert(
        merge_failed, "Merge of incompatible filters should raise error"
    )

    # Test successful merge
    bf.merge(bf2)  # Should succeed - compatible

    # Test optimized add_many
    var batch_data = List[List[UInt8]]()
    for i in range(50):
        batch_data.append(uint8_list_from_int(i + 1000))

    bf.add_many(batch_data)
    var bits_after_batch = bf.bits_set()
    test_assert(bits_after_batch > 0, "Bits set should increase after add_many")

    # Verify batch items are present
    var batch_test = uint8_list_from_int(1025)
    var batch_span = Span(batch_test)
    test_assert(bf.contains(batch_span), "Batch-added items should be present")

    # Test current_fpr and should_rotate
    var bf_test = StandardBloomFilter.create_for_fpr(100, 0.01)
    test_assert(bf_test.current_fpr() == 0.0, "Empty filter should have 0 FPR")
    test_assert(
        not bf_test.should_rotate(0.01), "Empty filter should not need rotation"
    )

    # Add items to increase FPR
    for i in range(200):  # Overload the filter
        var data = uint8_list_from_int(i)
        var span = Span(data)
        bf_test.add(span)

    test_assert(
        bf_test.current_fpr() > 0.01, "Overloaded filter should have high FPR"
    )
    test_assert(
        bf_test.should_rotate(0.01), "Overloaded filter should need rotation"
    )

    print("✓ Monitoring capabilities test passed")


fn test_intersect_functionality() raises:
    """Test the intersect functionality."""
    print("Testing intersect functionality...")

    var bf1 = StandardBloomFilter.create_for_fpr(1000, 0.01)
    var bf2 = StandardBloomFilter.create_for_fpr(1000, 0.01)

    # Add 0-49 to bf1
    for i in range(50):
        var data = uint8_list_from_int(i)
        var span = Span(data)
        bf1.add(span)

    # Add 25-74 to bf2 (25-49 overlap)
    for i in range(25, 75):
        var data = uint8_list_from_int(i)
        var span = Span(data)
        bf2.add(span)

    # Create intersection
    var intersection = bf1.intersect(bf2)

    # Test that overlapping items are in intersection
    for i in range(25, 50):
        var data = uint8_list_from_int(i)
        var span = Span(data)
        test_assert(
            intersection.contains(span),
            "Overlapping item should be in intersection",
        )

    # Check that intersection has fewer bits set than either original
    test_assert(
        intersection.bits_set() < bf1.bits_set(),
        "Intersection should have fewer bits than bf1",
    )
    test_assert(
        intersection.bits_set() < bf2.bits_set(),
        "Intersection should have fewer bits than bf2",
    )

    # Test cardinality estimate is reasonable
    var intersection_card = intersection.estimated_cardinality()
    test_assert(
        intersection_card >= 20 and intersection_card <= 30,
        "Intersection cardinality should be ~25",
    )

    # Test incompatible filters raise error
    var bf3 = StandardBloomFilter.create_for_fpr(1000, 0.001)  # Different k
    var failed = False
    try:
        _ = bf1.intersect(bf3)
    except:
        failed = True

    test_assert(failed, "Intersect with incompatible filter should raise error")

    print("✓ Intersect functionality test passed")


fn test_split_block_monitoring() raises:
    """Test Split Block monitoring capabilities."""
    print("Testing Split Block monitoring...")

    var sbf = SplitBlockBloomFilter.create_for_fpr(100, 0.01)

    # Test initial state
    test_assert(sbf.bits_set() == 0, "Initial bits_set should be 0")
    test_assert(sbf.fill_ratio() == 0.0, "Initial fill_ratio should be 0")
    test_assert(
        sbf.estimated_cardinality() == 0, "Initial cardinality should be 0"
    )
    test_assert(sbf.is_empty(), "New filter should be empty")

    # Add items and verify monitoring
    for i in range(10):
        var data = uint8_list_from_int(i)
        sbf.add(Span(data))

    test_assert(sbf.bits_set() > 0, "Bits set should be > 0 after adding")
    test_assert(sbf.fill_ratio() > 0.0, "Fill ratio should be > 0 after adding")
    test_assert(sbf.estimated_cardinality() > 0, "Cardinality should be > 0")
    test_assert(not sbf.is_empty(), "Filter should not be empty after adding")

    # Test clear
    sbf.clear()
    test_assert(sbf.bits_set() == 0, "Bits set should be 0 after clear")
    test_assert(sbf.is_empty(), "Filter should be empty after clear")

    print("✓ Split Block monitoring test passed")


fn test_split_block_contains_many() raises:
    """Test Split Block contains_many functionality."""
    print("Testing Split Block contains_many...")

    var sbf = SplitBlockBloomFilter.create_for_fpr(100, 0.01)

    # Add items
    var items = List[List[UInt8]]()
    for i in range(10):
        var item = uint8_list_from_int(i)
        items.append(item)
        sbf.add(Span(item))

    # Test contains_many with all present items
    var results = sbf.contains_many(items)
    for i in range(len(results)):
        test_assert(results[i], "contains_many should find all added items")

    # Test with mix of present and absent items
    var mixed_items = List[List[UInt8]]()
    for i in range(15):
        mixed_items.append(uint8_list_from_int(i))

    var mixed_results = sbf.contains_many(mixed_items)
    for i in range(10):
        test_assert(mixed_results[i], "First 10 items should be present")

    print("✓ Split Block contains_many test passed")


fn test_split_block_equality() raises:
    """Test Split Block equality operator."""
    print("Testing Split Block equality...")

    var sbf1 = SplitBlockBloomFilter.create_for_fpr(100, 0.01)
    var sbf2 = SplitBlockBloomFilter.create_for_fpr(100, 0.01)
    var sbf3 = SplitBlockBloomFilter.create_for_fpr(200, 0.01)  # Different size

    # Empty filters with same config should be equal
    test_assert(sbf1 == sbf2, "Empty filters with same config should be equal")
    test_assert(
        not (sbf1 == sbf3), "Filters with different config should not be equal"
    )

    # Add same items to both
    for i in range(5):
        var data = uint8_list_from_int(i)
        sbf1.add(Span(data))
        sbf2.add(Span(data))

    test_assert(sbf1 == sbf2, "Filters with same items should be equal")

    # Add one more item to sbf1
    sbf1.add(Span(uint8_list_from_int(100)))
    test_assert(
        not (sbf1 == sbf2), "Filters with different items should not be equal"
    )

    print("✓ Split Block equality test passed")


fn test_split_block_intersect() raises:
    """Test Split Block intersect functionality."""
    print("Testing Split Block intersect...")

    var sbf1 = SplitBlockBloomFilter.create_for_fpr(100, 0.01)
    var sbf2 = SplitBlockBloomFilter.create_for_fpr(100, 0.01)

    # Add 0-4 to sbf1
    for i in range(5):
        sbf1.add(Span(uint8_list_from_int(i)))

    # Add 3-7 to sbf2 (3-4 overlap)
    for i in range(3, 8):
        sbf2.add(Span(uint8_list_from_int(i)))

    # Create intersection
    var intersection = sbf1.intersect(sbf2)

    # Check overlapping items are in intersection
    for i in range(3, 5):
        test_assert(
            intersection.contains(Span(uint8_list_from_int(i))),
            "Overlapping items should be in intersection",
        )

    # Check intersection has fewer bits than originals
    test_assert(
        intersection.bits_set() < sbf1.bits_set(),
        "Intersection should have fewer bits than sbf1",
    )
    test_assert(
        intersection.bits_set() < sbf2.bits_set(),
        "Intersection should have fewer bits than sbf2",
    )

    # Test incompatible filters
    var sbf3 = SplitBlockBloomFilter.create_for_fpr(200, 0.01)  # Different size
    var failed = False
    try:
        _ = sbf1.intersect(sbf3)
    except:
        failed = True
    test_assert(failed, "Intersect with incompatible filter should raise error")

    print("✓ Split Block intersect test passed")


fn test_split_block_fpr_management() raises:
    """Test Split Block FPR management functions."""
    print("Testing Split Block FPR management...")

    var sbf = SplitBlockBloomFilter.create_for_fpr(50, 0.01)

    # Test initial FPR
    test_assert(sbf.current_fpr() == 0.0, "Empty filter should have 0 FPR")
    test_assert(
        not sbf.should_rotate(0.01), "Empty filter should not need rotation"
    )

    # Test optimal_n
    var optimal = sbf.optimal_n()
    test_assert(optimal > 0, "optimal_n should be positive")

    # Add items to increase FPR
    for i in range(100):  # Overload the filter
        sbf.add(Span(uint8_list_from_int(i)))

    var current_fpr = sbf.current_fpr()
    test_assert(current_fpr > 0.0, "FPR should be > 0 after adding items")
    test_assert(current_fpr <= 1.0, "FPR should be <= 1.0")

    # Check if rotation is needed (should be true for overloaded filter)
    test_assert(
        sbf.should_rotate(0.01), "Overloaded filter should need rotation"
    )

    print("✓ Split Block FPR management test passed")


fn main() raises:
    """Run all tests."""
    print("=== Running Bloom Filter Tests ===\n")

    try:
        test_bloomfilter_basic()
        print()

        test_splitblock_basic()
        print()

        test_fpr_comparison()
        print()

        test_false_negatives_and_fpr()
        print()

        test_serialization()
        print()

        test_merge_functionality()
        print()

        test_utility_functions()
        print()

        test_add_many_functionality()
        print()

        test_monitoring_capabilities()
        print()

        test_intersect_functionality()
        print()

        test_split_block_monitoring()
        print()

        test_split_block_contains_many()
        print()

        test_split_block_equality()
        print()

        test_split_block_intersect()
        print()

        test_split_block_fpr_management()
        print()

        print("=== ALL TESTS PASSED! ===")

    except e:
        print("Test failed with error:", e)
        raise e
