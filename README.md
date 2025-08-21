# Bloom Filters

Two implementations in Mojo. Uses xxHash.

## What is a Bloom Filter?

A Bloom filter is a space-efficient probabilistic data structure designed to test whether an element is a member of a set. It can tell you if an item is *definitely not* in the set or *might be* in the set. The key properties:

- **No false negatives**: If the filter says an item is not present, it's definitely not there
- **Possible false positives**: If the filter says an item is present, it might be wrong
- **Space efficient**: Uses much less memory than storing the actual items
- **Fast operations**: O(k) time for both add and contains, where k is the number of hash functions
- **Fixed size**: Must know approximate number of items in advance

Common use cases include web crawlers (detecting already-visited URLs), databases (avoiding unnecessary disk lookups), and distributed systems (reducing network calls).

## About This Implementation

This project provides two Bloom filter variants:

1. **Standard Bloom Filter** - Traditional implementation with enhanced double hashing
2. **Split Block Bloom Filter** - SIMD-optimized variant for better cache performance

Inspired by:
- [bits-and-blooms/bloom](https://github.com/bits-and-blooms/bloom/tree/master) - High-performance Go implementation used by Milvus and Beego
- [Split Block Bloom Filters](https://arxiv.org/pdf/2101.01719) - Research on cache-efficient Bloom filter design

## Install

```
git clone https://github.com/seif/mojo-bloomfilter.git
cd mojo-bloomfilter
```

## Use

Standard:
```mojo
from standard import StandardBloomFilter

var bf = StandardBloomFilter.create_for_fpr(10000, 0.01)
bf.add(data)
if bf.contains(data):
    print("maybe there")
```

Split block (SIMD):
```mojo
from splitblock import SplitBlockBloomFilter

var sbf = SplitBlockBloomFilter.create_for_fpr(10000, 0.01)
sbf.add(data)
if sbf.contains(data):
    print("maybe there")
```

## Choose

* **Standard**: predictable memory, smaller datasets.

* **Split block**: cache efficient, large datasets, SIMD.

Both have the same interface.

## API

### Create
- `create_for_fpr(n, fpr)` - for target false positive rate
- `create_for_bpv(n, bpv)` - for bits per value

### Use
- `add(data)` - add element
- `contains(data)` - check element
- `add_many(items)` - add multiple
- `contains_many(items)` - check multiple

### Manage
- `merge(other)` - merge filters
- `intersect(other)` - intersect filters
- `clear()` - reset
- `serialize()` - to bytes
- `deserialize(data)` - from bytes

### Monitor
- `bits_set()` - count set bits
- `fill_ratio()` - fill percentage
- `estimated_cardinality()` - estimate unique items
- `current_fpr()` - actual false positive rate
- `should_rotate(target_fpr)` - check if needs replacement

## Test

```
magic run mojo test_bloomfilters.mojo
```

## Benchmark

```
magic run mojo benchmark_comparison.mojo
```

## License

BSD 2-Clause License
