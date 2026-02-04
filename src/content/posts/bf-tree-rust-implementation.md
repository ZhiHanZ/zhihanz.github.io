---
title: "Lessons from BF-Tree: Building a Concurrent Larger-Than-Memory Index in Rust"
date: 2025-02-04
description: "A deep dive into the Rust implementation of BF-Tree, exploring custom allocators, RAII guards, atomic state machines, and deterministic concurrency testing with Shuttle."
tags: ["rust", "concurrency", "databases", "systems-programming"]
---

**TL;DR:** BF-Tree (VLDB 2024, Microsoft Research) replaces traditional 4KB page caching with variable-size mini-pages (64-4096 bytes) to reduce write amplification and memory waste. The Rust implementation builds a custom ring buffer allocator with a 6-state machine, RAII guards, optimistic locks, and deterministic concurrency testing. This post follows a mini-page through its entire lifecycle: birth (allocation), growth (buffering writes), concurrent access (readers and writers sharing it safely), and eviction (flushing to disk and recycling memory).

---

## Opening: Why Mini-Pages?

A B-tree index works well until the dataset outgrows RAM. At that point, every write to a leaf page means loading 4KB from disk, modifying a few bytes, and writing 4KB back. This is called write amplification: the ratio of bytes written to disk versus bytes the application actually changed. For a 100-byte record update on a 4KB page, that ratio is 40x.

Each leaf page holds a sorted array of key-value records. The internal layout is a dual-growing structure: record metadata (8 bytes each) grows forward from the front of the page, while key-value payloads grow backward from the end. When the two regions meet, the page is full.

<script type="text/typogram">
+------------------------------------------------------+
| Page header (24 bytes)                               |
+------------------------------------------------------+
| [KVMeta][KVMeta]...          ...[val|key|val|key]    |
|  ------------>              <----------------------  |
|  metadata grows forward     payloads grow backward   |
+------------------------------------------------------+
</script>

This layout is efficient for both point lookups (binary search on the sorted metadata) and range scans (records are physically adjacent). The problem appears when the dataset no longer fits in memory.

Consider what happens when a user inserts a 100-byte record. The B-tree traverses inner nodes to find the right leaf page. If that page is not in memory, the system loads the full 4KB page from disk, inserts 100 bytes, and writes 4KB back. The write amplification is 40x. Do this thousands of times per second across different pages, and disk bandwidth is consumed almost entirely by reading and writing pages that are mostly unchanged.

The traditional database solution is a buffer pool: an in-memory cache of recently-used disk pages. When a page is needed, the buffer pool checks if it already holds a copy. Hit: modify in memory, mark the page dirty, write it back later. Miss: load from disk into a free frame, evict a cold page if needed.

Buffer pools reduce the number of disk I/O operations, but they have a structural problem: the cache unit is the entire 4KB page. If your workload touches 100,000 different key ranges but only accesses 2-3 records per range, the buffer pool caches 100,000 pages at 4KB each (400MB), even though the actual hot data is 100,000 records at roughly 100 bytes each (10MB). With limited RAM, this means fewer key ranges can be cached, and more requests hit disk.

The underlying problem is that real workloads are skewed. The BF-Tree paper evaluates with a Zipf distribution at skew factor 0.9, where 80% of requests access just 33% of records. A small fraction of records receives most of the traffic. But those hot records are not clustered together on a few pages; they are scattered across many pages, interspersed with cold records that nobody is reading. A disk page with 40 records might have 3 that are accessed constantly and 37 that are untouched. A buffer pool caches the entire 4KB page to serve those 3 records. Multiply this across thousands of pages and the buffer pool fills with cold data, forcing hot pages out of memory and back to disk.

The paper frames it as a fundamental tension: disk page size is much larger than record size, and this coarse-grained unit of caching limits performance. The paper offers a thought experiment: if the disk page size equaled the record size, both write amplification and cache waste would disappear. You would read, modify, and write exactly the bytes you need. Mini-pages approximate this ideal without changing the on-disk format.

BF-Tree solves both problems by decoupling the in-memory representation from the on-disk format. Instead of caching full disk pages, it places a mini-page in front of each disk page. A mini-page is a small in-memory buffer (64 to 4096 bytes) that holds only the records worth caching. It is not a copy of the disk page. It holds only the records worth caching, at whatever size is needed: 128 bytes for a key range with one recent write, 2048 bytes for a hot range accumulating hundreds of updates. For that disk page with 40 records and 3 hot ones, BF-Tree creates a mini-page of maybe 256 bytes holding just those 3 records. Writes go to the mini-page (no disk I/O). Reads check the mini-page first, then fall through to the disk page on a miss. When memory is full, dirty data from cold mini-pages is merged back to disk. The dataset can be arbitrarily large; the buffer only needs to hold the working set.

The mini-page serves three functions described in the paper:

**Buffering writes.** When a write arrives for a key range with no mini-page, BF-Tree allocates one from the circular buffer, inserts the record, and updates the page table. No disk I/O. Subsequent writes to the same range go directly to the mini-page. Only when the mini-page is evicted are its dirty records merged to the 4KB base page on disk. Ten writes to the same key range cost a few hundred bytes of buffer space, not ten 4KB page writes.

**Caching hot records.** When a read misses the mini-page and falls through to the base page on disk, BF-Tree can promote just that record into the mini-page as a clean cache entry. Clean entries can be discarded during eviction without writing anything to disk. This is record-level caching: one hot record costs its key size plus its value size plus 8 bytes of metadata, instead of the 4KB that a buffer pool would use. With a fixed memory budget, BF-Tree caches more key ranges.

**Growing for scan workloads.** When a key range receives many reads across different records, the mini-page grows to hold more cached records. At the maximum size (the full 4KB page), the mini-page is promoted to `PageLocation::Full`, becoming a complete in-memory mirror of the disk page. This gives range scans the same locality benefit as a traditional buffer pool, because all records in the key range are physically adjacent in memory.

This variable sizing is why BF-Tree uses less memory than a buffer pool for the same number of cached key ranges. A buffer pool allocates 4KB per cached page regardless of how many records are hot. BF-Tree allocates 128 bytes for a cold key range, 512 bytes for a warm one, and 2048 bytes for a hot one. The same 32MB of buffer memory covers far more key ranges.

<script type="text/typogram">
                        +------------------+
                        |   B-Tree Index   |
                        |   (inner nodes)  |
                        +--------+---------+
                                 |
                    routes to leaf page IDs
                                 |
                                 v
    +-----------------------------------------------------+
    |         Page Table: PageID -> PageLocation          |
    |  +----------+ +----------+ +----------+             |
    |  |Mini(*ptr)| |Full(*ptr)| |Base(off) |  ...        |
    |  +-----+----+ +-----+----+ +-----+----+             |
    +--------+------------+------------+------------------+
             |            |            |
             v            v            |
    +-----------------------------------------------------+
    |              Circular Buffer (Ring)                 |
    |  +------+--------+--------+--------+------+-----+   |
    |  |128B  | 256B   | 128B   | 4096B  |960B  |free |   |
    |  |mini  | mini   | mini   | full   |mini  |     |   |
    |  +------+--------+--------+--------+------+-----+   |
    |  ^ head                              ^ tail         |
    |  (evict oldest)                      (allocate new) |
    +-----------------------------------------------------+
                          | flush dirty records on eviction
                          v
    +-----------------------------------------------------+
    |                  Disk (base pages)                  |
    |  +----------+----------+----------+-------------+   |
    |  | 4KB page | 4KB page | 4KB page |    ...      |   |
    |  +----------+----------+----------+-------------+   |
    +-----------------------------------------------------+
</script>

The mini-page starts small (128 bytes for a key range that has seen one write) and grows only as needed (up to 4096 bytes for a hot range accumulating many updates). Cold key ranges cost almost no memory. Hot key ranges absorb writes in-memory without touching disk. The same 32MB of buffer memory covers far more key ranges than a buffer pool, because each mini-page holds only the records worth caching, not the 4KB page surrounding them.

<script type="text/typogram">
Read "bob" (hot, in mini-page):           Read "charlie" (cold, not in mini-page):
  tree.read("bob")                           tree.read("charlie")
    -> traverse to leaf                        -> traverse to leaf
    -> PageLocation::Mini(ptr)                 -> PageLocation::Mini(ptr)
    -> search mini-page: FOUND                 -> search mini-page: NOT FOUND
    -> return (no disk I/O)                    -> follow next_level -> disk offset
                                               -> load 4KB base page from disk
                                               -> search base page: FOUND
                                               -> return (1 disk I/O)
</script>

This design gives BF-Tree the read performance of a B-tree (single traversal path, no multi-level merging) with write amplification closer to an LSM-tree (writes are batched in memory). In the paper's benchmarks, BF-Tree is 2x faster than both B-trees and LSM-trees for point lookups, 6x faster than B-trees for writes, and 2.5x faster than RocksDB for scans.

The key insight: decouple in-memory representation from on-disk format. The on-disk pages stay at 4KB (no change to the storage layer). The in-memory buffer uses variable-size mini-pages that track only what matters. This separation lets BF-Tree optimize for the actual working set rather than the physical page layout.

But the interesting engineering challenge is not the design idea. It is the implementation. A mini-page is born from a custom ring buffer allocator, used concurrently by readers and writers, grown through size classes when it fills up, rescued from eviction when it is still hot, and eventually evicted back to disk and recycled. Multiple threads compete over each stage of this lifecycle. The Rust implementation manages it with a 6-state machine, two RAII guard types, custom locks, and atomic compare-and-swap operations, all built on top of a circular buffer that acts as both allocator and memory manager.

We will follow a mini-page through its entire life: birth (allocation), growth (buffering writes), concurrent access (readers and writers sharing it safely), testing (systematically finding concurrency bugs), and eviction (flushing to disk and recycling memory). Each Rust concept is introduced at the point the lifecycle needs it.

---

## Section 1: Why Build Your Own Allocator?

Most Rust programmers never think about memory allocation. You write `Box::new(x)`, and a value appears on the heap. You push into a `Vec`, and it grows automatically. You clone an `Arc`, and reference counting handles the lifetime. The allocator is invisible.

### What `Box::new(x)` actually does

When you write `let b = Box::new(42u64)`, Rust calls `alloc::alloc(Layout::new::<u64>())` under the hood. This invokes the global allocator (usually the system allocator, or jemalloc if you have configured one). The global allocator finds a free region in the process's heap, returns a `*mut u8` pointer to 8 bytes with proper alignment, and Rust wraps that pointer in a `Box<u64>`. When `b` goes out of scope, `Box`'s `Drop` implementation calls `alloc::dealloc()`, returning the memory to the global allocator. The programmer never sees the raw pointer.

`Vec` and `Arc` work the same way underneath. `Vec` starts with a heap allocation, and when you `push` past its capacity, it calls `alloc::realloc()` to double the buffer size, copying existing elements to the new location. `Arc` wraps a heap-allocated control block containing the data and two reference counts (strong and weak). `Arc::clone()` increments the strong count atomically; when the last `Arc` is dropped, the strong count reaches zero and `dealloc()` frees the memory. All of these types delegate to the same global allocator.

The point is that everyday Rust code never touches raw memory. `Box`, `Vec`, and `Arc` handle allocation, deallocation, and lifetime management automatically. The global allocator is a black box. You ask for memory, you get memory. You drop a value, the memory is returned. This works well for most applications.

### What an allocator is

An allocator is a piece of code that manages a region of memory, handing out chunks on request and taking them back. Three core operations:

- **Allocate**: "Give me N bytes with alignment A." Returns a pointer.
- **Deallocate**: "I am done with this pointer." The allocator reclaims the memory.
- **Reallocate**: "I need more (or less) space at this pointer." The allocator grows or shrinks the allocation, possibly moving it.

<script type="text/typogram">
A memory region managed by an allocator:

+--------+----------+--------+------------------+--------+------+
| used   |   free   | used   |      free        | used   | free |
| 64B    |   128B   | 256B   |      512B        | 128B   |      |
+--------+----------+--------+------------------+--------+------+
</script>

The challenge is efficiency. A naive allocator that searches the entire free space on every allocation is too slow. Real allocators use sophisticated data structures (free lists, size classes, thread-local caches) to make allocation fast.

### The `GlobalAlloc` trait

Rust's standard allocator interface is the `GlobalAlloc` trait:

```rust
// From std::alloc
unsafe trait GlobalAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8;
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);
}
```

You can swap the global allocator with `#[global_allocator]`, but every allocation still goes through this interface. It is general-purpose: any size, any alignment, any lifetime.

### Why the global allocator does not work for BF-Tree

BF-Tree needs four guarantees that the global allocator cannot provide:

**Contiguous memory.** BF-Tree needs all mini-pages in a single contiguous byte array so it can use logical addressing (a pointer is just an offset from the start of the buffer). The global allocator scatters allocations across the heap. You cannot predict where two consecutive allocations will land, and you cannot compute the distance between them with simple arithmetic.

**Size-class-aware reuse.** When a 256-byte mini-page dies, BF-Tree wants to reuse that exact slot for the next 256-byte allocation without any searching. The global allocator has its own internal free lists optimized for its own heuristics. It might split a freed 256-byte region into smaller chunks, coalesce it with an adjacent free region to form a larger one, or return it to the OS entirely. None of these behaviors match what BF-Tree needs. BF-Tree's size classes (`[128, 192, 256, 512, 960, 1856, 2048, 4096]`) are specific to its workload characteristics. A 128-byte slot should be reused for the next 128-byte mini-page, not split or merged.

**Per-block metadata.** Every block in BF-Tree's buffer needs an 8-byte header tracking its lifecycle state (NotReady, Ready, BeginTombstone, Tombstone, FreeListed, Evicted) and its size. This header must be at a fixed, predictable offset relative to the block's data pointer so that any thread can find it in O(1) by pointer arithmetic. The global allocator does store metadata per allocation (it needs to know the size for `dealloc`), but that metadata is internal to the allocator and not accessible to application code. You cannot ask the global allocator "what is the lifecycle state of the block at this pointer?" You would need a separate side table mapping pointers to states, adding indirection and cache misses on every state check.

**Eviction ordering.** BF-Tree evicts the oldest blocks first (FIFO within the ring buffer). This requires knowing which block was allocated earliest, which the ring buffer provides naturally: blocks closer to `head_addr` are older, blocks closer to `tail_addr` are newer. The global allocator has no concept of allocation age or eviction order. You cannot ask it "which allocation is oldest?" or "give me the next block to evict." You would need to maintain a separate ordering structure (like a linked list or a priority queue), which adds complexity and synchronization overhead.

### The design choice

Instead of fighting the global allocator, BF-Tree allocates one large contiguous buffer at startup via `std::alloc::alloc` with 4096-byte alignment and a power-of-two size (default 32MB). From that point on, BF-Tree is its own allocator: it tracks free space, hands out variable-size blocks, reclaims dead blocks, and enforces lifecycle states. The global allocator is never called again for mini-page memory.

<script type="text/typogram">
Global allocator (scattered across heap):
  +---+    +----+         +--+   +-----+        +---+
  |256|    |128 |         |64|   | 512 |        |128|
  +---+    +----+         +--+   +-----+        +---+
  No ordering. No adjacency. No metadata.

Circular buffer (contiguous ring, ordered by age):
  +------------------------------------------------------------+
  |[hdr|256B][hdr|128B][hdr|512B][hdr|128B][hdr|960B]  free    |
  |^ head (oldest)                              ^ tail (newest)|
  +------------------------------------------------------------+
  FIFO order. Adjacent. 8-byte header per block.
</script>

Database buffer pools, GPU memory managers, and network packet pools all use this same strategy. What is interesting about BF-Tree is that it does so in Rust, where the ownership system and borrow checker impose constraints that C/C++ implementations do not face. The rest of this blog shows how the implementation works with Rust's type system rather than fighting it: RAII guards enforce state transitions, `PhantomData` ties pointer lifetimes to the buffer, and `mem::forget` suppresses destructors on the success path.

With this context, let us see how BF-Tree's circular buffer allocates a mini-page.

---

## Section 2: Birth, Allocating a Mini-Page

When a write arrives for a disk page with no mini-page, BF-Tree needs to create one. This section traces the allocation from trigger to published block, introducing the circular buffer's internals, the RAII guard that tracks ownership, and the first two states of the lifecycle machine.

### The trigger

The write path starts in `insert()` (`tree.rs:814`), a retry loop that calls `write_inner()`. Inside `write_inner()`, the logic dispatches on `PageLocation`:

```rust
// storage.rs
enum PageLocation {
    Mini(*mut LeafNode),  // mini-page in the circular buffer, base page on disk
    Full(*mut LeafNode),  // full 4KB page in the circular buffer
    Base(usize),          // disk offset only, no in-memory buffer
    Null,                 // no page allocated (cache-only mode)
}
```

When the page table says `PageLocation::Base(offset)` or `PageLocation::Null`, no mini-page exists yet. The tree allocates one from the circular buffer.

### Size classes

Mini-pages do not use arbitrary sizes. They jump through a precomputed set of size classes. The formula from `create_mem_page_size_classes` (`tree.rs:132`) is:

```
size_class[i] = ceil_to_cache_line(2^i * c + sizeof(LeafNode))
```

where `c = min_record_size + sizeof(LeafKVMeta)` (the minimum bytes per record including its 8-byte metadata), and `ceil_to_cache_line` rounds up to the next multiple of 64 bytes.

With the default configuration (`min_record_size=48`, `max_record_size=1952`, `leaf_page_size=4096`), this produces:

```
[128, 192, 256, 512, 960, 1856, 2048, 4096]
```

The first seven values (128 through 2048) are mini-page sizes. The last value (4096) is the full page size. Exponential growth means a cold key range with one record costs 128 bytes, while a hot range accumulating hundreds of updates can grow to 4096 bytes. The same buffer memory covers far more key ranges than a fixed-4KB buffer pool.

### The circular buffer

The circular buffer is a single contiguous block of memory, allocated at startup via `std::alloc::alloc` with 4096-byte alignment. The size must be a power of two (the default is 32MB). Three pointers divide the buffer into regions:

```rust
// circular_buffer/mod.rs
struct States {
    head_addr: AtomicUsize,  // oldest live data
    evicting_addr: usize,    // eviction reservation boundary
    tail_addr: usize,        // next allocation point
}
```

<script type="text/typogram">
+--------------------------------------------------------------+
|  evicted  |    live data (mini-pages)      |  free space     |
|           |                                |                 |
+--------------------------------------------------------------+
^ head      ^ evicting                       ^ tail
</script>

The invariant is `head_addr <= evicting_addr <= tail_addr`. New allocations bump `tail_addr` forward. Eviction reclaims from `head_addr`. The region between head and tail is live data. The `evicting_addr` sits between head and tail: it marks how far eviction has been claimed. Blocks between `head_addr` and `evicting_addr` are being evicted (their data may be flushed to disk right now). Blocks between `evicting_addr` and `tail_addr` are live.

But what are these addresses, exactly? They are not byte offsets into the array. They are logical addresses that increase monotonically and never wrap, even when the physical buffer wraps around.

### Logical vs physical addressing

In a concurrent ring buffer, multiple threads atomically update `head_addr` and `tail_addr`. If these were physical offsets that wrap around (0 -> capacity -> 0), you would face the ABA problem: Thread A reads `tail = 100`, Thread B allocates enough to wrap `tail` back to `100`, Thread A's compare-and-swap succeeds when it should fail because the buffer state has completely changed. Monotonically increasing logical addresses eliminate this bug class. Two addresses can only be equal if they refer to the same allocation, not to different allocations that happen to land at the same physical offset after wraparound.

The conversion from logical to physical is a single bitwise AND:

```rust
// circular_buffer/mod.rs
fn logical_to_physical(&self, addr: usize) -> *mut u8 {
    let offset = addr & (self.capacity - 1);
    unsafe { self.data_ptr.add(offset) }
}
```

Because `capacity` is a power of two, `capacity - 1` is a bitmask. A 32MB buffer (capacity = 0x2000000) has mask 0x1FFFFFF. Logical address 0x3000100 maps to physical offset 0x1000100. The unsafe pointer arithmetic is a one-liner, easy to audit.

With logical addressing, all the common questions have simple answers:
- Free space: `capacity - (tail_addr - head_addr)`. No wraparound check.
- Is block A older than block B? `logical_addr_A < logical_addr_B`. Always works.
- Is this block near eviction? `(tail_addr - block_addr) > threshold`. Simple subtraction.

One complication remains: physical memory is finite. When `tail_addr` reaches the end of the physical array, the next allocation must start at physical offset 0. But a variable-size block must be contiguous in memory (you cannot split a 256-byte `LeafNode` with half at the end and half at the beginning). If the remaining physical space is smaller than the requested allocation, `alloc()` writes a tombstone header to mark the gap as dead space and retries from physical offset 0:

```rust
// Inside alloc(): handling the physical boundary
let physical_remaining = self.capacity - (states.tail_addr & (self.capacity - 1));
if physical_remaining < required {
    // Write a tombstone header so the eviction scanner can skip this gap.
    // Advance tail_addr past the gap, then retry.
    return self.alloc(size);
}
```

The tombstone wastes at most 4096 bytes (the maximum allocation size) at each physical wraparound. The eviction scanner skips it by reading the header's size field and advancing past it.

### The 8-byte allocation header

Every allocation in the circular buffer is preceded by an 8-byte `AllocMeta` header:

```rust
// circular_buffer/mod.rs
#[repr(C, align(8))]
struct AllocMeta {
    size: u32,              // allocation size in bytes
    states: MetaRawState,   // lifecycle state (contains AtomicU8)
}
```

When the eviction scanner walks the buffer from head to tail, it reads these headers sequentially to find out how big each block is and what state it is in. The `states` field is where the 6-state lifecycle machine lives.

### How `alloc()` works

The simplest possible allocator is a bump allocator: keep a pointer to the next free byte, advance it by N on each request. The ring buffer extends this by reclaiming from the other end (the head), giving bump-allocation speed with FIFO reclamation. But mini-pages do not die in FIFO order. When a 128-byte mini-page grows into a 256-byte one, the old block is dead but sits between live blocks. The ring buffer cannot reclaim it by advancing the head. A free list (covered in Section 3) solves this by recycling dead blocks in the middle of the buffer.

The allocator has two strategies. It tries the free list first (reuse a previously-deallocated block of the right size class), then falls back to bumping the tail:

```rust
// circular_buffer/mod.rs, alloc() simplified
pub fn alloc(&self, size: usize) -> Result<CircularBufferPtr<'_>, CircularBufferError> {
    let (_lock, states) = self.lock_states();

    // Strategy 1: try reusing a freed block from the per-size-class free list.
    while let Some(ptr) = self.free_list.remove(size) {
        let meta = CircularBuffer::get_meta_from_data_ptr(ptr.as_ptr());

        // Skip blocks near the head - they are about to be evicted anyway.
        if self.ptr_is_copy_on_access(ptr.as_ptr()) {
            meta.states.free_list_to_tombstone();
            continue;
        }

        // CAS: FreeListed -> NotReady. Exactly one thread wins.
        match meta.states.state.compare_exchange_weak(
            MetaState::FreeListed.into(),
            MetaState::NotReady.into(),
            Ordering::AcqRel, Ordering::Relaxed,
        ) {
            Ok(_) => return Ok(CircularBufferPtr::new(ptr.as_ptr())),
            Err(_) => continue,  // another thread got it first
        }
    }

    // Strategy 2: bump-allocate from the tail.
    let aligned_size = align_up(size, CB_ALLOC_META_SIZE);
    let required = aligned_size + std::mem::size_of::<AllocMeta>();

    let logical_free = self.capacity - (states.tail_addr - states.head_addr());
    if logical_free < required {
        return Err(CircularBufferError::Full);  // caller must evict
    }

    // Write the 8-byte header, bump the tail, return a guard.
    unsafe {
        let physical_addr = self.logical_to_physical(states.tail_addr);
        physical_addr.cast::<AllocMeta>().write(
            AllocMeta::new(aligned_size as u32, false)
        );
    }
    states.tail_addr += required;

    Ok(CircularBufferPtr::new(
        self.logical_to_physical(states.tail_addr - aligned_size)
    ))
}
```

Notice the `Ordering` parameters on the CAS: `AcqRel` on success, `Relaxed` on failure. Modern CPUs reorder memory operations for performance. Without ordering constraints, Thread A might write data to a block and set its state to `Ready`, but Thread B might see `Ready` before it sees the data. Rust's `Ordering` enum controls this:

- **`Relaxed`**: The operation is atomic, but surrounding reads and writes can be reordered freely. Used on CAS failure paths where the thread will just retry.
- **`Acquire`**: All reads after this operation see data written before a corresponding `Release`. Used when reading a state that protects data.
- **`Release`**: All writes before this operation are visible to threads that do a corresponding `Acquire`. Used when publishing data.
- **`AcqRel`**: Both `Acquire` and `Release`. Used for read-modify-write operations that both consume and publish data. This is the most common ordering for state transitions in BF-Tree.

In the `alloc()` code: on a successful CAS (`FreeListed -> NotReady`), `AcqRel` ensures the allocating thread sees all data the previous owner wrote (`Acquire`), and any thread that later reads `NotReady` will see the allocating thread's writes (`Release`). On failure, `Relaxed` is fine because the thread just retries with the next free-list entry.

### The allocation guard: `CircularBufferPtr`

`alloc()` does not return a raw pointer. It returns a `CircularBufferPtr`, an RAII guard that tracks the block's lifecycle:

```rust
// circular_buffer/mod.rs
pub struct CircularBufferPtr<'a> {
    ptr: *mut u8,
    _pt: PhantomData<&'a ()>,
}

impl Drop for CircularBufferPtr<'_> {
    fn drop(&mut self) {
        let meta = CircularBuffer::get_meta_from_data_ptr(self.ptr);
        meta.states.to_ready();  // CAS: NotReady -> Ready
    }
}
```

The pointer inside the guard is a raw `*mut u8` with no lifetime. Raw pointers in Rust are untracked by the borrow checker: they can be copied, stored, and dereferenced (in unsafe code) without any compile-time ownership checks. This is necessary because the pointer points into the circular buffer's contiguous allocation, not to a separate heap object that Rust can track.

But raw pointers are dangerous precisely because Rust cannot track them. The `PhantomData<&'a ()>` field compensates: it tells the compiler "treat this guard as if it borrows from the `CircularBuffer` for lifetime `'a`." Without it, nothing prevents a caller from dropping the buffer (which frees the underlying memory) while still holding a guard that points into that memory. The phantom lifetime catches use-after-free at compile time.

### The block lifecycle: `NotReady` and `Ready`

Every block in the circular buffer carries a single byte of state inside its `AllocMeta` header. The state controls who can do what with the block. For now, only two states matter:

- **`NotReady` (0)**: The block was just allocated. Only the allocating thread can touch it. The evictor will skip it.
- **`Ready` (1)**: The block is live. Any thread can read it. The evictor can claim it.

The `CircularBufferPtr` guard is what controls this transition. The caller receives the guard from `alloc()`, initializes the block (writes the `LeafNode` header, inserts the first record, updates the page table), and then either lets the guard go out of scope or calls `drop(mini_page_guard)` explicitly. Either way, the guard's `Drop` impl fires, performing a CAS from `NotReady` to `Ready`. This is the "publish" step: the block becomes visible to other threads and to the evictor. Until the guard is dropped, the block is invisible.

There are four more states (`BeginTombstone`, `Tombstone`, `FreeListed`, `Evicted`) that govern deallocation, recycling, and eviction. We will introduce each one at the point the lifecycle needs it: `BeginTombstone` and `FreeListed` in Section 3 (growth), `Tombstone` and `Evicted` in Section 4 (eviction).

### End-to-end example: allocating a 128-byte mini-page

To make this concrete, trace what happens physically when the tree inserts a record into a cold key range (one with `PageLocation::Base`, meaning no mini-page exists yet).

**Step 1: The tree calls `storage.alloc_mini_page(128)`.** This calls `circular_buffer.alloc(128)`.

**Step 2: `alloc()` checks the free list.** The free list has per-size-class buckets. For size 128, it checks the 128-byte bucket. If empty, it falls through to bump allocation.

**Step 3: Bump allocation.** Suppose `tail_addr` is currently at logical address 1000. The allocator writes an 8-byte `AllocMeta` header at the physical address corresponding to logical 1000, then advances `tail_addr` by 136 (8-byte header + 128-byte payload). The buffer now looks like:

<script type="text/typogram">
Physical memory at logical address 1000:
+---------------------+------------------------------------------+
|  AllocMeta (8 bytes) |  payload (128 bytes, uninitialized)     |
|  size: 128           |                                         |
|  state: NotReady (0) |  <- guard.as_ptr() points here          |
+---------------------+------------------------------------------+
                                                tail_addr = 1136
</script>

**Step 4: The tree initializes the payload as a `LeafNode`.** It casts the `*mut u8` to `*mut LeafNode`:

```rust
let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
```

This cast requires `LeafNode` to have a predictable memory layout. Rust normally reorders struct fields for optimization. The `#[repr(C)]` attribute overrides this: fields are laid out in declaration order with C-compatible padding rules.

```rust
// nodes/leaf_node.rs
#[repr(C)]
pub(crate) struct LeafNode {
    pub(crate) meta: NodeMeta,              // 6 bytes
    prefix_len: u16,                         // 2 bytes
    pub(crate) next_level: MiniPageNextLevel,// 8 bytes
    pub(crate) lsn: u64,                     // 8 bytes
    data: [u8; 0],                           // flexible array member
}
const _: () = assert!(std::mem::size_of::<LeafNode>() == 24);
```

The `data: [u8; 0]` field is the C flexible array member pattern. The compiler sees a 24-byte struct, but the allocator provided 128 to 4096 bytes. The remaining bytes after the header are accessed through methods that compute offsets from `meta.node_size`, which records the actual allocation size. The compile-time assertion catches accidental size changes at build time: if someone adds a field to `LeafNode`, the build breaks before any pointer arithmetic goes wrong.

After the cast, `LeafNode::initialize_mini_page()` writes the 24-byte header into the first 24 bytes, then inserts the record using the dual-growing layout (metadata forward, payloads backward). The block is still in `NotReady` state, so the evictor cannot touch it.

<script type="text/typogram">
Physical memory at logical address 1000:
+---------------------+------------------------------------------+
|  AllocMeta (8 bytes) |  LeafNode (128 bytes)                   |
|  size: 128           |  +----------+-------+------------------+|
|  state: NotReady (0) |  |header 24B|KVMeta |  ...  |val|key   ||
|                      |  +----------+-------+------------------+|
+---------------------+------------------------------------------+
</script>

**Step 5: The tree updates the page table.** It writes `PageLocation::Mini(ptr)` into the page table entry for this leaf page. Now any future read or write to this key range will find the mini-page.

**Step 6: The tree drops the `CircularBufferPtr` guard.** The guard's `Drop` implementation fires, performing `CAS(NotReady -> Ready)`. The block is now published.

<script type="text/typogram">
Physical memory (after guard drop):
+---------------------+------------------------------------------+
|  AllocMeta (8 bytes) |  LeafNode (128 bytes, initialized)      |
|  size: 128           |  ... contains one record ...            |
|  state: Ready (1)    |                                         |
+---------------------+------------------------------------------+
</script>

This six-step sequence (alloc, cast, initialize, update page table, drop guard, published) happens every time the tree creates a new mini-page. The allocator provides raw bytes; the tree interprets them as a `LeafNode`; the guard ensures the block is not evicted before initialization completes.

What happens if the thread panics during step 4? Rust's unwinding calls `Drop` on the `CircularBufferPtr` guard. The block transitions to `Ready` even though initialization is incomplete. This is by design: the circular buffer does not know what "correct initialization" means. The tree-level code ensures a partially-initialized mini-page is either recoverable or will be evicted without harm.

The mini-page is born and `Ready`. What happens when the next write arrives?

---

## Section 3: Growth: Buffering Writes and Recycling Memory

A mini-page in `Ready` state can follow three paths:

1. **Absorb writes.** If the record fits, insert it directly. The mini-page stays in place.
2. **Grow.** If the record does not fit, allocate a larger mini-page, copy all records to it, and deallocate the old one.
3. **Die.** When the mini-page is evicted or replaced, its memory must be recycled for future allocations.

Paths 2 and 3 both require *deallocation*: reclaiming a block that was previously `Ready`. But deallocation in a concurrent system is tricky. Multiple threads might try to claim the same block (the growth path racing against the evictor). The solution is a second RAII guard, `TombstoneHandle`, that uses the same pattern as `CircularBufferPtr`: acquire exclusive rights via CAS, do the work, and let `Drop` handle cleanup if anything goes wrong.

This section traces the growth path, introduces `TombstoneHandle`, and shows how dead blocks are recycled through a free list.

### The write path

To understand growth, we first need to see where it happens. The `insert()` method is a retry loop that calls `write_inner()`:

```rust
// tree.rs, insert() - simplified
pub fn insert(&self, key: &[u8], value: &[u8]) -> LeafInsertResult {
    let backoff = Backoff::new();
    loop {
        match self.write_inner(WriteOp::make_insert(key, value), aggressive_split) {
            Ok(_) => return LeafInsertResult::Success,
            Err(TreeError::NeedRestart) => { /* version changed, retry */ }
            Err(TreeError::CircularBufferFull) => {
                self.evict_from_circular_buffer();  // free some memory
            }
            Err(TreeError::Locked) => {
                backoff.snooze();  // another thread holds the lock
            }
        }
    }
}
```

Inside `write_inner()`, the logic dispatches on `PageLocation`:

```rust
// tree.rs, write_inner() - simplified
fn write_inner(&self, write_op: WriteOp, aggressive_split: bool)
    -> Result<(), TreeError>
{
    let (pid, parent) = self.traverse_to_leaf(write_op.key, aggressive_split)?;
    let mut leaf_entry = self.mapping_table().get_mut(&pid);  // acquires write lock

    match leaf_entry.get_page_location() {
        PageLocation::Null => {
            // Cache-only mode: no disk backing. Allocate a fresh mini-page.
            let mini_page_guard = self.storage.alloc_mini_page(size)?;
            LeafNode::initialize_mini_page(&mini_page_guard, ...);
            leaf_entry.create_cache_page_loc(PageLocation::Mini(new_mini_ptr));
        }
        _ => {
            // Mini, Full, or Base: delegate to leaf_entry.insert()
            leaf_entry.insert(key, value, parent, op_type, &self.storage)?;

            // Check if this mini-page is in the copy-on-access zone.
            if leaf_entry.cache_page_about_to_evict(&self.storage) {
                leaf_entry.move_cache_page_to_tail(&self.storage);
            }
        }
    }
    Ok(())
}
```

The dispatch logic:
- `PageLocation::Base` (disk only): create a mini-page (Section 2), then insert.
- `PageLocation::Mini`: try to insert into the existing mini-page. If the record fits, done. If not, grow.
- `PageLocation::Full`: the mini-page has already been promoted to a full 4KB in-memory page. Insert directly.

The interesting case is when the record does not fit. What happens next?

### The growth decision

A 128-byte mini-page holds roughly 1-2 records (24-byte header + 8-byte metadata per record + key-value payload). When a third record arrives and doesn't fit, the mini-page must grow. But grow to what size?

BF-Tree uses a precomputed array of size classes: `[128, 192, 256, 512, 960, 1856, 2048, 4096]`. The insert code checks if the current mini-page has room. If not, it picks the next size class. A 128-byte mini-page grows to 192 bytes. A 256-byte mini-page grows to 512 bytes. The exponential progression means hot key ranges quickly get enough space, while cold key ranges stay small.

What if the mini-page is already at 2048 bytes? That is the maximum *mini*-page size. At that point, BF-Tree has two options: either grow to 4096 bytes and become a *full* page (`PageLocation::Full`, a complete in-memory mirror of the disk page), or merge the mini-page's dirty records into the 4KB base page on disk and change the page table entry to `Base(offset)`. Which path is chosen depends on access patterns. If the key range stays hot, the mini-page grows to full size for efficient range scans. If it cools down, it is evicted to disk. Either way, the mini-page lifecycle ends and a new one may begin.

### The growth operation

Growing a mini-page is not as simple as reallocating. The old block is in the circular buffer. Other threads might be reading from it. The evictor might be trying to claim it. We need to:
1. Prevent anyone else from deallocating the old block while we copy from it.
2. Allocate the new block.
3. Copy the records.
4. Atomically switch the page table pointer from old to new.
5. Deallocate the old block so its memory can be reused.

Here is the four-step sequence in code:

```rust
// mini_page_op.rs - simplified
// Step 1: Claim exclusive rights on the old mini-page.
let h = storage.begin_dealloc_mini_page(mini_page)?;

// Step 2: Allocate a bigger block from the circular buffer.
let mini_page_guard = storage.alloc_mini_page(new_size)?;

// Step 3: Copy records from old to new.
mini_page.copy_initialize_to(new_ptr, new_size, true);

// Step 4: Update page table, insert new record, publish, deallocate old.
self.create_cache_page_loc(PageLocation::Mini(new_mini_ptr));
self.load_cache_page_mut(new_mini_ptr).insert(key, value, op_type, 0);
drop(mini_page_guard);           // publishes new block (NotReady -> Ready)
storage.finish_dealloc_mini_page(h);  // reclaims old block
```

Step 1 is where things get interesting. We need to claim exclusive rights to the old block while copying from it. But what if the evictor is also trying to claim this block at the same moment? Two threads cannot both deallocate the same block. We need a way to ensure exactly one wins.

### `compare_exchange` (CAS) and the deallocation guard

The solution is atomic compare-and-swap, a single CPU instruction (`CMPXCHG` on x86): "If the current value is X, set it to Y; otherwise tell me what it actually is." The entire read-check-write is atomic. If two threads race to claim a block, exactly one CAS succeeds and the other fails. Here is how BF-Tree uses it:

```rust
// circular_buffer/mod.rs
fn try_begin_tombstone(&self) -> bool {
    self.state.compare_exchange(
        MetaState::Ready.into(),          // expected: 1
        MetaState::BeginTombStone.into(), // desired: 3
        Ordering::AcqRel,                 // memory ordering on success
        Ordering::Relaxed,                // memory ordering on failure
    ).is_ok()
}
```

The thread that wins the CAS owns the block exclusively until it finishes deallocation. But what if something goes wrong? The thread might panic, or the eviction callback might fail. The block would be stuck in `BeginTombstone` forever, neither usable nor reclaimable.

This is the same problem `CircularBufferPtr` solved for allocation: we need an RAII guard whose `Drop` handles the abort path. `TombstoneHandle` is that guard for deallocation:

```rust
// circular_buffer/mod.rs
pub struct TombstoneHandle {
    pub(crate) ptr: *mut u8,
}

impl Drop for TombstoneHandle {
    fn drop(&mut self) {
        let meta = CircularBuffer::get_meta_from_data_ptr(self.ptr);
        meta.states.revert_to_ready();  // CAS: BeginTombstone -> Ready
    }
}
```

The `Drop` impl reverts the block to `Ready`. This is a safety net: if you panic or bail out before finishing deallocation, the block stays usable rather than leaking. The `TombstoneHandle`'s `Drop` always does the conservative thing (revert to Ready), just as `CircularBufferPtr`'s `Drop` always does the conservative thing (publish to Ready).

Acquiring a `TombstoneHandle` requires winning the CAS race:

```rust
fn acquire_exclusive_dealloc_handle(ptr: *mut u8) -> Result<TombstoneHandle, ...> {
    let meta = CircularBuffer::get_meta_from_data_ptr(ptr);
    if meta.states.try_begin_tombstone() {  // CAS: Ready -> BeginTombstone
        Ok(TombstoneHandle { ptr })
    } else {
        Err(...)  // another thread got it first
    }
}
```

Not all CAS operations in the codebase use the same error handling:

- **`try_begin_tombstone()` returns `bool`**: Multiple threads can race to claim a block (growth path vs evictor). The loser gets `false` and retries. This is expected contention.
- **`to_ready()` panics on failure**: Only the thread holding the `CircularBufferPtr` guard should call this. Failure means a logic error, not contention.
- **`to_tombstone()` and `to_freelist()` panic on failure**: Only the thread holding the `TombstoneHandle` should call these. The handle guarantees exclusive access.

The rule: transitions where multiple threads compete return a fallible result. Transitions where a single thread has exclusive rights (guaranteed by a guard) panic on failure because failure means a bug.

### `mem::forget`: suppressing incorrect cleanup

Back to the growth operation. We have a `TombstoneHandle` on the old block. We have allocated a new larger block, copied the records, and updated the page table. Now we need to finish deallocating the old block.

The state machine has three possible next states from `BeginTombstone`:
- **`Ready`**: Abort. Something went wrong. Revert so the block can be used again.
- **`FreeListed`**: Success. Add the block to the free list for immediate reuse.
- **`Tombstone`**: Success, but skip the free list. The block is dead but not yet reclaimable by `head_addr`.

Why does `Tombstone` exist? Because `head_addr` advances in order, and only the evictor can mark blocks `Evicted`. A block deallocated by the growth path might sit in the middle of the buffer, far from `head_addr`. It cannot be marked `Evicted` immediately. Instead, it is marked `Tombstone`: "I'm dead, evictor please clean me up when you get here." When the evictor eventually reaches a `Tombstone` block, it simply flips it to `Evicted` with no further work.

The `TombstoneHandle`'s `Drop` does the first one (revert to `Ready`). But on the success path, we want `FreeListed` or `Tombstone`. Here is the problem:

```rust
// WRONG: what happens if we just use the handle normally?
fn dealloc_wrong(handle: TombstoneHandle) {
    let ptr = handle.ptr;           // copy the raw pointer out
    let meta = get_meta(ptr);
    meta.states.to_freelist();      // state: BeginTombstone -> FreeListed

    // But now handle goes out of scope...
    // Drop fires automatically!
    // Drop tries: BeginTombstone -> Ready
    // But state is already FreeListed, not BeginTombstone!
    // The CAS fails or corrupts the state machine.
}
```

The handle's `Drop` will always run when the handle goes out of scope. We cannot stop Rust from calling `Drop`. But we can prevent the handle from going out of scope in the normal way: consume it with `std::mem::forget`, which takes ownership and never runs the destructor.

```rust
impl TombstoneHandle {
    fn into_ptr(self) -> *mut u8 {
        let ptr = self.ptr;
        std::mem::forget(self);  // suppress Drop
        ptr
    }
}
```

`std::mem::forget` takes ownership of a value and prevents its destructor from running. It is a safe function in Rust (not `unsafe`) because Rust's safety model never depends on destructors running. Leaking memory is not undefined behavior.

To see why this matters, trace both paths:

<script type="text/typogram">
WITHOUT mem::forget (BROKEN):
  1. acquire_exclusive_dealloc_handle(ptr)  -> state: 1 -> 3
  2. let raw_ptr = handle.ptr;              -> copy the pointer
  3. handle goes out of scope               -> Drop fires!
     +-- revert_to_ready()                  -> state: 3 -> 1  <- WRONG
  4. meta.states.to_freelist()              -> expects state 3, finds 1 -> PANIC

WITH mem::forget (CORRECT):
  1. acquire_exclusive_dealloc_handle(ptr)  -> state: 1 -> 3
  2. let raw_ptr = handle.into_ptr();       -> copy pointer, then forget(self)
     +-- Drop is suppressed                 -> state stays at 3
  3. meta.states.to_freelist()              -> state: 3 -> 4  <- CORRECT
</script>

The `dealloc_inner()` function uses this pattern:

```rust
// circular_buffer/mod.rs, dealloc_inner() - simplified
fn dealloc_inner(&self, handle: TombstoneHandle, add_to_freelist: bool) {
    let ptr = handle.into_ptr();  // mem::forget suppresses Drop
    let meta = CircularBuffer::get_meta_from_data_ptr(ptr);

    if !add_to_freelist || self.ptr_is_copy_on_access(ptr) {
        meta.states.to_tombstone();  // CAS: BeginTombstone -> Tombstone
        return;
    }

    match self.free_list.try_add(ptr, meta.size as usize) {
        Ok(_) => meta.states.to_freelist(),    // CAS: BeginTombstone -> FreeListed
        Err(_) => meta.states.to_tombstone(),  // contention, skip free list
    }
}
```

The design principle: `Drop` handles the abort path (revert to safe state). The success path calls `mem::forget` to suppress `Drop` and performs the forward transition explicitly. If the code panics between `acquire_exclusive_dealloc_handle` and `into_ptr`, the `Drop` fires and reverts to `Ready`. If the code completes normally, `mem::forget` suppresses `Drop` and the explicit CAS moves forward. Every code path leaves the state machine in a valid state.

### The free list

Recall from Section 2: mini-pages do not die in FIFO order. When a 128-byte mini-page grows to 256 bytes, the old 128-byte block is dead but sits between live blocks. The ring buffer cannot reclaim it by advancing the head. Without a free list, that 128-byte hole would be wasted until the head eventually catches up.

The free list solves this by recycling dead blocks immediately. When a block is deallocated (not near the eviction head, and the free list lock is available), it is added to a per-size-class free list. The next `alloc(128)` call can reuse it instead of bumping the tail.

The free list is intrusive: instead of allocating separate node structs on the heap, it writes a linked list pointer directly into the dead block's bytes. The block is dead, so its payload bytes are available for reuse as a `ListNode`:

```rust
// circular_buffer/freelist.rs
struct ListNode {
    next: *mut ListNode,
}

pub(super) struct FreeList {
    size_classes: Vec<usize>,
    list_heads: Vec<Mutex<*mut ListNode>>,  // one linked list per size class
}
```

Here is what happens to the physical memory at each stage:

<script type="text/typogram">
STAGE 1: Block is live (state: Ready, holding a LeafNode with records)

  +----------------------+--------------------------------------+
  |  AllocMeta (8 bytes) |  payload: LeafNode (128 bytes)       |
  |  size: 128           |  [header][metadata][record data]     |
  |  state: Ready        |                                      |
  +----------------------+--------------------------------------+

STAGE 2: Block dies, added to 128-byte free list (state: FreeListed)

  +----------------------+--------------------------------------+
  |  AllocMeta (8 bytes) |  payload (128 bytes, reinterpreted)  |
  |  size: 128           |  +--------------+------------------+ |
  |  state: FreeListed   |  | next: *ptr   | ...stale bytes...| |
  +----------------------+--+--------------+------------------+-+
                                   |
                    points to next dead block in the 128-byte list

STAGE 3: alloc(128) reuses this block (state: NotReady -> Ready)

  +----------------------+--------------------------------------+
  |  AllocMeta (8 bytes) |  payload: NEW LeafNode (128 bytes)   |
  |  size: 128           |  [header][metadata][new records]     |
  |  state: Ready        |                                      |
  +----------------------+--------------------------------------+
</script>

Adding a dead block casts the payload pointer to `*mut ListNode` and links it into the per-size-class list. Retrieval is the reverse: pop from the list head and return the pointer. One subtlety: `try_add` uses `try_lock` (non-blocking), while `remove` uses `lock` (blocking). If adding to the free list would block, the code skips it and marks the block `Tombstone` instead. The block will still be reclaimed when the evictor passes over it. On the allocation side, waiting briefly for the free list lock is acceptable since allocation can already block on eviction.

After `remove()` returns a pointer, `alloc()` still needs to claim the block with CAS (`FreeListed -> NotReady`) because the evictor might be racing to reclaim the same block.

### Scenario: the growth path vs. the evictor

A mini-page is full. Thread A (writer) wants to grow it to a larger size class. Thread B (evictor) wants to evict it because the buffer is full. Both need exclusive access to the block. Who wins?

<script type="text/typogram">
Time --------------------------------------------------------------->

Thread A (growth path):
  storage.begin_dealloc_mini_page(old_ptr)
  +-- acquire_exclusive_dealloc_handle(old_ptr)
      +-- CAS(Ready -> BeginTombstone)  <- WINS (thread A was first)
      +-- returns TombstoneHandle

Thread B (evictor):
  acquire_exclusive_dealloc_handle(old_ptr)
  +-- CAS(Ready -> BeginTombstone)  <- FAILS (state is already 3)
  +-- returns Err(WouldBlock)
</script>

Thread A wins. It now holds a `TombstoneHandle`, which gives exclusive deallocation rights. Thread A can safely:
1. Allocate a bigger block from the circular buffer.
2. Copy all records from old block to new block. (The old block's data is still intact; `BeginTombstone` prevents deallocation but does not zero the bytes.)
3. Update the page table pointer from old to new.
4. Deallocate the old block (transitions `BeginTombstone -> FreeListed`).

Thread B, the evictor, enters a retry loop. It checks the block's state. If Thread A finishes quickly and the block becomes `FreeListed`, the evictor removes it from the free list and marks it `Evicted`. If Thread A is slow, the evictor spins until the state changes. Either way, exactly one thread performs the deallocation, and the evictor ultimately marks the block `Evicted` so `head_addr` can advance.

### Scenario: eviction callback fails

The evictor claimed a block (state is `BeginTombstone`). It calls the tree's `eviction_callback()` to merge dirty records to disk. But the callback discovers that the page table entry has changed (another thread grew the mini-page). The callback returns an error.

<script type="text/typogram">
Time --------------------------------------------------------------->

Evictor:
  1. acquire_exclusive_dealloc_handle(ptr)
     +-- CAS(Ready -> BeginTombstone): SUCCESS
     +-- returns TombstoneHandle { ptr }

  2. callback(handle) returns Err(handle)
     +-- The eviction callback failed. The TombstoneHandle is returned.

  3. drop(handle)
     +-- TombstoneHandle::Drop fires
     +-- CAS(BeginTombstone -> Ready): state byte changes from 3 to 1
     Block is back to Ready. It can be used normally.

  4. backoff.spin(), then retry from step 1
</script>

The `TombstoneHandle`'s `Drop` is the safety net. If eviction fails for any reason (tree traversal error, version mismatch, disk I/O failure), the block reverts to `Ready` and nothing is lost. Without this safety net, a failed eviction would leave the block in `BeginTombstone` permanently. No thread could read it, evict it, or free it. The block would be stuck. The `Drop` implementation makes this impossible.

The mini-page can now grow and shrink. But what happens when memory fills up? The oldest blocks must be evicted to make room for new allocations.

---

## Section 4: Eviction, the Death of a Mini-Page

A mini-page dies when memory runs out. Eviction has three jobs: flush dirty records to disk, drop clean records, and reclaim the memory. This section follows a mini-page through its death.

### When eviction triggers

When `alloc()` finds no space, it returns `CircularBufferError::Full`. The calling thread catches this error in its retry loop and becomes an evictor:

```rust
// tree.rs - the retry loop
Err(TreeError::CircularBufferFull) => {
    self.evict_from_circular_buffer();  // I am the evictor now
    // then retry the insert...
}
```

There is no background evictor thread. The thread that needs space frees space. This simplifies coordination but means multiple threads may evict simultaneously under heavy load.

### What gets flushed, what gets dropped

Before evicting a mini-page, we need to know which records require disk I/O. Every record carries an `OpType`:

<script type="text/typogram">
               Present         Absent
  Dirty        Insert          Delete
  Clean        Cache           Phantom
</script>

**Dirty records** (`Insert`, `Delete`) represent changes not yet on disk. They must be merged into the 4KB base page before the mini-page can be freed.

**Clean records** (`Cache`, `Phantom`) are copies of data already on disk. They can be dropped immediately with no I/O.

The merge reads the base page from disk, applies all dirty inserts and deletes, writes it back. This is where write amplification is paid: one 4KB write for all the inserts that accumulated. If the mini-page buffered 20 writes, the effective write amplification is 4096/(20x100) = 2x, much better than the 40x without mini-pages.

### The two-phase eviction protocol

Eviction must not block allocations. Disk I/O takes milliseconds; holding a lock that long would serialize the entire system. The solution: a two-phase protocol with a third pointer.

<script type="text/typogram">
+--------------------------------------------------------------+
|  evicted  |  being evicted  |      live      |     free     |
+--------------------------------------------------------------+
^ head      ^ evicting         ^ tail
</script>

**Phase 1 (circular buffer lock held):** Bump `evicting_addr` forward by one block's size. Release the lock.

**Phase 2 (no circular buffer lock):** The slow part:
1. Acquire `TombstoneHandle` via CAS (Ready -> BeginTombstone)
2. Acquire the page table lock for this leaf
3. Read 4KB base page from disk, apply dirty records, write 4KB back
4. Update page table: `Mini(ptr)` -> `Base(offset)`
5. Release page table lock
6. Mark block as `Evicted`

**Phase 3 (circular buffer lock held):** Bump `head_addr` forward past all `Evicted` blocks.

The disk I/O in step 3 blocks the evicting thread for milliseconds. But the circular buffer lock was released in Phase 1, so other threads can still allocate. The page table lock (per-entry) is held during I/O, blocking only writers to that specific leaf page.

### Copy-on-access: rescuing hot blocks

The circular buffer evicts in FIFO order: oldest first. But FIFO is a poor policy when some old blocks are still hot. BF-Tree approximates LRU with copy-on-access.

The buffer is split into two zones:

<script type="text/typogram">
+----------------------------------------------------------------+
|  copy-on-access zone (10%)  |  in-place update zone (90%)      |
+----------------------------------------------------------------+
^ head                                                     tail ^
  (old, near eviction)                              (young, safe)
</script>

When any operation touches a block in the copy-on-access zone, it copies the block to the tail:

1. Claim the old block (CAS: Ready -> BeginTombstone)
2. Allocate a new block at the tail
3. Copy the records
4. Update the page table pointer
5. Deallocate the old block

The mini-page is "rescued." It now lives at the tail, far from eviction.

Why not a true LRU list? Updating a doubly-linked list on every access requires a lock (or CAS). In a concurrent system with millions of operations per second, that lock becomes a bottleneck. Copy-on-access pays only for blocks near the eviction boundary. Most accesses pay zero overhead.

### Full state machine

All six states have now appeared in context. Here is the complete diagram with annotations:

<script type="text/typogram">
                        +--------------+
               alloc()  |   NotReady   |  Section 2: allocated, being initialized
                        +------+-------+
                     drop guard| CAS(0->1)     Section 2: guard publishes
                        +------v-------+
                        |    Ready     |  live, usable
                        +------+-------+
                               |
      +------------------------+------------------------+
      | acquire_exclusive_dealloc_handle: CAS(1->3)     |
      |                                                 |
      v                                                 |
+-----------------+                                     |
| BeginTombstone  |  Section 3: exclusive dealloc claim |
+------+----------+                                     |
       |                                                |
       +---- drop TombstoneHandle --> revert to Ready --+
       |     (abort path)                 (safety net)
       |
       +---- try_add to free list succeeds --> +--------------+
       |                                       |  FreeListed  |  Section 3
       |                                       +------+-------+
       |                                              |
       |                                    (evictor removes from free list)
       |                                              |
       +---- try_add fails (lock contention) --> +----v---------+
                                                 |  Tombstone   |  Section 3: dead, not reusable
                                                 +------+-------+
                                                        |
                                                 (evictor walks in order)
                                                        |
                                                 +------v-------+
                                                 |   Evicted    |  Section 4: head can advance past
                                                 +--------------+
                                                        |
                                                        +--> back to NotReady (reused via alloc)
</script>

Every arrow is a `compare_exchange` on an `AtomicU8`. The state byte is the single source of truth for who owns the block.

The lifecycle is complete: birth (Section 2) -> growth and recycling (Section 3) -> death and eviction (Section 4). Now let's address the cross-cutting concern: how do multiple threads execute this lifecycle concurrently without corrupting the tree?

---

## Section 5: Concurrent Life: Locks, Versions, and Retry Loops

BF-Tree is concurrent. Multiple threads traverse inner nodes, read from mini-pages, insert records, and evict blocks simultaneously. The naive approach would wrap every node in a `std::sync::Mutex`. That serializes all operations: while one thread reads an inner node, every other thread waits. With millions of operations per second, the mutex becomes a bottleneck.

The insight: different tree levels have different access patterns.

- **Inner nodes** are traversed on every read and every write, but they only change during splits and merges. Reads vastly outnumber writes.
- **Page table entries** are where inserts land. A read may need to upgrade to a write (for read promotion). Writes are frequent.

Different access patterns call for different lock strategies.

### Inner nodes: optimistic versioned locks (seqlock)

Scenario: Thread A is reading an inner node to find which child pointer to follow. Thread B is splitting that node (a write). With a standard `RwLock`, Thread B would wait for Thread A to finish reading. With a seqlock, Thread B proceeds immediately. Thread A detects the concurrent modification and retries.

The mechanism: instead of acquiring a lock, a reader snapshots a version number, reads the data, then checks whether the version changed. If unchanged, no writer interfered. If changed, the reader discards its result and retries. Writers never wait for readers. Readers absorb the retry cost.

This tradeoff works when reads vastly outnumber writes. Inner nodes fit this pattern: every operation traverses them, but they change only during splits and merges.

The implementation uses a single `AtomicU16`. Bit 1 indicates "write-locked." Each lock acquisition adds 2 to the version (setting bit 1), and each release adds 2 again (clearing bit 1). So a complete write cycle advances the version by 4. A reader snapshots the version, does its work, then re-reads. If the version matches, the read was consistent. If bit 1 is set (write in progress) or the version advanced, the reader retries.

```rust
// utils/inner_lock.rs
impl<'a> ReadGuard<'a> {
    pub(crate) fn try_read(ptr: *const InnerNode) -> Result<ReadGuard<'a>, TreeError> {
        let node = unsafe { &*ptr };
        let v = node.version_lock.load(Ordering::Acquire);
        if (v & 0b10) == 0b10 {           // bit 1 set = write-locked
            Err(TreeError::Locked)
        } else {
            Ok(Self::new(v, node))        // snapshot the version
        }
    }

    pub(crate) fn check_version(&self) -> Result<u16, TreeError> {
        let v = self.as_ref().version_lock.load(Ordering::Acquire);
        if v == self.version { Ok(v) } else { Err(TreeError::Locked) }
    }
}

impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        self.as_mut().version_lock.fetch_add(0b10, Ordering::Release);  // bump version, clear lock
    }
}
```

`try_read()` snapshots the version and returns `Locked` if a write is in progress. `check_version()` re-reads the version; if it changed, the read was inconsistent. The `WriteGuard`'s `Drop` releases the lock by bumping the version, which tells all concurrent readers "your snapshot is stale, retry."

### Page table entries: custom RwLock with `try_upgrade`

Scenario: Thread A reads a mini-page and finds the record. But the mini-page is in the copy-on-access zone (near eviction). Thread A wants to copy it to the tail. That requires a write lock. Thread A already holds a read lock. Can it upgrade without releasing?

Rust's standard `RwLock` does not support upgrading a read lock to a write lock. Releasing and re-acquiring would create a window where another thread could change the entry. BF-Tree builds a custom `RwLock` with `try_upgrade()`.

The encoding uses a single `AtomicU32`:
- `0` = unlocked
- Bit 0 = writer waiting (used for fairness)
- Even values >= 2 = reader count (each reader adds 2)
- `u32::MAX` = write-locked

```rust
// utils/rw_lock.rs
impl<'a, T> RwLockReadGuard<'a, T> {
    pub fn try_upgrade(self) -> Result<RwLockWriteGuard<'a, T>, RwLockReadGuard<'a, T>> {
        match self.lock.lock_val.compare_exchange_weak(
            2,           // expected: exactly one reader (me)
            u32::MAX,    // desired: write-locked
            Ordering::Acquire, Ordering::Relaxed,
        ) {
            Ok(_) => {
                let lock = self.lock;
                std::mem::forget(self);  // suppress ReadGuard's Drop
                Ok(RwLockWriteGuard { lock })
            }
            Err(_) => Err(self),  // other readers exist, cannot upgrade
        }
    }
}
```

The expected value is 2 because each reader adds 2. A value of 2 means exactly one reader: the caller. If the value is 4 (two readers), upgrade fails because the other reader expects the data to stay consistent.

On success, `mem::forget(self)` suppresses the `ReadGuard`'s `Drop` (which would decrement the reader count). The `WriteGuard` now owns the lock. On failure, the `ReadGuard` is returned to the caller, still valid. The caller can continue reading or give up.

The `try_upgrade()` is best-effort: if other readers are present, upgrade fails and the read path skips the optimization. The system stays correct either way. Copy-on-access and read promotion are performance optimizations, not correctness requirements. When upgrade fails, the read operation completes normally without rescuing the block from eviction; the block may be evicted later, and that is fine.

### Typed errors as control flow

The retry loop in `insert()` uses Rust's `Result` type and pattern matching as a concurrency control mechanism:

```rust
// error.rs
pub(crate) enum TreeError {
    Locked,             // a lock is held by another thread
    CircularBufferFull, // no space, must evict
    NeedRestart,        // optimistic lock detected concurrent modification
}
```

Each error maps to a different recovery action:
- `Locked`: back off (another thread is modifying the same node).
- `CircularBufferFull`: become an evictor (free memory, then retry).
- `NeedRestart`: restart the traversal from the root (the tree structure changed).

This is not exception handling. There is no stack unwinding, no try-catch. Each `Result` is an explicit return value that the caller pattern-matches. The compiler enforces exhaustive matching: if a new variant is added to `TreeError`, every `match` in the codebase must handle it or the build fails. This makes concurrency retry logic visible and auditable at every call site.

The retry loop in `insert()` has no maximum retry count. In practice:
- `CircularBufferFull` resolves after eviction frees space. The calling thread temporarily becomes an evictor, frees enough memory, then retries its write.
- `NeedRestart` resolves after the structural modification (split or merge) that caused the version mismatch completes. On retry, the code sets `aggressive_split = true`, which tells the traversal to preemptively split any full inner nodes on the way down, clearing space before the leaf insert is attempted again.
- `Locked` resolves after the other thread releases the lock. The `Backoff` struct implements exponential backoff: first spin (fast path for short critical sections), then yield (give the OS scheduler a chance to run the lock holder).

Note how each error drives a different recovery action. `CircularBufferFull` requires doing work (eviction). `NeedRestart` requires starting over (the tree structure changed). `Locked` requires waiting (the lock holder will finish soon). Without typed errors, all three would be a generic "retry" that wastes work.

We have built custom locks, state machines, and retry loops. But how do we know they are correct? A single missed interleaving can corrupt the tree. We need a way to systematically test every possible thread schedule.

---

## Section 6: Testing Concurrency with Shuttle

How do you test a concurrent data structure? The naive approach: spawn threads, run random operations, check for crashes. BF-Tree has exactly this test:

```rust
// tests/concurrent.rs
#[test]
fn concurrent_ops() {
    let bf_tree = Arc::new(BfTree::with_config(config, None).unwrap());

    for _ in 0..3 {  // 3 threads
        let tree = bf_tree.clone();
        thread::spawn(move || {
            for _ in 0..400 {  // 400 operations each
                match rng.gen_range(0..4) {
                    0..=1 => { tree.insert(&key, &value); }
                    2     => { tree.read(&key, &mut buffer); }
                    3     => { tree.delete(&key); }
                }
            }
        });
    }
}
```

Three threads, 400 random operations each, keys chosen from a small range so threads collide frequently. Run it with `cargo test`. It passes. Run it 10,000 times. Still passes. Ship to production. A month later, under load, the tree corrupts.

The problem: the OS scheduler picks thread interleavings that happen to avoid the bug. The bug triggers only when Thread B executes between Thread A's copy and Thread A's pointer update. That window is nanoseconds. The OS scheduler almost never hits it.

### What Shuttle does

[Shuttle](https://github.com/awslabs/shuttle) takes control of thread scheduling. Instead of letting the OS decide which thread runs next, Shuttle's scheduler makes that decision at every synchronization point (lock acquire, atomic operation, thread spawn). Run the same test thousands of times with different scheduling decisions. If a bug exists, eventually a schedule will trigger it.

The key insight: Shuttle does not change your test code. The same `concurrent_ops` function runs under both normal `cargo test` and Shuttle. The difference is how threads are scheduled.

### How BF-Tree integrates Shuttle

The integration has two parts.

**Part 1: Compile-time swap.** BF-Tree imports all synchronization primitives through a `sync` module:

```rust
// Everywhere in the codebase:
use crate::sync::thread;
use crate::sync::atomic::AtomicU8;
// Never: use std::sync::...
```

The `sync` module re-exports either `std::sync` or `shuttle::sync` based on a feature flag. When you run `cargo test --features shuttle`, every `thread::spawn`, `Mutex`, and `AtomicU8` in the codebase becomes Shuttle's instrumented version. Production builds use `std` with zero overhead.

**Part 2: The test runner.** Shuttle provides a runner that executes your test function many times with different schedules:

```rust
// tests/concurrent.rs
#[cfg(feature = "shuttle")]
#[test]
fn shuttle_bf_tree_concurrent_operations() {
    let mut runner = shuttle::PortfolioRunner::new(true, config);

    // Run 4 PCT schedulers in parallel, 4000 iterations each
    for _ in 0..4 {
        runner.add(shuttle::scheduler::PctScheduler::new(10, 4_000));
    }

    runner.run(concurrent_ops);  // same test function as before
}
```

`PctScheduler` uses Probabilistic Concurrency Testing: it focuses on schedules with few preemptions (thread switches), because most bugs require only a small number of preemptions to trigger. Running 4 schedulers with 4,000 iterations each explores 16,000 different schedules.

### When a bug is found

If any schedule causes the test to fail (assertion, panic, deadlock), Shuttle saves the schedule to a file. You can replay it:

```rust
#[test]
fn shuttle_replay() {
    shuttle::replay_from_file(concurrent_ops, "target/schedule000.txt");
}
```

This runs the exact same sequence of thread switches that triggered the bug. Set a breakpoint, step through, watch the race condition unfold in slow motion. No more "works on my machine" or "can't reproduce."

### Yield points: helping Shuttle explore

Shuttle controls scheduling at synchronization points. But some code paths spin or busy-wait without hitting a sync point. Under Shuttle's cooperative model (all "threads" are coroutines on one OS thread), spinning starves other threads.

BF-Tree inserts explicit yields at these locations:

```rust
// tree.rs - after a version mismatch retry
Err(TreeError::NeedRestart) => {
    #[cfg(all(feature = "shuttle", test))]
    shuttle::thread::yield_now();  // let the writer finish
    // then retry...
}
```

The pattern: every place where the real code would spin, wait, or back off, the Shuttle build yields instead. This gives Shuttle opportunities to switch threads and explore more interleavings.

---

## Closing: Patterns That Generalize

We followed a mini-page through its entire lifecycle:

- **Birth** (Section 2): The circular buffer bump-allocates a block. A `CircularBufferPtr` guard holds it in `NotReady` state during initialization. Dropping the guard publishes it to `Ready`.
- **Growth** (Section 3): When the mini-page fills up, a `TombstoneHandle` claims the old block via CAS. A new larger block is allocated, records are copied, and `mem::forget` suppresses the handle's `Drop` so the forward transition sticks. Dead blocks are recycled through an intrusive free list.
- **Eviction** (Section 4): A two-phase protocol minimizes lock contention during disk I/O. The eviction callback uses optimistic read-then-verify to handle the race between eviction and concurrent growth. Copy-on-access rescues hot blocks near the eviction boundary, approximating LRU without per-access locking.
- **Concurrency** (Section 5): Optimistic versioned locks let readers traverse inner nodes without blocking. A custom RwLock with `try_upgrade` lets the read path promote records without releasing the lock. Typed errors drive retry loops that handle lock contention, version mismatches, and memory pressure.
- **Testing** (Section 6): Shuttle replaces `std::sync` at compile time, giving a controlled scheduler access to every synchronization point. PCT schedulers run thousands of iterations in parallel, finding interleavings that the OS scheduler would never produce. Failing schedules are saved to disk for deterministic replay.

### The safe/unsafe split

The codebase has roughly 263 lines containing `unsafe` out of about 14,000 total lines of Rust. The unsafe code is concentrated in three places: the circular buffer (raw pointer arithmetic for the ring), the lock implementations (atomics and `UnsafeCell`), and the `LeafNode` layout (`#[repr(C)]` casts). Each unsafe block has a narrow, auditable contract. The tree logic, the retry loops, the state machine transitions, the growth and eviction protocols are all safe Rust that relies on the guards.

### Patterns that generalize

Several patterns from this codebase apply to any system with custom memory management:

**RAII guards for state machines.** `CircularBufferPtr` and `TombstoneHandle` encode lifecycle transitions in `Drop`. The `Drop` always does the safe, conservative thing (publish or revert). The success path uses `mem::forget` to suppress `Drop` and performs the forward transition explicitly. This ensures every code path leaves the state machine in a valid state.

**Typed errors as retry control flow.** `TreeError::Locked`, `NeedRestart`, and `CircularBufferFull` each map to a different recovery strategy. The compiler enforces exhaustive matching.

**Compile-time test infrastructure swaps.** The `sync.rs` module swaps `std::sync` for `shuttle::sync` with a feature flag. Zero cost in production. Systematic concurrency testing in CI.

**Optimistic reads with locked verification.** Read a value without a lock, do work based on it, then re-check under a lock. If the value changed, retry. This pattern appears in eviction callbacks, inner node traversal, and the read promotion path.

These patterns appear wherever systems manage their own memory: GPU buffer managers, network packet pools, shared-memory IPC, and other database engines. The specific data structure (B-tree, LSM-tree, hash table) changes, but the ownership questions remain the same.

### Links

- **Paper**: [BF-Tree: A Read-Write-Optimized Concurrent Larger-Than-Memory Range Index](https://badrish.net/papers/bftree-vldb2024.pdf) (VLDB 2024)
- **Codebase**: [Microsoft bf-tree on crates.io](https://crates.io/crates/bf-tree)
- **Shuttle**: [AWS Labs shuttle on GitHub](https://github.com/awslabs/shuttle)
