use sendspin::audio::{BufferPool, Sample};

#[test]
fn test_buffer_pool_creation() {
    let pool = BufferPool::new(10, 1024);
    assert_eq!(pool.capacity(), 1024);
}

#[test]
fn test_buffer_pool_get_and_return() {
    let pool = BufferPool::new(5, 1024);

    // Get a buffer
    let mut buf = pool.get();
    assert_eq!(buf.capacity(), 1024);

    // Modify it
    buf.extend_from_slice(&vec![Sample::ZERO; 100]);
    assert_eq!(buf.len(), 100);

    // Return it
    pool.put(buf);

    // Get another - should be the same buffer (reused)
    let buf2 = pool.get();
    assert_eq!(buf2.capacity(), 1024);
    assert_eq!(buf2.len(), 0); // Should be cleared
}

#[test]
fn test_buffer_pool_fallback_allocation() {
    let pool = BufferPool::new(2, 1024);

    // Get all buffers from pool
    let _buf1 = pool.get();
    let _buf2 = pool.get();

    // This should allocate a new buffer
    let buf3 = pool.get();
    assert_eq!(buf3.capacity(), 1024);
}
