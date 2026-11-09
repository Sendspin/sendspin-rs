// ABOUTME: Shared test helper for retrying flaky mDNS-multicast integration tests

/// Re-run an async test body up to `attempts` times, succeeding as soon as one
/// attempt doesn't panic and only failing (re-raising the last attempt's panic)
/// if every attempt does.
///
/// Intended for tests that depend on real mDNS multicast timing, where
/// occasional packet loss or scheduling delay on a shared LAN is environmental
/// noise rather than a real failure. A test that fails the same way every time
/// still fails after `attempts` tries.
///
/// `test_fn` must be a zero-argument async function so each retry gets a fresh
/// attempt; `tokio::spawn` isolates a panicking attempt and yields a `JoinError`
/// to detect it.
/// Whether real-network tests (which use live mDNS multicast) are enabled, via
/// the `SENDSPIN_NET_TESTS` environment variable. Off by default so an ordinary
/// `cargo test` run doesn't depend on multicast being available.
#[allow(dead_code)]
pub fn net_tests_enabled() -> bool {
    std::env::var_os("SENDSPIN_NET_TESTS").is_some()
}

// Each integration test file that does `mod common;` compiles its own copy, and
// no single file uses both helpers — hence the allow.
#[allow(dead_code)]
pub async fn retry_flaky<F, Fut>(attempts: u32, test_fn: F)
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    for attempt in 1..=attempts {
        match tokio::spawn(test_fn()).await {
            Ok(()) => return,
            Err(join_err) if attempt < attempts => {
                eprintln!(
                    "flaky test attempt {attempt}/{attempts} failed ({join_err}), retrying..."
                );
            }
            Err(join_err) => std::panic::resume_unwind(join_err.into_panic()),
        }
    }
}

/// Synchronous counterpart to [`retry_flaky`], for plain `#[test]` functions
/// that don't need a tokio runtime (e.g. tests using `mdns_sd`'s blocking
/// `recv_timeout` directly rather than `ClientBrowser`'s async API).
#[allow(dead_code)]
pub fn retry_flaky_sync<F>(attempts: u32, test_fn: F)
where
    F: Fn(),
{
    for attempt in 1..=attempts {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(&test_fn)) {
            Ok(()) => return,
            Err(payload) if attempt < attempts => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                eprintln!("flaky test attempt {attempt}/{attempts} failed ({msg}), retrying...");
            }
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
}
