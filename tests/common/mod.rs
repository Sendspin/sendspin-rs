// ABOUTME: Shared test helper for retrying flaky integration tests.
// ABOUTME: Not a test binary itself — `tests/common/` (mod.rs, not
// ABOUTME: common.rs) is the standard way to share code between
// ABOUTME: integration test files without cargo compiling it as its own test.

/// Re-run an async test body up to `attempts` times, succeeding as soon as
/// one attempt doesn't panic and only failing (re-raising the *last*
/// attempt's panic) if every attempt does.
///
/// Use this for tests whose correctness genuinely depends on real mDNS
/// multicast timing — they're exercising real network behavior, and on a
/// LAN shared with other concurrently-running test binaries (and possibly
/// real Sendspin hardware, see tests/dial_discovery.rs's discovery of an
/// actual Home Assistant Voice PE mid-test-run) occasional packet loss or
/// scheduling delay is expected, not a signal that the code under test is
/// wrong. Retrying absorbs that environmental noise; it does not mask a test
/// that fails the *same way* every time, since it still fails after
/// `attempts` tries.
///
/// `test_fn` must be a zero-argument async function (not a closure capturing
/// local state) so each retry gets a fresh, independent attempt —
/// `tokio::spawn` both isolates a panicking attempt from tearing down the
/// whole test process and gives us a `JoinError` to detect that panic.
// Each integration test file that does `mod common;` compiles its own copy
// of this module, and no single test file uses both helpers — hence the
// `allow`, rather than genuinely dead code.
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
