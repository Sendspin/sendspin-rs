use crate::generation::PlaybackResetGeneration;
use crate::log_sampling::should_log_sample;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

const MAX_DEFERRED_DIAGNOSTICS: usize = 8;

#[derive(Clone, Copy)]
pub(crate) enum PlaybackDiagnostic {
    Enqueued {
        timestamp: i64,
        duration_us: i64,
        enqueue_count: u64,
        queued_us: u64,
        buffers: usize,
        cursor_us: i64,
        reset: PlaybackResetGeneration,
    },
    Cleared {
        queued_us: u64,
        buffers: usize,
        cursor_us: i64,
        reset: PlaybackResetGeneration,
    },
}

#[cfg(test)]
type DiagnosticObserver = Box<dyn FnMut(PlaybackDiagnostic)>;

struct DiagnosticState {
    pending: [Option<PlaybackDiagnostic>; MAX_DEFERRED_DIAGNOSTICS],
    len: usize,
    depth: usize,
    emitting: bool,
    #[cfg(test)]
    observer: Option<DiagnosticObserver>,
}

impl DiagnosticState {
    const fn new() -> Self {
        Self {
            pending: [None; MAX_DEFERRED_DIAGNOSTICS],
            len: 0,
            depth: 0,
            emitting: false,
            #[cfg(test)]
            observer: None,
        }
    }
}

thread_local! {
    static STATE: RefCell<DiagnosticState> = const { RefCell::new(DiagnosticState::new()) };
}

/// Defers diagnostics until the outermost scope on this thread is dropped.
/// Scopes nest LIFO; forgetting one leaves the depth stuck and the queue unflushed.
pub(crate) struct DiagnosticScope {
    _not_send: PhantomData<Rc<()>>,
}

impl DiagnosticScope {
    pub(crate) fn new() -> Self {
        STATE.with(|state| state.borrow_mut().depth += 1);
        Self {
            _not_send: PhantomData,
        }
    }
}

impl Drop for DiagnosticScope {
    fn drop(&mut self) {
        let pending = STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.depth -= 1;
            if state.depth != 0 {
                return None;
            }
            let pending = std::mem::replace(&mut state.pending, [None; MAX_DEFERRED_DIAGNOSTICS]);
            let len = state.len;
            state.len = 0;
            Some((pending, len))
        });
        if let Some((pending, len)) = pending {
            for diagnostic in pending.into_iter().take(len).flatten() {
                emit_playback_diagnostic(diagnostic);
            }
        }
    }
}

pub(crate) fn record_playback_diagnostic(diagnostic: PlaybackDiagnostic) {
    let deferred = STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.depth != 0 {
            if state.len < MAX_DEFERRED_DIAGNOSTICS {
                let index = state.len;
                state.pending[index] = Some(diagnostic);
                state.len += 1;
            }
            true
        } else {
            false
        }
    });
    if !deferred {
        emit_playback_diagnostic(diagnostic);
    }
}

fn emit_playback_diagnostic(diagnostic: PlaybackDiagnostic) {
    let already_emitting = STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.emitting {
            true
        } else {
            state.emitting = true;
            false
        }
    });
    if already_emitting {
        return;
    }
    struct EmittingGuard;
    impl Drop for EmittingGuard {
        fn drop(&mut self) {
            STATE.with(|state| state.borrow_mut().emitting = false);
        }
    }
    let _guard = EmittingGuard;

    #[cfg(test)]
    STATE.with(|state| {
        let callback = state.borrow_mut().observer.take();
        if let Some(callback) = callback {
            struct ObserverGuard<'a> {
                state: &'a RefCell<DiagnosticState>,
                callback: Option<DiagnosticObserver>,
            }
            impl Drop for ObserverGuard<'_> {
                fn drop(&mut self) {
                    if let Some(callback) = self.callback.take() {
                        self.state.borrow_mut().observer = Some(callback);
                    }
                }
            }
            let mut callback_guard = ObserverGuard {
                state,
                callback: Some(callback),
            };
            if let Some(callback) = callback_guard.callback.as_mut() {
                callback(diagnostic);
            }
        }
    });

    match diagnostic {
        PlaybackDiagnostic::Enqueued {
            timestamp,
            duration_us,
            enqueue_count,
            queued_us,
            buffers,
            cursor_us,
            reset,
        } => {
            if log::log_enabled!(log::Level::Trace) && should_log_sample(enqueue_count) {
                log::trace!("Audio buffer enqueued: enqueue={enqueue_count}, ts={timestamp}µs, duration={:.1}ms, queued={:.1}ms, buffers={buffers}, cursor={cursor_us}µs, playback_reset_generation={reset:?}", duration_us as f64 / 1000.0, queued_us as f64 / 1000.0);
            }
        }
        PlaybackDiagnostic::Cleared {
            queued_us,
            buffers,
            cursor_us,
            reset,
        } => {
            if log::log_enabled!(log::Level::Debug) {
                log::debug!("Cleared playback queue: queued={:.1}ms, buffers={buffers}, cursor={cursor_us}µs, playback_reset_generation={reset:?}", queued_us as f64 / 1000.0);
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn install_diagnostic_observer(observer: impl FnMut(PlaybackDiagnostic) + 'static) {
    STATE.with(|state| state.borrow_mut().observer = Some(Box::new(observer)));
}

#[cfg(test)]
pub(crate) fn clear_diagnostic_observer() {
    STATE.with(|state| state.borrow_mut().observer = None);
}
