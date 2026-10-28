use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};

use crate::audio::{CommitRejected, OutputError, SessionStatus};
use crate::generation::StreamGeneration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoundaryAction {
    Clear,
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProducerError {
    ReceiverClosed,
    GenerationExhausted,
}

#[derive(Clone, Default)]
pub(crate) struct Urgent {
    pub clear: Option<StreamGeneration>,
    pub end: Option<StreamGeneration>,
    pub closed: bool,
}

/// Custom role output callbacks run synchronously; callback errors are terminal for
/// that session and the first error is retained.
pub(crate) trait RoleAdapter: Send + 'static {
    type Prepared: Send + 'static;
    fn commit(&mut self, prepared: Self::Prepared) -> Result<(), OutputError>;
    fn boundary(&mut self, action: BoundaryAction) -> Result<(), OutputError>;
    fn cleanup(&mut self) -> Result<(), OutputError>;
}

struct State<A: RoleAdapter> {
    floor: StreamGeneration,
    status: SessionStatus,
    closed: bool,
    output: Option<A>,
    cleaned: bool,
}

pub(crate) struct Shared<A: RoleAdapter> {
    state: Mutex<State<A>>,
    pub urgent: watch::Sender<Urgent>,
    pub status: watch::Sender<SessionStatus>,
    pub terminated: Arc<AtomicBool>,
}

pub(crate) struct Coordinator<A: RoleAdapter> {
    pub shared: Arc<Shared<A>>,
}

pub(crate) struct Work<A: RoleAdapter> {
    pub shared: Arc<Shared<A>>,
    pub generation: StreamGeneration,
}

impl<A: RoleAdapter> Coordinator<A> {
    pub fn with_urgent(
        output: A,
        urgent: watch::Sender<Urgent>,
        terminated: Arc<AtomicBool>,
    ) -> Self {
        let (status, _) = watch::channel(SessionStatus::Open);
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    floor: StreamGeneration::initial(),
                    status: SessionStatus::Open,
                    closed: false,
                    output: Some(output),
                    cleaned: false,
                }),
                urgent,
                status,
                terminated,
            }),
        }
    }

    pub fn work(&self, generation: StreamGeneration) -> Work<A> {
        Work {
            shared: Arc::clone(&self.shared),
            generation,
        }
    }

    pub fn reconcile_urgent(&self) -> Result<(), OutputError> {
        let snapshot = self.shared.urgent.borrow().clone();
        let mut boundaries = Vec::new();
        if let Some(generation) = snapshot.clear {
            boundaries.push((generation, BoundaryAction::Clear));
        }
        if let Some(generation) = snapshot.end {
            boundaries.push((generation, BoundaryAction::End));
        }
        boundaries.sort_by_key(|(generation, _)| *generation);
        for (generation, action) in boundaries {
            self.boundary(generation, action)?;
        }
        Ok(())
    }

    pub fn boundary(
        &self,
        generation: StreamGeneration,
        action: BoundaryAction,
    ) -> Result<(), OutputError> {
        let mut state = self.shared.state.lock();
        if state.closed || !matches!(state.status, SessionStatus::Open) || generation <= state.floor
        {
            return Ok(());
        }
        state.floor = generation;
        let Some(mut output) = state.output.take() else {
            return Ok(());
        };
        let result = output.boundary(action);
        match result {
            Ok(()) => {
                state.output = Some(output);
                Ok(())
            }
            Err(error) => {
                fail(&mut state, output, error.clone());
                let _ = self.shared.status.send_replace(state.status.clone());
                self.shared.terminated.store(true, Ordering::Release);
                Err(error)
            }
        }
    }

    pub fn close(&self) {
        let mut state = self.shared.state.lock();
        if state.closed {
            return;
        }
        state.closed = true;
        if matches!(state.status, SessionStatus::Open) {
            state.status = SessionStatus::Closed;
            let _ = self.shared.status.send_replace(SessionStatus::Closed);
        }
        self.shared.terminated.store(true, Ordering::Release);
        if let Some(mut output) = state.output.take() {
            if !state.cleaned {
                let _ = output.cleanup();
                state.cleaned = true;
            }
        }
        self.shared
            .urgent
            .send_modify(|urgent| urgent.closed = true);
    }

    pub fn terminal(&self) -> bool {
        let state = self.shared.state.lock();
        self.shared.terminated.load(Ordering::Acquire)
            || state.closed
            || !matches!(state.status, SessionStatus::Open)
    }

    pub fn status_watch(&self) -> watch::Receiver<SessionStatus> {
        self.shared.status.subscribe()
    }

    pub fn current_generation(&self) -> StreamGeneration {
        self.shared.state.lock().floor
    }

    pub fn fail(&self, error: OutputError) {
        let mut state = self.shared.state.lock();
        if state.closed || !matches!(state.status, SessionStatus::Open) {
            return;
        }
        if !state.cleaned {
            if let Some(mut output) = state.output.take() {
                let _ = output.cleanup();
                state.cleaned = true;
            }
        }
        state.status = SessionStatus::Failed(error.clone());
        let _ = self.shared.status.send_replace(state.status.clone());
        self.shared.terminated.store(true, Ordering::Release);
        self.shared
            .urgent
            .send_modify(|urgent| urgent.closed = true);
    }
}

impl<A: RoleAdapter> Work<A> {
    pub fn commit(self, prepared: A::Prepared) -> Result<(), CommitRejected> {
        let coordinator = Coordinator {
            shared: Arc::clone(&self.shared),
        };
        coordinator
            .reconcile_urgent()
            .map_err(CommitRejected::Output)?;
        let mut state = self.shared.state.lock();
        if self.shared.terminated.load(Ordering::Acquire)
            || state.closed
            || !matches!(state.status, SessionStatus::Open)
        {
            return Err(CommitRejected::Closed);
        }
        if self.generation != state.floor {
            return Err(CommitRejected::Obsolete);
        }
        let Some(mut output) = state.output.take() else {
            return Err(CommitRejected::Closed);
        };
        match output.commit(prepared) {
            Ok(()) => {
                state.output = Some(output);
                Ok(())
            }
            Err(error) => {
                let rejection = CommitRejected::Output(error.clone());
                fail(&mut state, output, error);
                let _ = self.shared.status.send_replace(state.status.clone());
                self.shared.terminated.store(true, Ordering::Release);
                Err(rejection)
            }
        }
    }
}

fn fail<A: RoleAdapter>(state: &mut State<A>, mut output: A, error: OutputError) {
    if !matches!(state.status, SessionStatus::Open) || state.closed {
        return;
    }
    if !state.cleaned {
        let _ = output.cleanup();
        state.cleaned = true;
    }
    state.status = SessionStatus::Failed(error);
}

pub(crate) fn advance(
    generation: &Mutex<StreamGeneration>,
) -> Result<StreamGeneration, ProducerError> {
    let mut current = generation.lock();
    let next = current
        .checked_advance()
        .ok_or(ProducerError::GenerationExhausted)?;
    *current = next;
    Ok(next)
}

pub(crate) fn publish_boundary(
    urgent: &watch::Sender<Urgent>,
    action: BoundaryAction,
    generation: StreamGeneration,
) {
    urgent.send_modify(|snapshot| match action {
        BoundaryAction::Clear => snapshot.clear = Some(generation),
        BoundaryAction::End => snapshot.end = Some(generation),
    });
}

pub(crate) fn channel<T>() -> (mpsc::UnboundedSender<T>, mpsc::UnboundedReceiver<T>) {
    mpsc::unbounded_channel()
}
