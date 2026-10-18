use sendspin::audio::{
    AppliedControls, AudioWork, InstallRejected, LazyPlayerSession, PlaybackControl,
    PlayerEvent, PlayerReceiver, SessionStatus,
};

fn accepts_lazy_session(mut session: LazyPlayerSession) {
    let _status: tokio::sync::watch::Receiver<SessionStatus> = session.status();
    let _controls = session.controls();
    let _permit = session.begin_output_install();
    let _release: fn(&mut LazyPlayerSession) -> Result<(), sendspin::audio::OutputError> =
        LazyPlayerSession::release_output;
    let _install: fn(
        sendspin::audio::OutputInstallPermit,
        PlaybackControl,
    ) -> Result<(), InstallRejected> = sendspin::audio::OutputInstallPermit::install;
    let _receive = session.recv();
    let _event: Option<PlayerEvent> = None;
    let _work: Option<AudioWork> = None;
}

fn main() {
    let _factory = PlayerReceiver::attach_lazy;
    let _controls = AppliedControls { volume: 50, muted: false, static_delay_ms: 0 };
    let _ = accepts_lazy_session;
}
