//! Admissibility of `server/activate` for the PSK that keyed the session.

use crate::protocol::messages::{Activity, GoodbyeReason, ServerActivate};
use crate::protocol::noise::PskCategory;

/// Apply `activation` on top of the roles persisted by earlier activations
/// (`None` before the first one).
///
/// Returns the activation with `active_roles` filled in, or the goodbye
/// reason the client closes the connection with.
pub(crate) fn admit(
    persisted_roles: Option<&[String]>,
    activation: ServerActivate,
    psk: PskCategory,
    unpaired_access: bool,
) -> Result<ServerActivate, GoodbyeReason> {
    let active_roles = activation
        .active_roles
        .or_else(|| persisted_roles.map(<[String]>::to_vec))
        .unwrap_or_default();
    let admissible = |unpaired_access: bool| {
        let allowed = |activities: &[Activity]| match psk {
            PskCategory::LongTerm => !activities.contains(&Activity::Pairing),
            PskCategory::Pairing | PskCategory::Sentinel => {
                unpaired_access || !activities.contains(&Activity::Playback)
            }
        };
        let playback_capable = || {
            let mut with_playback = activation.activities.clone();
            with_playback.push(Activity::Playback);
            allowed(&with_playback)
        };
        allowed(&activation.activities) && (active_roles.is_empty() || playback_capable())
    };
    if admissible(unpaired_access) {
        return Ok(ServerActivate {
            activities: activation.activities,
            active_roles: Some(active_roles),
        });
    }
    let unpaired_session = psk != PskCategory::LongTerm;
    if unpaired_session && !unpaired_access && admissible(true) {
        Err(GoodbyeReason::PairingRequired)
    } else {
        Err(GoodbyeReason::Unauthorized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn activation(activities: &[Activity], roles: Option<&[&str]>) -> ServerActivate {
        ServerActivate {
            activities: activities.to_vec(),
            active_roles: roles.map(|roles| roles.iter().map(|r| r.to_string()).collect()),
        }
    }

    #[test]
    fn unpaired_playback_needs_unpaired_access() {
        let playback = activation(&[Activity::Playback], Some(&["player@v1"]));
        assert!(admit(None, playback.clone(), PskCategory::Sentinel, true).is_ok());
        assert_eq!(
            admit(None, playback, PskCategory::Sentinel, false),
            Err(GoodbyeReason::PairingRequired)
        );
    }

    #[test]
    fn roles_on_an_unpaired_idle_connection_need_unpaired_access() {
        let idle_with_roles = activation(&[], Some(&["controller@v1"]));
        assert_eq!(
            admit(None, idle_with_roles, PskCategory::Sentinel, false),
            Err(GoodbyeReason::PairingRequired)
        );
    }

    #[test]
    fn pairing_on_a_long_term_session_is_unauthorized() {
        assert_eq!(
            admit(
                None,
                activation(&[Activity::Pairing], Some(&[])),
                PskCategory::LongTerm,
                true
            ),
            Err(GoodbyeReason::Unauthorized)
        );
    }

    #[test]
    fn omitted_roles_persist_from_the_previous_activation() {
        let persisted = vec!["player@v1".to_string()];
        let admitted = admit(
            Some(&persisted),
            activation(&[Activity::Playback], None),
            PskCategory::LongTerm,
            false,
        )
        .unwrap();
        assert_eq!(admitted.active_roles, Some(persisted));
    }

    #[test]
    fn a_first_activation_without_roles_activates_none() {
        let admitted = admit(None, activation(&[], None), PskCategory::Sentinel, false).unwrap();
        assert_eq!(admitted.active_roles, Some(vec![]));
    }
}
