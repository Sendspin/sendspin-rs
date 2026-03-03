# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/Sendspin/sendspin-rs/compare/v0.1.0...v0.1.1) - 2026-03-03

### Added

- implement clock synchronization with server ([#18](https://github.com/Sendspin/sendspin-rs/pull/18))
- add software volume/mute control and fix playback start blip ([#10](https://github.com/Sendspin/sendspin-rs/pull/10))

### Fixed

- add 16 bit PCM fallback for Music Assistant 2.7 ([#24](https://github.com/Sendspin/sendspin-rs/pull/24))
- validate channels > 0 at construction to prevent callback panic ([#14](https://github.com/Sendspin/sendspin-rs/pull/14))
- reject time conversions when filter drift is implausible ([#13](https://github.com/Sendspin/sendspin-rs/pull/13))
- five clock sync filter correctness issues ([#12](https://github.com/Sendspin/sendspin-rs/pull/12))
- correct f32 sample normalization and extract Sample::to_f32() ([#11](https://github.com/Sendspin/sendspin-rs/pull/11))

### Other

- create release workflow ([#25](https://github.com/Sendspin/sendspin-rs/pull/25))
- Add instructions for deps. ([#23](https://github.com/Sendspin/sendspin-rs/pull/23))
- consolidate mutex acquisitions in audio callback ([#17](https://github.com/Sendspin/sendspin-rs/pull/17))
- use a typed builder  ([#7](https://github.com/Sendspin/sendspin-rs/pull/7))
- add devcontainer ([#21](https://github.com/Sendspin/sendspin-rs/pull/21))
- Allow custom SSL and auth via `IntoClientRequest` ([#8](https://github.com/Sendspin/sendspin-rs/pull/8))
