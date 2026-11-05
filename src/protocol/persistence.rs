use parking_lot::Mutex;
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;

/// Stores opaque byte blobs under fixed keys chosen by the library.
///
/// The library owns every blob's format, so an implementation stores the
/// bytes as given and must not parse them. Today the library keeps the
/// client's static identity key here, which must survive restarts: losing it
/// gives the client a new `client_id`.
pub trait PersistenceProvider: Send + Sync {
    /// Return the blob stored under `key`, or `None` when there is none.
    fn load(&self, key: &str) -> io::Result<Option<Vec<u8>>>;

    /// Store `value` under `key`, replacing any earlier blob.
    fn save(&self, key: &str, value: &[u8]) -> io::Result<()>;
}

/// Keeps each blob in its own file inside a directory.
#[derive(Debug, Clone)]
pub struct FileStore {
    dir: PathBuf,
}

impl FileStore {
    /// Store blobs in `dir`, which is created on the first save.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
}

impl PersistenceProvider for FileStore {
    fn load(&self, key: &str) -> io::Result<Option<Vec<u8>>> {
        match std::fs::read(self.dir.join(key)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn save(&self, key: &str, value: &[u8]) -> io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        // Write a sibling file first so a crash never leaves a truncated blob.
        let staged = self.dir.join(format!("{key}.tmp"));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(&staged)?;
        file.write_all(value)?;
        file.sync_all()?;
        std::fs::rename(staged, self.dir.join(key))
    }
}

/// Keeps blobs in memory only, so they are lost when the process exits.
#[derive(Debug, Default)]
pub struct MemoryStore {
    blobs: Mutex<HashMap<String, Vec<u8>>>,
}

impl MemoryStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PersistenceProvider for MemoryStore {
    fn load(&self, key: &str) -> io::Result<Option<Vec<u8>>> {
        Ok(self.blobs.lock().get(key).cloned())
    }

    fn save(&self, key: &str, value: &[u8]) -> io::Result<()> {
        self.blobs.lock().insert(key.to_string(), value.to_vec());
        Ok(())
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn file_store_keeps_blobs_private_to_the_owner() {
        let dir = std::env::temp_dir().join(format!("sendspin-store-{}", std::process::id()));
        let store = FileStore::new(&dir);
        store.save("identity", b"secret").unwrap();
        let mode = std::fs::metadata(dir.join("identity"))
            .unwrap()
            .permissions()
            .mode();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(mode & 0o777, 0o600);
    }
}
