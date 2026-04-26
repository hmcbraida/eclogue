//! Shared runtime context used by all tool implementations.
//!
//! Tools need a few shared capabilities (workspace root resolution, diff pagination store,
//! and shared HTTP client). Centralizing these concerns keeps individual tool modules focused
//! on their domain behavior.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use reqwest::Client;
use tokio::sync::Mutex;

/// Immutable shared state cloned into each tool implementation.
#[derive(Clone)]
pub struct ToolContext {
    workspace_root: Arc<PathBuf>,
    diff_store: Arc<Mutex<HashMap<String, String>>>,
    http_client: Client,
}

impl ToolContext {
    /// Returns the workspace root used for resolving relative tool paths.
    pub fn workspace_root(&self) -> &Path {
        self.workspace_root.as_path()
    }

    /// Resolves a path argument against the workspace root.
    ///
    /// - Absolute input paths are returned unchanged.
    /// - Relative input paths are joined against `workspace_root`.
    pub fn resolve_path(&self, path: &str) -> PathBuf {
        let candidate = PathBuf::from(path);
        if candidate.is_absolute() {
            candidate
        } else {
            self.workspace_root.join(candidate)
        }
    }

    /// Normalizes an on-disk path into a model-facing string.
    ///
    /// We prefer workspace-relative paths where possible because they are stable and concise
    /// for model consumption.
    pub fn display_path(&self, path: &Path) -> String {
        path.strip_prefix(self.workspace_root())
            .map(|relative| relative.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"))
    }

    /// Stores a full diff blob so `get_diff` can page it later.
    pub async fn put_diff(&self, diff_id: String, diff: String) {
        let mut guard = self.diff_store.lock().await;
        guard.insert(diff_id, diff);
    }

    /// Returns a previously stored diff by id.
    pub async fn get_diff(&self, diff_id: &str) -> Option<String> {
        let guard = self.diff_store.lock().await;
        guard.get(diff_id).cloned()
    }

    /// Shared reqwest client for network tools.
    pub fn http_client(&self) -> &Client {
        &self.http_client
    }
}

/// Builder for [`ToolContext`].
pub struct ToolContextBuilder {
    workspace_root: Option<PathBuf>,
    http_client: Option<Client>,
}

impl ToolContextBuilder {
    /// Creates a new context builder.
    pub fn new() -> Self {
        Self {
            workspace_root: None,
            http_client: None,
        }
    }

    /// Sets the workspace root.
    pub fn with_workspace_root(mut self, workspace_root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(workspace_root.into());
        self
    }

    /// Overrides the shared HTTP client.
    pub fn with_http_client(mut self, http_client: Client) -> Self {
        self.http_client = Some(http_client);
        self
    }

    /// Finalizes the context, defaulting root to current working directory.
    pub fn build(self) -> std::io::Result<ToolContext> {
        let workspace_root = match self.workspace_root {
            Some(path) => path,
            None => std::env::current_dir()?,
        };

        Ok(ToolContext {
            workspace_root: Arc::new(workspace_root),
            diff_store: Arc::new(Mutex::new(HashMap::new())),
            http_client: self.http_client.unwrap_or_default(),
        })
    }
}

impl Default for ToolContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}
