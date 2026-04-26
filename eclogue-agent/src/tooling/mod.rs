//! Tooling module for model-callable local tools.
//!
//! This module is intentionally split by responsibility:
//! - [`protocol`] contains shared JSON protocol helpers and normalized tool errors.
//! - [`registry`] contains tool registration and invocation dispatch.
//! - [`context`] contains shared runtime state passed to concrete tools.
//! - [`tools`] contains one implementation module per tool.

pub mod context;
pub mod protocol;
pub mod registry;
pub mod tools;

pub use context::{ToolContext, ToolContextBuilder};
pub use protocol::{ToolError, ToolErrorCode, next_request_id};
pub use registry::{Tool, ToolDefinition, ToolRegistry, ToolRegistryBuilder, ToolRegistryError};
pub use tools::register_default_tools;
