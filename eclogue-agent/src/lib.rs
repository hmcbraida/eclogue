//! Core abstractions and provider implementations for conversational coding agents.
//!
//! This crate intentionally separates:
//! - The provider-agnostic agent interface (`AgentSession`).
//! - The tool registration and invocation surface (`tooling` module).
//! - Provider-specific implementations (`openai` module).
//!
//! That separation keeps callers independent from the selected LLM backend while still
//! allowing provider-specific construction details (auth, model selection, etc.).

/// Common error types used by agent implementations.
pub mod error;
/// OpenAI-backed implementation of the provider-agnostic agent trait.
pub mod openai;
/// Provider-agnostic agent traits and event payload types.
pub mod session;
/// Tool trait plus a registry builder used during agent initialization.
pub mod tooling;

// Re-export the most commonly used API so downstream users can import from crate root.
pub use error::AgentError;
pub use session::{AgentEvent, AgentEventStream, AgentReply, AgentSession};
pub use tooling::{
    Tool, ToolDefinition, ToolError, ToolRegistry, ToolRegistryBuilder, ToolRegistryError,
};
