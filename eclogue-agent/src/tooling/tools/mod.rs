//! Concrete built-in tool implementations.
//!
//! Each tool lives in its own module to keep behavior focused and maintainable.

mod edit_file;
mod fetch_url;
mod get_diff;
mod list_files;
mod read_file;
mod run_command;
mod search_text;
mod stat_file;
mod util;

use crate::tooling::context::ToolContext;
use crate::tooling::registry::ToolRegistryBuilder;

use self::edit_file::EditFileTool;
use self::fetch_url::FetchUrlTool;
use self::get_diff::GetDiffTool;
use self::list_files::ListFilesTool;
use self::read_file::ReadFileTool;
use self::run_command::RunCommandTool;
use self::search_text::SearchTextTool;
use self::stat_file::StatFileTool;

/// Registers the full default tool suite defined in `tools.md` (except `web_search`).
pub fn register_default_tools(
    builder: ToolRegistryBuilder,
    context: ToolContext,
) -> ToolRegistryBuilder {
    builder
        .register_tool(ListFilesTool::new(context.clone()))
        .register_tool(SearchTextTool::new(context.clone()))
        .register_tool(StatFileTool::new(context.clone()))
        .register_tool(ReadFileTool::new(context.clone()))
        .register_tool(EditFileTool::new(context.clone()))
        .register_tool(GetDiffTool::new(context.clone()))
        .register_tool(RunCommandTool::new(context.clone()))
        .register_tool(FetchUrlTool::new(context))
}
