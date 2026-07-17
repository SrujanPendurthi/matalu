//! Post-ASR text correction seam.
//!
//! v1 relies on Nemotron's built-in punctuation/casing, so the wired-in
//! corrector is [`PassThrough`]. The trait is the extension point for the
//! context-aware grammar/formatting LLM described in the plan: a future
//! `OllamaCorrector` (HTTP to a local model) or an in-process `llama.cpp`
//! implementation can be dropped in without touching the ASR engine.

/// Refines raw ASR text (grammar, punctuation, formatting).
pub trait Corrector: Send + Sync {
    /// Return a cleaned-up version of `text`. Must be cheap enough to run on
    /// finalized utterances (and, for continuous refinement, on partials).
    fn refine(&self, text: &str) -> String;
}

/// Identity corrector — emits ASR text unchanged.
pub struct PassThrough;

impl Corrector for PassThrough {
    fn refine(&self, text: &str) -> String {
        text.to_string()
    }
}
