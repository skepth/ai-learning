//! Tokenizer lib re-implements Byte Pair Encoding (BPE)
//! and other related concepts over text data.
//! Tokenizer's job is to encode and decode text for models.
//! Key Concepts that we will cover:
//!  * How to split text so the model can effectively learn its meaning while
//!    training.
//!  * How do we encode text so it can be compressed into efficient packages as
//!    well as decompressed later.

mod token_ids;
mod tokenizer;
pub use tokenizer::Tokenizer;
