//! Tokenizer lib re-implements Byte Pair Encoding (BPE)
//! and other related concepts over text data.
//! Tokenizer's job is to encode and decode text for models.
//! Key Concepts that we will cover:
//!  * How to split text so the model can effectively learn its meaning while
//!    training.
//!  * How do we encode text so it can be compressed into efficient packages as
//!    well as decompressed later.

use std::collections::{BTreeMap, HashMap};

/// A sequence of token IDs.
///
/// Text starts as UTF-8 bytes, where each byte (`u8`) is a value from 0 to 255.
/// BPE merges create new IDs starting at 256, which don't fit in a byte, so
/// each element uses a wider integer type. Before any merges, the sequence is
/// just the text's bytes. After merges, it's a mix of byte IDs and merged IDs.
#[derive(Debug, PartialEq, Eq)]
struct TokenIds {
    data: Vec<u64>,
}

impl TokenIds {
    /// New creates a new instance of TokenId from the given &str.
    pub fn new(input: &str) -> Self {
        TokenIds {
            data: input.bytes().map(u64::from).collect(),
        }
    }

    /// Merge is a helper that takes TokenIds, merges all of the repeating
    /// pairs, from left to right, with the new ID and returns the merged
    /// TokenIds.
    fn merge(&self, pair: (u64, u64), id: u64) -> TokenIds {
        let mut result: Vec<u64> = vec![];
        let mut data = self.data.iter().copied().peekable();

        while let Some(current) = data.next() {
            if current == pair.0 && data.next_if_eq(&pair.1).is_some() {
                result.push(id);
                // data.next_if_eq points the iterator to after the pair.
            } else {
                result.push(current);
            }
        }
        TokenIds { data: result }
    }
}

/// Merges represents the learned dictionary of merges and repective ordering
/// that is used for encoding repeating pairs TokenIds.
///
/// NOTE: Merges is also used to derive vocabulary needed for decoding.
type Merges = HashMap<(u64, u64), u64>;

/// Vocabulary represents the ordered Token ID to flattened tokens mapping
/// needed for decoding.
type Vocabulary = BTreeMap<u64, Vec<u8>>;

/// GetMostCommonPairs returns the most repeating pair of ID's in TokenIds.
fn GetMostCommonPairs(input: TokenIds) -> Option<(u64, u64)> {
    todo!()
}

/// Train processes the TokenIds and generates the tokenizers learned Merges in
/// this crate.
///
/// The steps param is used to control how deep to run the BPE. A negative
/// number indicates that we run this until we have no more repeating pairs.
fn Train(input: TokenIds, steps: i64) -> Merges {
    todo!()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn merges_test() {
        struct Test {
            name: &'static str,
            input: &'static str,
            pair: (u64, u64),
            id: u64,
            want: Vec<u64>,
        }

        let tests: Vec<Test> = vec![
            Test {
                name: "empty",
                input: "",
                pair: (97, 97), // (a, a)
                id: 288,
                want: vec![],
            },
            Test {
                name: "single",
                input: "a",
                pair: (97, 97), // (a, a)
                id: 288,
                want: vec![97],
            },
            Test {
                name: "absent",
                input: "abc",
                pair: (97, 97), // (a, a)
                id: 288,
                want: vec![97, 98, 99],
            },
            Test {
                name: "whole input",
                input: "ab",
                pair: (97, 98), // (a, b)
                id: 288,
                want: vec![288],
            },
            Test {
                name: "end",
                pair: (99, 99), // (c, c)
                input: "abcc",
                id: 288,
                want: vec![97, 98, 288],
            },
            Test {
                name: "even",
                input: "aaaa",
                pair: (97, 97), // (a, a)
                id: 288,
                want: vec![288, 288],
            },
            Test {
                name: "odd",
                input: "aaa",
                pair: (97, 97), // (a, a)
                id: 288,
                want: vec![288, 97],
            },
            Test {
                name: "basic",
                input: "aaabdaaabac",
                pair: (97, 97), // (a, a)
                id: 256,
                want: vec![256, 97, 98, 100, 256, 97, 98, 97, 99],
            },
            Test {
                name: "with spaces",
                input: "ab ab  ab",
                pair: (32, 32), // (space, space)
                id: 288,
                want: vec![97, 98, 32, 97, 98, 288, 97, 98],
            },
            Test {
                name: "complex utf-8",
                input: "aaé",
                pair: (0xC3, 0xA9), // é's two bytes
                id: 288,
                want: vec![97, 97, 288],
            },
        ];

        for t in tests {
            let got = TokenIds::new(t.input).merge(t.pair, t.id);
            assert_eq!(got.data, t.want, "case: {}", t.name);
        }
    }

    #[test]
    fn double_merge_test() {
        let input = TokenIds::new("aabaabab");

        let merged = input
            .merge((u64::from('a'), u64::from('a')), 256)
            .merge((256, 98), 257);

        assert_eq!(merged.data, vec![257, 257, 97, 98]);
    }
}
