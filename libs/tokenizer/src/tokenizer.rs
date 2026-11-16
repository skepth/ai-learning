use std::collections::{BTreeMap, HashMap};

use crate::token_ids::TokenIds;

/// Merges represents the learned dictionary of merges and respective ordering
/// that is used for encoding repeating pairs TokenIds.
///
/// NOTE: Merges is also used to derive vocabulary needed for decoding.
pub type Merges = HashMap<(u64, u64), u64>;

/// Vocabulary represents the ordered Token ID to flattened tokens mapping
/// needed for decoding.
type Vocabulary = BTreeMap<u64, Vec<u8>>;

/// Tokenizer is the core public API that holds merges and vocabulary.
pub struct Tokenizer {
    merges: Merges,
    vocab: Vocabulary,
}

impl Tokenizer {
    /// Train processes text and generates the tokenizers learned Merges
    /// in this crate.
    ///
    /// The steps param is used to control how deep to run the BPE. A None input
    /// indicates that we run this until we have no more repeating pairs.
    pub fn train(text: &str, steps: Option<usize>) -> Self {
        let mut data = TokenIds::new(text);
        let mut merges: Merges = HashMap::new();
        let mut id = 256;

        let limit = steps.unwrap_or(usize::MAX);
        for _ in 0..limit {
            let Some(common_pair) = data.get_most_common_pair() else {
                break;
            };
            data = data.merge(common_pair, id);
            merges.insert(common_pair, id);
            id += 1;
        }

        Tokenizer {
            merges,
            vocab: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_test() {
        let tokenizer = Tokenizer::train("aaabdaaabac", Some(3));
        let want = HashMap::from([
            ((257, 98), 258),
            ((97, 97), 256),
            ((256, 97), 257),
        ]);
        assert_eq!(tokenizer.merges, want)
    }

    #[test]
    fn training_inf_test() {
        let tokenizer = Tokenizer::train("aaabdaaabac", None);
        let want = HashMap::from([
            ((257, 98), 258),
            ((97, 97), 256),
            ((256, 97), 257),
        ]);
        assert_eq!(tokenizer.merges, want)
    }
}
