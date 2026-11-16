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

    /// Encode uses the merges to apply BPE on the input text and returns the
    /// token IDs.
    pub fn encode(&self, text: &str) -> Vec<u64> {
        let mut tokens = TokenIds::new(text);

        while let Some((pair, id)) = self.lowest_rank_pair(&tokens) {
            tokens = tokens.merge(pair, id);
        }

        tokens.data()
    }

    /// Helper function that returns the lowest ranked merge pair & its ID.
    fn lowest_rank_pair(&self, tokens: &TokenIds) -> Option<((u64, u64), u64)> {
        tokens
            // Get iterator over pairs.
            .pairs()
            // Now only filter for pairs that exist in self.merges.
            // Then return only iterator over Option<(pair, id)>.
            .filter_map(|pair| self.merges.get(&pair).map(|&id| (pair, id)))
            // Now iterate over and only select the item where the id is lowest.
            .min_by_key(|&(_, id)| id)
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

    #[test]
    fn encode_test() {
        // Merges: (a,a)->256, (256,a)->257, (257,b)->258
        let tokenizer = Tokenizer::train("aaabdaaabac", Some(3));

        struct Test {
            name: &'static str,
            input: &'static str,
            want: Vec<u64>,
        }

        let tests = vec![
            Test {
                name: "empty",
                input: "",
                want: vec![],
            },
            Test {
                name: "no merges apply",
                input: "xyz",
                want: vec![120, 121, 122],
            },
            // Regression: a repeating pair that isn't a merge must not hang.
            Test {
                name: "repeats but not a merge",
                input: "xyxy",
                want: vec![120, 121, 120, 121],
            },
            Test {
                name: "training input",
                input: "aaabdaaabac",
                want: vec![258, 100, 258, 97, 99],
            },
            Test {
                name: "chained merges",
                input: "aaab",
                want: vec![258],
            },
            Test {
                name: "partial chain",
                input: "aaa",
                want: vec![257],
            },
            // (256, b) is not a merge, so encoding stops.
            Test {
                name: "stops early",
                input: "aab",
                want: vec![256, 98],
            },
            // Step 4: (a,a) is the only merge present.
            Test {
                name: "rank order",
                input: "abaa",
                want: vec![97, 98, 256],
            },
            // Step 6: greedy longest-match would give [257, 97].
            Test {
                name: "bpe vs greedy",
                input: "aaaa",
                want: vec![256, 256],
            },
            // Characters outside training stay as raw UTF-8 bytes.
            Test {
                name: "multi-byte",
                input: "é",
                want: vec![195, 169],
            },
        ];

        for t in tests {
            assert_eq!(tokenizer.encode(t.input), t.want, "case: {}", t.name);
        }
    }
}
