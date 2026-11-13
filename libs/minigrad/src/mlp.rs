// MLP is a Multi Layer Perceptron construction of neuron layers.

use std::error::Error;

use crate::{
    grad::Value,
    nn::{ActivationType, Layer},
};

pub struct MLP {
    layers: Vec<Layer>,
}

struct LayerOptions {
    size: usize,
    activation: ActivationType,
}

pub struct MlpBuilder {
    input_size: usize,
    layers: Vec<LayerOptions>,
}

impl MlpBuilder {
    pub fn layer(mut self, size: usize, activation: ActivationType) -> Self {
        self.layers.push(LayerOptions { size, activation });
        self
    }

    pub fn output(mut self, size: usize, activation: ActivationType) -> Self {
        self.layers.push(LayerOptions { size, activation });
        self
    }

    pub fn build(&self, rng: &mut rand::rngs::StdRng) -> MLP {
        let mut layers: Vec<Layer> = Vec::new();

        let mut prev_out_size = self.input_size;
        for opt in &self.layers {
            layers.push(Layer::new(prev_out_size, opt.size, rng, opt.activation));
            prev_out_size = opt.size;
        }

        MLP { layers }
    }
}

impl MLP {
    pub fn new(inputs: usize) -> MlpBuilder {
        MlpBuilder {
            input_size: inputs,
            layers: vec![],
        }
    }

    pub fn forward(&self, inputs: &[Value]) -> Result<Vec<Value>, Box<dyn Error>> {
        let mut previous_outputs = inputs.to_vec();

        for layer in &self.layers {
            previous_outputs = layer.forward(&previous_outputs)?;
        }

        Ok(previous_outputs)
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut results: Vec<Value> = Vec::new();

        for layer in &self.layers {
            results.extend(layer.parameters());
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_mlp() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);

        let mlp = MLP::new(2)
            .layer(4, ActivationType::Tanh)
            .layer(4, ActivationType::Tanh)
            .output(1, ActivationType::Linear)
            .build(&mut rng);

        let a = Value::leaf(2.0);
        let b = Value::leaf(-3.0);

        let out = mlp.forward(&[a.clone(), b.clone()]).unwrap();

        assert_eq!(out.len(), 1);

        // 1: w0, w1, b => 3 params and 4 times for 4 neurons therefore total = 12;
        // 2: w0...w3, b => 5 params and 4 times = 20
        // 3: 5 params and only 1 time = 5
        assert_eq!(mlp.parameters().len(), 37);
    }
}
