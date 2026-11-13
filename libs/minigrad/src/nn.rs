// NN represents a unit of neural network.

use rand::distr::{Distribution, Uniform};
use std::error::Error;

use crate::grad::Value;

#[derive(Clone, Copy)]
pub enum ActivationType {
    Tanh,
    ReLU,
    Linear,
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(
        inputs: usize,
        neurons: usize,
        rng: &mut rand::rngs::StdRng,
        activation: ActivationType,
    ) -> Self {
        let mut nns: Vec<Neuron> = Vec::new();

        for _ in 0..neurons {
            nns.push(Neuron::new(inputs, rng, activation));
        }

        Layer { neurons: nns }
    }

    pub fn forward(&self, inputs: &[Value]) -> Result<Vec<Value>, Box<dyn Error>> {
        let mut outputs: Vec<Value> = Vec::new();

        for nn in &self.neurons {
            outputs.push(nn.forward(inputs)?);
        }
        Ok(outputs)
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut result: Vec<Value> = Vec::new();

        for nn in &self.neurons {
            result.append(&mut nn.parameters())
        }

        result
    }
}

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    activation: ActivationType,
}

impl Neuron {
    pub fn new(inputs: usize, rng: &mut rand::rngs::StdRng, activation: ActivationType) -> Self {
        let mut weights: Vec<Value> = Vec::new();
        let dist = Uniform::new_inclusive(-1.0, 1.0).unwrap();

        for _ in 0..inputs {
            let weight = Value::leaf(dist.sample(rng));
            weights.push(weight);
        }
        Neuron {
            weights: weights,
            bias: Value::leaf(0.0),
            activation,
        }
    }

    pub fn forward(&self, inputs: &[Value]) -> Result<Value, Box<dyn Error>> {
        if self.weights.len() != inputs.len() {
            return Err(format!("weights and values are of different lengths.").into());
        }

        let mut result = self.bias.clone();
        for (weight, input) in self.weights.iter().zip(inputs) {
            result = result.add(&weight.mul(input));
        }

        let result = match self.activation {
            ActivationType::Tanh => result.tanh(),
            ActivationType::Linear => result,
            ActivationType::ReLU => result.relu(),
        };

        Ok(result)
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut result = self.weights.clone();
        result.push(self.bias.clone());

        result
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    fn approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn test_basic_nn() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);

        let nn = Neuron::new(3, &mut rng, ActivationType::Linear);

        let a = Value::leaf(2.0);
        let b = Value::leaf(-3.0);
        let c = Value::leaf(1.0);
        let out = nn.forward(&[a.clone(), b.clone(), c.clone()]).unwrap();

        let w = nn.weights;

        // Forward Pass test.
        assert_eq!(
            out.data(),
            (2.0 * w[0].data() - 3.0 * w[1].data() + w[2].data())
        );

        out.backward();
        // Backward Pass test.
        approx_eq(w[0].grad(), 2.0);
        approx_eq(w[1].grad(), -3.0);
        approx_eq(w[2].grad(), 1.0);
        approx_eq(a.grad(), w[0].data());
        approx_eq(b.grad(), w[1].data());
        approx_eq(c.grad(), w[2].data());
        approx_eq(nn.bias.grad(), 1.0);
    }

    #[test]
    fn test_tanh_nn() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);

        let nn = Neuron::new(3, &mut rng, ActivationType::Tanh);

        let a = Value::leaf(2.0);
        let b = Value::leaf(-3.0);
        let c = Value::leaf(1.0);
        let out = nn.forward(&[a.clone(), b.clone(), c.clone()]).unwrap();

        let w = nn.parameters();
        let expected_out = (2.0 * w[0].data() - 3.0 * w[1].data() + w[2].data()).tanh();

        // Forward Pass test.
        approx_eq(out.data(), expected_out);

        out.backward();
        // Backward Pass test.
        let k = 1.0 - expected_out.powf(2.0);
        approx_eq(w[0].grad(), 2.0 * k);
        approx_eq(w[1].grad(), -3.0 * k);
        approx_eq(w[2].grad(), 1.0 * k);
        approx_eq(a.grad(), w[0].data() * k);
        approx_eq(b.grad(), w[1].data() * k);
        approx_eq(c.grad(), w[2].data() * k);
        approx_eq(nn.bias.grad(), k);
    }

    #[test]
    fn test_tanh_layer() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);

        let layer = Layer::new(3, 2, &mut rng, ActivationType::Tanh);

        let a = Value::leaf(2.0);
        let b = Value::leaf(-3.0);
        let c = Value::leaf(1.0);
        let out = layer.forward(&[a.clone(), b.clone(), c.clone()]).unwrap();

        // We have 2 neurons in this layer so we expect 2 outputs.
        assert_eq!(out.len(), 2);

        let w = layer.parameters();

        // We need the weights to be different for wach neurons.
        assert_ne!(w[0].data(), w[4].data());
        assert_ne!(w[1].data(), w[5].data());
        assert_ne!(w[2].data(), w[6].data());

        let expected_out_0 = (2.0 * w[0].data() - 3.0 * w[1].data() + w[2].data()).tanh();
        let expected_out_1 = (2.0 * w[4].data() - 3.0 * w[5].data() + w[6].data()).tanh();

        // Forward Pass test.
        approx_eq(out[0].data(), expected_out_0);
        approx_eq(out[1].data(), expected_out_1);

        // Sum of out's.
        let loss = out[0].add(&out[1]);
        approx_eq(loss.data(), expected_out_0 + expected_out_1);

        // Backward Pass test.
        loss.backward();

        let k_0 = 1.0 - expected_out_0.powf(2.0);
        let k_1 = 1.0 - expected_out_1.powf(2.0);

        approx_eq(w[0].grad(), 2.0 * k_0);
        approx_eq(w[1].grad(), -3.0 * k_0);
        approx_eq(w[2].grad(), 1.0 * k_0);

        approx_eq(w[4].grad(), 2.0 * k_1);
        approx_eq(w[5].grad(), -3.0 * k_1);
        approx_eq(w[6].grad(), 1.0 * k_1);

        approx_eq(a.grad(), (w[0].data() * k_0) + (w[4].data() * k_1));
        approx_eq(b.grad(), w[1].data() * k_0 + (w[5].data() * k_1));
        approx_eq(c.grad(), w[2].data() * k_0 + (w[6].data() * k_1));

        approx_eq(layer.neurons[0].bias.grad(), k_0);
        approx_eq(layer.neurons[1].bias.grad(), k_1);
    }
}
