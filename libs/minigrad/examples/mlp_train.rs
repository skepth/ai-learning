// Runs a tiny training loop on the Exercise A graph using minigrad.
// a and b are the learnable parameters; c, f and target are fixed data.

use clap::Parser;
use minigrad::{grad::Value, mlp};
use rand::SeedableRng;

/// XOR training example.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of training steps.
    #[arg(short, long, default_value_t = 1000)]
    steps: i32,

    /// Learning rate.
    #[arg(short, long, default_value_t = 0.01)]
    learning_rate: f64,

    /// Rng seed value.
    #[arg(short, long, default_value_t = 10)]
    rng: u64,

    /// Print the loss every N steps.
    #[arg(short, long, default_value_t = 50)]
    print_every: i32,
}

struct DataSet {
    set: Vec<(Vec<Value>, Value)>,
}

fn new_forward(mlp: &mlp::MLP, input: &[Value], target: &Value) -> (Value, Value) {
    let output = mlp.forward(input).unwrap()[0].clone();

    let loss = target.sub(&output).pow(2.0);

    (output, loss)
}

fn update(mlp: &mlp::MLP, lr: f64) {
    for param in mlp.parameters() {
        param.update(lr);
    }
}

fn main() {
    let args = Args::parse();
    let lr = args.learning_rate;
    let steps = args.steps;
    let seed_val = args.rng;

    let dataset = DataSet {
        set: vec![
            (vec![Value::leaf(0.0), Value::leaf(0.0)], Value::leaf(0.0)),
            (vec![Value::leaf(0.0), Value::leaf(1.0)], Value::leaf(1.0)),
            (vec![Value::leaf(1.0), Value::leaf(0.0)], Value::leaf(1.0)),
            (vec![Value::leaf(1.0), Value::leaf(1.0)], Value::leaf(0.0)),
        ],
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed_val);

    let mlp = minigrad::mlp::MLP::new(2)
        .layer(4, minigrad::nn::ActivationType::Tanh)
        .layer(4, minigrad::nn::ActivationType::ReLU)
        .output(1, minigrad::nn::ActivationType::Linear)
        .build(&mut rng);

    println!(
        "steps = {steps}, lr = {lr}, params = {}",
        mlp.parameters().len()
    );
    println!("{:>6} | {:>12}", "step", "total loss");
    println!("{:-<6}-+-{:-<12}", "", "");

    for step in 0..steps {
        let mut total_loss = Value::leaf(0.0);
        for (input, target) in &dataset.set {
            let (_prediction, loss) = new_forward(&mlp, &input, &target);
            total_loss = total_loss.add(&loss);
        }
        if step % args.print_every == 0 || step == steps - 1 {
            println!("{:>6} | {:>12.6}", step, total_loss.data());
        }
        total_loss.zero_grad();
        total_loss.backward();
        update(&mlp, lr);
    }

    // Final report: prediction vs target for each XOR input.
    println!();
    println!(
        "{:>3} {:>3} | {:>6} | {:>10} | {:>6}",
        "x", "y", "target", "prediction", "ok?"
    );
    println!("{:-<7}-+-{:-<6}-+-{:-<10}-+-{:-<6}", "", "", "", "");

    let mut all_ok = true;
    for (input, target) in &dataset.set {
        let (prediction, _loss) = new_forward(&mlp, input, target);
        let ok = (prediction.data() - target.data()).abs() < 0.1;
        all_ok &= ok;
        println!(
            "{:>3} {:>3} | {:>6.1} | {:>10.4} | {:>6}",
            input[0].data(),
            input[1].data(),
            target.data(),
            prediction.data(),
            if ok { "yes" } else { "no" }
        );
    }

    println!();
    println!(
        "{}",
        if all_ok {
            "All four within 0.1 of target. XOR learned."
        } else {
            "Not there yet. Try more steps or a different lr."
        }
    );
}
