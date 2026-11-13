// Runs a tiny training loop on the Exercise A graph using minigrad.
// a and b are the learnable parameters; c, f and target are fixed data.

use clap::Parser;
use minigrad::grad::Value;

/// Training Example.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of passes.
    #[arg(short, long, default_value_t = 20)]
    steps: i32,

    /// Learning Rate.
    #[arg(short, long, default_value_t = 0.001)]
    learning_rate: f64,
}

/// Builds a fresh graph from the current parameter values.
/// Returns (prediction, loss).
fn forward(a: &Value, b: &Value, c: &Value, f: &Value, target: &Value) -> (Value, Value) {
    let e = a.mul(b);
    let d = e.add(c);
    let prediction = d.mul(f);

    // loss = (target - prediction)^2
    let loss = target.sub(&prediction).pow(2.0);

    (prediction, loss)
}

fn main() {
    let args = Args::parse();
    let lr = args.learning_rate;
    let steps = args.steps;

    // Parameters: created once, updated every step.
    let a = Value::leaf(2.0);
    let b = Value::leaf(-3.0);

    // Fixed data: same every step (we're overfitting one example).
    let c = Value::leaf(10.0);
    let f = Value::leaf(-2.0);
    let target = Value::leaf(4.0);

    println!("target = {:.4}, lr = {lr}", target.data());
    println!(
        "{:>4} | {:>10} | {:>12} | {:>8} | {:>8}",
        "step", "prediction", "loss", "a", "b"
    );

    for step in 0..steps {
        // 1. Forward: new graph from the current a and b.
        let (prediction, loss) = forward(&a, &b, &c, &f, &target);

        // 2. Zero: clears stale grads (the new graph includes a and b).
        loss.zero_grad();

        // 3. Backward: fills in a.grad and b.grad.
        loss.backward();

        println!(
            "{:>4} | {:>10.4} | {:>12.4} | {:>8.4} | {:>8.4}",
            step,
            prediction.data(),
            loss.data(),
            a.data(),
            b.data()
        );

        // 4. Update: only the parameters.
        a.update(args.learning_rate);
        b.update(args.learning_rate);
    }
}
