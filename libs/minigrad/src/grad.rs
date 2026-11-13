// Package minigrad

use std::collections::HashSet;
use std::fmt::Debug;
use std::{cell::RefCell, rc::Rc};

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone)]
pub struct Value(Rc<RefCell<Node>>);

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0.borrow())
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        let id_eq = self.0.borrow().id == other.0.borrow().id;
        // let data_eq = self.0.borrow().data == other.0.borrow().data;
        // let grad_eq = self.0.borrow().grad == other.0.borrow().grad;

        if id_eq {
            return true;
        }

        return false;
    }
}

// #[derive(Debug)]
pub enum Op {
    Leaf,
    Add,
    Sub,
    Mul,
    Pow(f64),
    Tanh,
    ReLU,
}

// Define a global atomic u64 counter initialized to 0
static COUNTER: AtomicU64 = AtomicU64::new(0);

fn increment_counter() -> u64 {
    // fetch_add increments the value and returns the previous value
    COUNTER.fetch_add(1, Ordering::Relaxed) + 1
}

struct Node {
    id: u64,
    data: f64,
    grad: f64,
    parents: Vec<Value>,
    op: Op,
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "  |ID: {}, DATA: {}, GRAD: {}|  ",
            self.id,
            self.data,
            self.grad // "  |ID: {}|  ",
                      // self.id
        )
    }
}

impl Node {
    fn new(data: f64, op: Op, parents: Vec<Value>) -> Value {
        Value(Rc::new(RefCell::new(Node {
            id: increment_counter(),
            data,
            grad: 0.0,
            parents,
            op,
        })))
    }
}

impl Value {
    pub fn leaf(data: f64) -> Value {
        Node::new(data, Op::Leaf, vec![])
    }
    pub fn add(&self, other: &Value) -> Value {
        Node::new(
            self.data() + other.data(),
            Op::Add,
            vec![self.clone(), other.clone()],
        )
    }
    pub fn sub(&self, other: &Value) -> Value {
        Node::new(
            self.data() - other.data(),
            Op::Sub,
            vec![self.clone(), other.clone()],
        )
    }
    pub fn mul(&self, other: &Value) -> Value {
        Node::new(
            self.data() * other.data(),
            Op::Mul,
            vec![self.clone(), other.clone()],
        )
    }
    pub fn pow(&self, exp: f64) -> Value {
        Node::new(self.data().powf(exp), Op::Pow(exp), vec![self.clone()])
    }

    pub fn tanh(&self) -> Value {
        Node::new(self.data().tanh(), Op::Tanh, vec![self.clone()])
    }

    pub fn relu(&self) -> Value {
        Node::new(f64::max(self.data(), 0.0), Op::ReLU, vec![self.clone()])
    }

    pub fn backward(&self) -> Vec<f64> {
        let path = self.reverse_top_sort();
        let mut result: Vec<f64> = Vec::new();

        // Seed the loss gradiant.
        path[0].0.borrow_mut().grad = 1.0;

        for node in &path {
            let parents = node.parents();
            match node.0.borrow().op {
                Op::Leaf => {
                    // Do nothing as grads are already accumulated.
                }
                Op::Add => {
                    parents
                        .iter()
                        .for_each(|p| p.0.borrow_mut().grad += node.grad());
                }
                Op::Sub => {
                    let left = &parents[0];
                    let right = &parents[1];
                    left.0.borrow_mut().grad += node.grad();
                    right.0.borrow_mut().grad -= node.grad();
                }
                Op::Mul => {
                    let left = &parents[0];
                    let right = &parents[1];
                    left.0.borrow_mut().grad += right.data() * node.grad();
                    right.0.borrow_mut().grad += left.data() * node.grad();
                }
                Op::Pow(k) => {
                    parents.iter().for_each(|p| {
                        p.0.borrow_mut().grad += k * p.data().powf(k - 1.0) * node.grad()
                    });
                }
                Op::Tanh => {
                    parents.iter().for_each(|p| {
                        p.0.borrow_mut().grad += (1.0 - (node.data()).powf(2.0)) * node.grad()
                    });
                }
                Op::ReLU => {
                    parents.iter().for_each(|p| {
                        let mut dx_relu = 0.0;
                        if node.data() > 0.0 {
                            dx_relu = 1.0;
                        }
                        p.0.borrow_mut().grad += dx_relu * node.grad()
                    });
                }
            }
            result.push(node.grad())
        }

        return result;
    }

    pub fn zero_grad(&self) {
        let path = self.reverse_top_sort();

        for node in &path {
            node.0.borrow_mut().grad = 0.0;
        }
    }

    pub fn update(&self, learning_rate: f64) {
        let curr_data = self.data();
        let curr_grad = self.grad();
        self.0.borrow_mut().data = curr_data - (curr_grad * learning_rate)
    }

    pub fn id(&self) -> u64 {
        self.0.borrow().id
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn parents(&self) -> Vec<Value> {
        let mut result: Vec<Value> = Vec::new();

        for p in &self.0.borrow().parents {
            result.push(p.clone());
        }

        return result;
    }

    pub fn has_parent(&self, id: u64) -> bool {
        for p in &self.0.borrow().parents {
            if p.0.borrow().id == id {
                return true;
            }
        }

        return false;
    }

    fn reverse_top_sort(&self) -> Vec<Value> {
        let mut result: Vec<Value> = Vec::new();
        let mut visited: HashSet<u64> = HashSet::new();

        traverse(self, &mut result, &mut visited);
        result.reverse();

        return result;
    }
}

fn traverse(node: &Value, result: &mut Vec<Value>, visited: &mut HashSet<u64>) {
    if !visited.insert(node.id()) {
        return;
    }
    for parent in node.parents() {
        traverse(&parent, result, visited);
    }
    result.push(node.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_pass() {
        let a = Value::leaf(2.0);
        let b = Value::leaf(-3.0);
        let c = Value::leaf(10.0);
        let f = Value::leaf(-2.0);
        let e = a.mul(&b);
        let d = e.add(&c);
        let l = d.mul(&f);

        assert_eq!(e.data(), -6.0);
        assert_eq!(d.data(), 4.0);
        assert_eq!(l.data(), -8.0);

        assert_eq!(l.reverse_top_sort(), vec![l, f, d, c, e, b, a]);
    }

    #[test]
    fn test_top_sort() {
        let a = Value::leaf(2.0);
        let b = Value::leaf(-3.0);
        let c = Value::leaf(10.0);
        let f = Value::leaf(-2.0);
        let e = a.mul(&b);
        let d = e.add(&c);
        let l = d.mul(&f);

        assert_eq!(l.reverse_top_sort(), vec![l, f, d, c, e, b, a]);
    }

    #[test]
    fn test_backward_pass() {
        let a = Value::leaf(2.0);
        let b = Value::leaf(-3.0);
        let c = Value::leaf(10.0);
        let f = Value::leaf(-2.0);
        let e = a.mul(&b);
        let d = e.add(&c);
        let l = d.mul(&f);

        assert_eq!(l.backward(), vec![1.0, 4.0, -2.0, -2.0, -2.0, -4.0, 6.0]);
        l.zero_grad();
        assert_eq!(l.backward(), vec![1.0, 4.0, -2.0, -2.0, -2.0, -4.0, 6.0]);
    }

    #[test]
    fn test_with_loss_op() {
        let a = Value::leaf(2.0);
        let b = Value::leaf(-3.0);
        let c = Value::leaf(10.0);
        let f = Value::leaf(-2.0);
        let e = a.mul(&b);
        let d = e.add(&c);
        let l = d.mul(&f);

        // target.
        let target = Value::leaf(4.0);

        // l is prediction.

        // loss is (target - prediction)^2.
        let int = target.sub(&l);
        let loss = int.pow(2.0);

        assert_eq!(
            loss.backward(),
            vec![
                1.0, 24.0, -24.0, -96.0, 48.0, 48.0, 48.0, 96.0, -144.0, 24.0
            ]
        );
    }

    #[test]
    fn test_with_loss_lr_op() {
        let a = Value::leaf(2.0);
        let b = Value::leaf(-3.0);
        let c = Value::leaf(10.0);
        let f = Value::leaf(-2.0);
        let e = a.mul(&b);
        let d = e.add(&c);
        let l = d.mul(&f);

        // target.
        let target = Value::leaf(4.0);

        // l is prediction.

        // loss is (target - prediction)^2.
        let int = target.sub(&l);
        let loss = int.pow(2.0);
        loss.backward();

        let old_loss = loss.data();
        let old_prediction = l.data();

        // Update the inout parameters.
        a.update(0.01);
        b.update(0.01);

        loss.zero_grad();

        // Forward pass.
        let e = a.mul(&b);
        let d = e.add(&c);
        let l = d.mul(&f);

        // New loss is (target - prediction)^2.
        let int = target.sub(&l);
        let loss = int.pow(2.0);
        loss.backward();

        println!("Target: {}", target.data());
        println!(
            "Old Prediction: {old_prediction:.4}, New Prediction: {:.4}",
            l.data(),
        );
        println!("Old loss: {old_loss:.4}, New loss: {:.4}", loss.data());

        assert!(
            loss.data() < old_loss,
            "\n\nOld Loss: {}\nNew Loss: {}\n\n",
            old_loss,
            loss.data()
        );
    }

    #[test]
    fn test_tanh() {
        let x = Value::leaf(0.5);
        let y = x.tanh();

        y.backward();

        assert_eq!(x.grad(), 1.0 - f64::tanh(0.5).powf(2.0));
    }

    #[test]
    fn test_relu() {
        let x = Value::leaf(3.0);
        let y = x.relu();

        y.backward();
        assert_eq!(x.grad(), 1.0);

        let x = Value::leaf(-2.0);
        let y = x.relu().add(&x);
        y.backward();

        assert_eq!(x.grad(), 1.0);
    }
}
