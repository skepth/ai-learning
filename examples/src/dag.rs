// DAG (Directed Acyclic Graph) implementation for family tree.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet},
    fmt::{Debug, Display},
    rc::Rc,
};

type PersonRef = Rc<RefCell<Person>>;

struct Person {
    id: u32,
    name: String,
    parents: Vec<PersonRef>,
    paths: u32,
}

impl Person {
    // Is target an ancestor of person.
    fn has_path(&self, target: &PersonRef) -> bool {
        for parent in &self.parents {
            if Rc::ptr_eq(parent, target) {
                return true;
            } else if parent.borrow().has_path(target) {
                return true;
            }
        }
        false
    }
}

impl Debug for Person {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.name, self.paths)
    }
}

struct Tree {
    // BTreeMap Fixes unordered hashmap issue.
    people: BTreeMap<u32, PersonRef>,

    id_counter: u32,
}

impl Display for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut graph: Vec<String> = Vec::new();

        graph.push("graph LR".to_string());

        graph.append(
            &mut self
                .people
                .iter()
                .map(|p| p.1.borrow().name.clone())
                .collect::<Vec<String>>(),
        );

        for (_, node) in self.people.iter() {
            let name = node.borrow().name.clone();

            let mut edges = node
                .borrow()
                .parents
                .iter()
                .map(|p| format!("{} --> {}", p.borrow().name, name))
                .collect::<Vec<String>>();

            graph.append(&mut edges);
        }
        let src = graph.join("; ");

        // Issue with rendering, skip.
        // let _output = mermaid_text::render(src.as_str()).unwrap();
        // println!("{output}");
        // write!(f, "{:?}\n\n{}", src, output)
        write!(f, "{}", src)
    }
}

impl Tree {
    fn new() -> Self {
        Tree {
            people: BTreeMap::new(),
            id_counter: 0,
        }
    }

    fn new_person(&mut self, name: &str) -> Rc<RefCell<Person>> {
        self.id_counter += 1;

        let new = Rc::new(RefCell::new(Person {
            id: self.id_counter,
            name: name.to_string(),
            parents: Vec::new(),
            paths: 0,
        }));

        self.people.insert(self.id_counter, new.clone());

        return new;
    }

    fn new_parent_child(
        &self,
        parent: &Rc<RefCell<Person>>,
        child: &Rc<RefCell<Person>>,
    ) -> Result<(), String> {
        if parent.borrow().id == child.borrow().id {
            return Err(String::from("self loop: child and parent are same."));
        }

        match self.people.get(&parent.borrow().id) {
            None => return Err(String::from("Unknown person.")),
            Some(pre) => {
                if !Rc::ptr_eq(pre, &parent) {
                    return Err(String::from("Unknown person from another tree."));
                }
            }
        }

        match self.people.get(&child.borrow().id) {
            None => return Err(String::from("Unknown person.")),
            Some(pre) => {
                if !Rc::ptr_eq(pre, &child) {
                    return Err(String::from("Unknown person from another tree."));
                }
            }
        }

        // Parent Child repationship already exists.
        for p in child.borrow().parents.as_slice() {
            if parent.borrow().id == p.borrow().id {
                return Err(String::from("relationship already exists."));
            }
        }

        // Child cannot have more that 2 parents.
        if child.borrow().parents.len() == 2 {
            return Err(String::from("Too many parents..."));
        }

        if parent.borrow().has_path(child) {
            return Err(String::from("cyclic relationship."));
        }

        child.borrow_mut().parents.push(Rc::clone(&parent));

        Ok(())
    }

    fn reverse_top_sort_from(&self, person: &PersonRef) -> Vec<PersonRef> {
        let mut result: Vec<PersonRef> = Vec::new();
        let mut visited: HashSet<u32> = HashSet::new();

        traverse(person, &mut visited, &mut result);

        result.reverse();

        return result;
    }

    fn backwards(&self, person: &PersonRef) -> Vec<PersonRef> {
        let result = self.reverse_top_sort_from(person);

        // Seed the degree on root node.
        result[0].borrow_mut().paths = 1;

        for node in &result {
            let grad = node.borrow().paths;

            node.borrow()
                .parents
                .iter()
                .for_each(|x| x.borrow_mut().paths += grad);
        }

        return result;
    }

    fn zero(&self) {
        self.people.values().for_each(|x| x.borrow_mut().paths = 0);
    }
}

fn traverse(person: &PersonRef, visited: &mut HashSet<u32>, result: &mut Vec<PersonRef>) {
    if !visited.insert(person.borrow().id) {
        return;
    };
    for parent in &person.borrow().parents {
        traverse(&parent, visited, result);
    }
    result.push(person.clone());
}

fn main() {
    let mut t = Tree::new();

    let a = t.new_person("A");
    let b = t.new_person("B");
    let c = t.new_person("C");
    let d = t.new_person("D");
    let e = t.new_person("E");
    let f = t.new_person("F");

    let x = t.new_person("X");
    let y = t.new_person("Y");
    let z = t.new_person("Z");

    let p = t.new_person("P");

    t.new_parent_child(&a, &c).unwrap();
    t.new_parent_child(&b, &c).unwrap();

    t.new_parent_child(&c, &d).unwrap();

    t.new_parent_child(&d, &e).unwrap();
    t.new_parent_child(&d, &f).unwrap();

    t.new_parent_child(&c, &x).unwrap();

    t.new_parent_child(&x, &y).unwrap();
    t.new_parent_child(&z, &y).unwrap();
    t.new_parent_child(&c, &z).unwrap();

    let w = t.new_person("W");
    t.new_parent_child(&a, &w).unwrap();
    t.new_parent_child(&d, &w).unwrap();

    println!("F has path to C: {}", f.borrow().has_path(&c));
    println!("F has path to X: {}", f.borrow().has_path(&x));
    println!("Y has path to Z: {}", y.borrow().has_path(&z));

    println!("{t}");

    println!("Reverse Top Sort from Y: {:?}", t.reverse_top_sort_from(&y));
    println!("====");
    println!("Reverse Top Sort from F: {:?}", t.reverse_top_sort_from(&f));
    println!("Reverse Top Sort from A: {:?}", t.reverse_top_sort_from(&a));
    println!("Reverse Top Sort from P: {:?}", t.reverse_top_sort_from(&p));

    println!("Reverse Top Sort from W: {:?}", t.reverse_top_sort_from(&w));

    println!("====");

    println!("Reverse Top Sort from Y: {:?}", t.backwards(&y));
    println!("Reverse Top Sort from Y (no zero): {:?}", t.backwards(&y));
    t.zero();
    println!("Reverse Top Sort from Y: {:?}", t.backwards(&y));
}
