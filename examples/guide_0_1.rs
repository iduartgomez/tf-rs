#![allow(non_snake_case)]

extern crate tf_rs as tf;

use tf::prelude::*;

fn main() {
    let root = &mut Scope::new();
    let a = root.placeholder(DataType::Int32);
    let b = Constant::new(root, &[3, 3, 3, 3], [2, 2].as_ref());
    // [[3, 3], [3, 3]]
    let add = ops::add(root, a, b, "").unwrap();

    let mut session = ClientSession::new(root).unwrap();
    // Feed a <- [[1, 2], [3, 4]]
    let feed_a = TypedTensor::<i32>::new(&[2, 2])
        .with_values(&[1, 2, 3, 4])
        .unwrap();
    session.feed(vec![(a, vec![TensorContent::Int32(feed_a)])]);
    let outputs = session.fetch(&[add]).run(None).unwrap();
    let values = match outputs[0] {
        TensorContent::Int32(ref tensor) => tensor.iter().collect::<Vec<_>>(),
        _ => panic!(),
    };
    println!("values: {:?}", &values); // expect [[4, 5], [6, 7]]
    ::std::process::exit(0)
}
