#![allow(non_snake_case)]

extern crate tf_rs as tf;

use tf::prelude::*;

fn main() {
    let root = &mut Scope::new();
    // Matrix A = [3 2; -1 0]
    let A = Constant::new(root, &[3.0_f32, 2., -1., 0.], [2, 2].as_ref());
    // Vector b = [3 5]
    let b = Constant::new(root, &[3.0_f32, 5., 0., 0.], [2, 2].as_ref());
    // v = Ab^T
    let v = ops::matmul(root, A, b, false, true, false, false, false, false, "v").unwrap();

    let outputs = {
        let mut session = ClientSession::new(root).unwrap();
        // Run and fetch v
        session.fetch(&[v]).run(None).unwrap()
    };
    let values = match outputs[0] {
        TensorContent::Float(ref tensor) => tensor.iter().cloned().collect::<Vec<_>>(),
        _ => panic!(),
    };
    println!("values: {:?}", &values); // expect [19, 0, -3, 0]
    ::std::process::exit(0)
}
