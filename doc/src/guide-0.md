# Getting started

The API is designed to be simple and concise: graph operations are expressed using a "functional" construction style, including easy specification of names, etc., and the resulting graph can be efficiently run and the desired outputs fetched in a few lines of code. This guide explains the basic concepts and data structures needed to get started with TensorFlow graph construction and execution in Rust. This guide was "translated" from the original C++ guide.

## The Basics

Let's start with a simple example that illustrates graph construction and execution using the Rust API.

```Rust
// examples/guide_0_0.rs
extern crate tf_rs as tf;

use tf::prelude::*;

fn main() {
    let root = &mut Scope::new();
    // Matrix A = [3, 2; -1, 0]
    let A = Constant::new(root, &[3.0_f32, 2., -1., 0.], &[2, 2]);
    // Vector b = [3, 5; 0, 0]
    let b = Constant::new(root, &[3.0_f32, 5., 0., 0.], &[2, 2]);
    // v = Ab^T
    let v =  ops::matmul(
            root, A, b, 
            false, true, false, false, false, false, "v")
        .unwrap();

    let outputs = {
        let mut session = ClientSession::new(root).unwrap();
        // Run and fetch v
        session.fetch(&[v]).run(None).unwrap()
    };
    let values = match outputs[0] {
        TensorContent::Float(ref tensor) => {
            tensor.iter().cloned().collect::<Vec<_>>()
        }       
        _ => panic!() 
    };
    println!("values: {:?}", &values); // expect [19, 0; -3, 0]
    ::std::process::exit(0)
}
```

Build a cargo binary project and write this code in the main file. You should be able to run it using `cargo run`.

This example shows some of the important features of the Rust API such as the following:

* Constructing tensor constants from a slice of values and a shape
* Constructing and naming of TensorFlow operations
* Executing and fetching the tensor values from the TensorFlow session.

We will delve into the details of each below.

## Graph Construction

### Scope

tensorflow::Scope is the main data structure that holds the current state of graph construction. A Scope acts as a handle to the graph being constructed, as well as storing TensorFlow operation properties. The Scope object is the first argument to operation constructors, and operations that use a given Scope as their first argument inherit that Scope's properties, such as a common name prefix. Multiple Scopes can refer to the same graph, as explained further below.

Create a new Scope object by calling the Scope constructor `Scope::new`. This creates some resources such as a graph to which operations are added. It also creates a tensorflow::Status object which will be used to indicate errors encountered when constructing operations.

The Scope object returned by Scope::new is referred to as the root scope. "Child" scopes can be constructed from the root scope by calling various methods of the Scope type, thus forming a hierarchy of scopes. A child scope inherits all of the properties of the parent scope and typically has one property added or changed. For instance, `scope.name_scope("name", None)` appends name to the prefix of names for operations created using the returned Scope object.

Here are some of the properties controlled by a Scope object:

* Operation names
* Set of control dependencies for an operation
* Create new variables, constants or placeholders under the current scope
* _Device placement for an operation_ (not implemented yet)
* _Kernel attribute for an operation_ (not implemented yet)

Please refer to Scope documentation in the API docs for the complete list of member functions that let you create child scopes with new properties.

### Operation Constructors

You can create graph operations with operation constructors, one function per TensorFlow operation.

The first parameter for all operation constructors is always a Scope object. Each operation requires a number of Tensor inputs and mandatory attributes form the rest of the arguments.

For optional arguments, constructors have an optional parameters that allows optional attributes.

The arguments and return values of operations are handled in different ways depending on their type. They can be either a single Tensor, a tuple of tensors or a vector of tensors.

### Graph Execution

When executing a graph, you will need a session. The C++ API provides a tensorflow::ClientSession class that will execute ops created by the operation constructors. TensorFlow will automatically determine which parts of the graph need to be executed, and what values need feeding. For example:

```Rust
let mut root = Scope::new();
let a = Constant::new(root, &[1, 1], &[2]);
let b = Constant::new(root, &[2, 2], &[2]);
let add = ops::add(root, a, b, "").unwrap();

let mut session = ClientSession::new(root).unwrap();
let outputs = session.fetch(&[add]).run(None).unwrap();
// outputs[0] == [3, 3]
```

Similarly, the object returned by the operation constructor can be used as the argument to specify a value being fed when executing the graph. Furthermore, the value to feed can be specified with the different kinds of Rust values used to specify tensor constants. For example:

```Rust
// examples/guide_0_1.rs
let root = &mut Scope::new();
let a = root.placeholder(DataType::Int32);
let b = Constant::new(root, &[3, 3, 3, 3], &[2, 2]);
// b = [[3, 3], [3, 3]]
let add = ops::add(root, a, b, "").unwrap();

let mut session = ClientSession::new(root).unwrap();
// Feed a <- [[1, 2], [3, 4]]
let feed_a = {
    let mut t = TypedTensor::<i32>::new(&[2, 2]);
    for (i, x) in [1, 2, 3, 4].iter().enumerate() {
        t[i] = *x;
    }
    t
};
session.feed(vec![(a, vec![TensorContent::Int32(feed_a)])]);
let outputs = session.fetch(&[add]).run(None).unwrap();
let values = match outputs[0] {
    TensorContent::Int32(ref tensor) => {
        tensor.iter().collect::<Vec<_>>()
    }       
    _ => panic!() 
};
println!("values: {:?}", &values); // expect [[4, 5], [6, 7]]
```

Please see the Tensor documentation in the API docs for more information on how to use the execution output.
