# Introduction

`tf-rs` is a library built on top of the TensorFlow C API that allows the composition and computation of data flow graphs.

## TensorFlow

TensorFlow™ is an open source software library for numerical computation using data flow graphs. To use `tf-rs` familiarity with the TensorFlow framework is required. Visit the [official website](https://www.tensorflow.org/) for more information.

## C API

For maximum compatibility and to avoid fragmentation and redundancy, the low level interaction is done through the "unoffficial-official" bindings, located at the [tensorflow/rust](https://github.com/tensorflow/rust) repository. In order to use this library you must be able to build/install the requeriments.
