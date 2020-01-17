//! A high level (Pythonic) API for the TensorFlow framework.

#![allow(unused_variables)]

#[macro_use]
extern crate error_chain;
extern crate num_complex;
extern crate tensorflow as tf;
extern crate uuid;

#[macro_use]
mod macros;
mod errors;
#[macro_use]
pub mod ops;
pub mod train;
pub(crate) mod client;
pub(crate) mod framework;

pub type OperationData = tf::Operation;
pub type TensorData<T> = tf::Tensor<T>;
pub type TensorShape = tf::Shape;
use tf::{OperationDescription, Output};
use tf::{DataType, Graph, Session, SessionOptions};
use tf::{BFloat16, QInt16, QInt32, QUInt16, QUInt8};
use num_complex::{Complex32, Complex64};

pub mod prelude {
    pub use super::framework::{Attribute, Constant, GetOp, NodeIdent, Operation, Scope, ShapeOps,
                               Tensor, TensorArray, TensorContent, TensorOps, Variable};
    pub use super::client::ClientSession;
    pub use super::{OperationData, TensorData};
    pub use super::errors::Error as TFError;
    pub use tf::{DataType, Status};

    pub use super::train;
    pub use super::ops;
}
