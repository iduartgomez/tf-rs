use super::*;
use super::framework::*;

mod moving_averages;
pub use self::moving_averages::*;

mod slot_creator;
use self::slot_creator::*;

pub mod nn;

fn validate_convnet_data_dormat(data_format: &str) -> Result<&'static str, ::Error> {
    match data_format {
        "NHWC" => Ok("NHWC"),
        "NCHW" => Ok("NCHW"),
        _ => Err(::Error::Stub),
    }
}

fn _validate_convnet_3d_data_dormat(data_format: &str) -> Result<&'static str, ::Error> {
    match data_format {
        "NDHWC" => Ok("NDHWC"),
        "NCDHW" => Ok("NCDHW"),
        _ => Err(::Error::Stub),
    }
}

/////////////////////////////////
//   Pre-built train models    //
/////////////////////////////////

/*
#[derive(Debug, Clone)]
pub(crate) struct Optimizer;

/*
beta1: Option<f64>, // 0.9
beta2: Option<f64>, // 0.999
epsilon: Option<f64>, // 1e-08
*/

impl Optimizer {
    pub fn new<V>(kind: Optimizers, learning_rate: V, use_locking: bool) -> Self
        where V: Into<Tensor>
    {
        unimplemented!()
    }

    pub fn configure(&mut self) -> &mut OptimizerInterface {
        unimplemented!()
    }

    pub fn composite_optimizer(optimizer1: Optimizer,
                               optimizer2: Optimizer,
                               switch: bool,
                               use_locking: bool)
                               -> Self {
        unimplemented!()
    }

    pub fn compute_gradients(&mut self,
                             loss: Tensor,
                             var_list: Vec<Tensor>)
                             -> Vec<(Option<Tensor>, Tensor)> {
        unimplemented!()
    }

    pub fn apply_gradients(&mut self,
                           grads_and_vars: Vec<(Tensor, Tensor)>,
                           global_step: Tensor)
                           -> Tensor {
        unimplemented!()
    }
}

pub(crate) trait OptimizerInterface {
    fn add_param(&mut self, name: &str, value: f64) {
        unimplemented!()
    }
}

pub enum Optimizers {
    GradientDescent,
    Adam,
    LazyAdam,
    Momentum,
    Composite,
}

// GradientDescentOptimizer
#[derive(Clone)]
struct GradientDescentOptimizer;

impl GradientDescentOptimizer {}


// AdamOptimizer
#[derive(Clone)]
struct AdamOptimizer;

impl AdamOptimizer {}


// LazyAdamOptimizer
#[derive(Clone)]
struct LazyAdamOptimizer;

impl LazyAdamOptimizer {}

// MomentumOptimizer
#[derive(Clone)]
struct MomentumOptimizer;

impl MomentumOptimizer {}


// CompositeOptimizer
#[derive(Clone)]
struct CompositeOptimizer;

impl CompositeOptimizer {}


#[derive(Debug, Clone)]
pub struct ExponentialDecay;

impl ExponentialDecay {
    pub fn new(base_rate: Tensor,
               step_var: Tensor,
               decay_steps: u32,
               decay_base: f64,
               staircase: bool)
               -> ExponentialDecay {
        ExponentialDecay
    }

    pub fn run(self) -> Result<Tensor, ::Error> {
        unimplemented!()
    }
}
*/
