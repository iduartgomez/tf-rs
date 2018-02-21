//! Error module.

use tf::Status;

error_chain! {
    errors {
        Stub
        UndefinedFuncAttr
        UndefinedGrad
        UndefinedTensorShape
        OpNotFound
        TensorNotFound
        UnimplTraitMethod
    }

    foreign_links {
        FFINulError(::std::ffi::NulError);
        Utf8Error(::std::str::Utf8Error);
    }
}

impl From<Status> for Error {
    fn from(err: Status) -> Error {
        Error::from(format!("{}", err))
    }
}
