//! Error module.

use tf::{Code, Status};

error_chain! {
    errors {
        Stub
        UndefinedFuncAttr
        UndefinedGrad
        UndefinedTensorShape
        OpNotFound
        TensorNotFound
        UnimplTraitMethod
        NoneError
    }

    foreign_links {
        FFINulError(::std::ffi::NulError);
        Utf8Error(::std::str::Utf8Error);
        BorrowError(::std::cell::BorrowMutError);
    }
}

impl From<Status> for Error {
    fn from(err: Status) -> Error {
        Error::from(format!("{}", err))
    }
}

impl From<Error> for Status {
    fn from(err: Error) -> Status {
        Status::new_set(Code::Unknown, err.description()).unwrap()
    }
}
