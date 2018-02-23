#![allow(dead_code)]

use std::collections::HashMap;

pub(crate) enum AttrValue {
    S(String),
    B(bool),
}

impl AttrValue {
    pub fn unwrap_b(self) -> bool {
        match self {
            AttrValue::B(b) => b,
            _ => unreachable!(),
        }
    }

    pub fn unwrap_s(self) -> String {
        match self {
            AttrValue::S(s) => s,
            _ => unreachable!(),
        }
    }
}

pub(crate) struct NameAttrList {
    pub name: String,
    pub attr: HashMap<String, AttrValue>,
}

impl NameAttrList {
    pub fn new(name: String) -> NameAttrList {
        NameAttrList {
            name,
            attr: HashMap::new(),
        }
    }

    pub fn serialize(self) -> Vec<u8> {
        unimplemented!()
    }
}
