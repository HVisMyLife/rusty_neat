#![allow(dead_code)]  // remove for production
#![feature(extract_if)]

mod svg;
mod connection;
mod node;
mod nn;
mod neat;

pub use neat::{NeatIntermittent, NeatContinous};
pub use nn::NN;
pub use connection::Connection;
pub use node::{Node, ActFunc, NodeKey, Genre};
pub use svg::svg_nn;


// https://github.com/JingIsCoding/neat_lib/blob/main/src/neat/genome.rs
// inspiration

// TODO's:
// - gates visualisation
// - penalizing stagnation
// - penalizing huge size growth with little fitness growth



#[cfg(test)]
mod tests {
    //use super::*;

    #[test]
    fn it_works() {
        assert_eq!(1, 1);
    }
}
