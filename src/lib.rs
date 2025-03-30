#![feature(extract_if)]

mod connection;
mod node;
mod nn;
mod neat;
#[cfg(feature = "image")]
mod svg;

pub use neat::{NeatIntermittent, NeatContinous};
pub use nn::NN;
pub use connection::Connection;
pub use node::{Node, ActFunc, NodeKey, Genre};
#[cfg(feature = "image")]
pub use svg::svg_nn;


// https://github.com/JingIsCoding/neat_lib/blob/main/src/neat/genome.rs
// inspiration

// TODO's:
// - gates visualisation
// - penalizing stagnation
// - penalizing huge size growth with little fitness growth
// - tests



#[cfg(test)]
mod tests {
    //use super::*;

    #[test]
    fn it_works() {
        assert_eq!(1, 1);
    }
}
