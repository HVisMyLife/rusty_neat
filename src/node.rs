use serde::{Serialize, Deserialize};
//use itertools::Itertools;
use std::{collections::HashSet, fmt};
use rand::seq::IndexedRandom;

#[derive(Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActFunc {
    Sigmoid,
    SigmoidBipolar,
    HyperbolicTangent,
    HardHyperbolicTangent,
    Softsign,
    GaussianBump,
    Sinusoid,
    ReLU,
    LeakyReLU,
    SELU,
    Identity,
    BentIdentity,
    Inverse,
    BinaryStep,
    Bipolar,
    None
}

impl ActFunc {
    pub fn random(list: &[Self]) -> Self {
        let mut rng = rand::rng();
        list.choose(&mut rng).unwrap().clone()
    }
    pub fn run(&self, x: f32, value: f32) -> f32 {
        match self {
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::SigmoidBipolar => (2.0 / (1.0 + (-x).exp())) - 1.0, 
            Self::HyperbolicTangent => x.tanh(),
            Self::HardHyperbolicTangent => {x.clamp(-1.0, 1.0)},
            Self::Softsign => x / (1.0 + x.abs()),
            Self::GaussianBump => (-x * x / 2.0).exp(),
            Self::Sinusoid => x.sin(),
            Self::ReLU => x.max(0.0),                  // Rectified Linear Unit
            Self::LeakyReLU => {
                const ALPHA: f32 = 0.01;
                if x > 0.0 {x} 
                else {ALPHA * x}
            },
            Self::SELU => {
                const ALPHA: f32 = 1.67326324235;
                const SCALE: f32 = 1.05070098735;
                if x > 0.0 {SCALE * x} 
                else {SCALE * ALPHA * (x.exp() - 1.0)}
            },
            Self::Identity => x,
            Self::BentIdentity => if x > 0.0 { x } else { x / 5.0 },
            Self::Inverse => -x,
            Self::BinaryStep => if x < 0.0 { 0.0 } else { 1.0 },
            Self::Bipolar => if x < 0.0 { -1.0 } else { 1.0 }, // Sign function
            Self::None => value + x, // for inputs
        }
    }
}
impl fmt::Debug for ActFunc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut l = "*".to_string();
        l += match self {
            Self::Sigmoid => "Sigmoid",
            Self::SigmoidBipolar => "SigBi  ",
            Self::HyperbolicTangent => "Tanh   ",
            Self::HardHyperbolicTangent => "HTanh  ",
            Self::Softsign => "Softsn ",
            Self::GaussianBump => "Gauss  ",
            Self::Sinusoid => "Sin    ",
            Self::ReLU => "ReLU   ",
            Self::LeakyReLU => "LReLU  ",
            Self::SELU => "SELU   ",
            Self::Identity => "Ident  ",
            Self::BentIdentity => "BIdent ",
            Self::Inverse => "Inverse",
            Self::BinaryStep => "Binary ",
            Self::Bipolar => "Bipolar",
            Self::None => "None   ",
        };
        write!(f, "{}", l)
    }
}


#[derive(Eq, PartialOrd, Ord, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Genre {
    Hidden,
    Input,
    Output,
}

#[derive(PartialOrd, Ord, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NodeKey {
    pub sconn: usize,
    pub dup: usize,
}
impl fmt::Debug for NodeKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:>4}:{}", self.sconn, self.dup)
    }
}
impl fmt::Display for NodeKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{} ", self.sconn, self.dup)
    }
}
impl NodeKey {
    pub fn new(sconn: usize, dup: usize) -> Self {
        Self { sconn, dup }
    }
}


#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub struct Node {
    pub value: f32,
    pub value_gate: f32,
    pub value_old: f32,
    pub genre: Genre, // 0 - hidden, 1 - input, 2 - output
    pub act_func: ActFunc,
    pub free_nodes_f: HashSet<NodeKey>, // vec containing nodes, to which there is free path
    pub free_nodes_r: HashSet<NodeKey>, // vec containing nodes, to which there is free path
}
impl Node {
    // initializing to random values 
    pub fn new(genre: Genre, af: &ActFunc) -> Self { 
        // input nodes don't have activation functions
        let af_c = if genre == Genre::Input {ActFunc::None}
        else {af.clone()};

        Self { 
            value: 0.0, 
            value_gate: 0.0,
            value_old: 0.0, 
            genre,
            act_func: af_c,
            free_nodes_f: HashSet::new(),
            free_nodes_r: HashSet::new(),
        } 
    }

}
impl Default for Node {
    fn default() -> Self {Self::new(Genre::Hidden, &ActFunc::HyperbolicTangent)}
}
impl fmt::Debug for Node {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut l = "".to_string();
        l += match self.genre {
            Genre::Input => " I",
            Genre::Hidden => " H",
            Genre::Output => " O",
        };
        l += &("( ".to_string() + &format!("{:>+.3}", self.value) + " )" );
        l += &format!("{:?}", self.act_func);
        l += &(" | G( ".to_string() + &format!("{:>+.3}", self.value_gate) + " )" );
        //l += " => F[ ";
        //self.free_nodes_f.iter().sorted_by_key(|k| k.sconn ).for_each(|k| {
        //    l += &(format!("{}:{}, ", k.sconn, k.dup));
        //} );
        //l += "]";
        //l += " => R[ ";
        //self.free_nodes_r.iter().sorted_by_key(|k| k.sconn ).for_each(|k| {
        //    l += &(format!("{}:{}, ", k.sconn, k.dup));
        //} );
        //l += "]";
        write!(fmt, "{}", l)
    }
}
