use serde::{Serialize, Deserialize};
//use itertools::Itertools;
use std::{collections::HashSet, fmt};

#[derive(Eq, PartialOrd, Ord, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActFunc {
    Sigmoid,
    SigmoidBipolar,
    Tanh,
    ReLU,
    None
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
        write!(f, "{:>3}:{}", self.sconn, self.dup)
    }
}
impl fmt::Display for NodeKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:>3}:{}", self.sconn, self.dup)
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
    fn default() -> Self {Self::new(Genre::Hidden, &ActFunc::Tanh)}
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
        l += match self.act_func {
            ActFunc::Sigmoid => "*S_",
            ActFunc::SigmoidBipolar => "*Sb",
            ActFunc::Tanh => "*T_",
            ActFunc::ReLU => "*R_",
            ActFunc::None => "*1_",
        };
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
