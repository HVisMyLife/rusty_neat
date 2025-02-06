use serde::{Serialize, Deserialize};
use rand::Rng;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActFunc {
    Sigmoid,
    SigmoidBipolar,
    Tanh,
    ReLU,
    None
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Genre {
    Hidden,
    Input,
    Output,
}


#[derive(Clone, Serialize, Deserialize)]
pub struct Node {
    pub value: f64,
    pub genre: Genre, // 0 - hidden, 1 - input, 2 - output
    pub act_func: ActFunc,
    pub free_nodes: Vec<usize>, // vec containing nodes, to which there is free path
}
impl Node {
    // initializing to random values 
    pub fn new(genre: Genre, af: &ActFunc) -> Self { 
        // input nodes don't have activation functions
        let af_c = if genre == Genre::Input {ActFunc::None}
        else {af.clone()};

        Self { 
            value: 0.0, 
            genre,
            act_func: af_c,
            free_nodes: vec![],
        } 
    }
}
impl Default for Node {
    fn default() -> Self {Self::new(Genre::Hidden, &ActFunc::Tanh)}
}
impl fmt::Debug for Node {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut l = "Node".to_string();
        l += match self.genre {
            Genre::Input => " I",
            Genre::Hidden => " H",
            Genre::Output => " O",
        };
        l += &("( ".to_string() + &format!("{:.3}", self.value) + " )" );
        l += match self.act_func {
            ActFunc::Sigmoid => "* S",
            ActFunc::SigmoidBipolar => "* Sb",
            ActFunc::Tanh => "* T",
            ActFunc::ReLU => "* R",
            ActFunc::None => "* 1",
        };
        l += "\t";
        l += &(format!("F: {:?}", self.free_nodes));
        write!(fmt, "{}", l)
    }
}


#[derive(Clone, Serialize, Deserialize)]
pub struct Connection {
    pub from: usize, // idx of start node
    pub to: usize, // idx of end node
    pub weight: f64,
    pub active: bool, // connections can be deactivated through mutations
    pub recurrent: bool,
    pub innov_id: usize, 
}
impl Connection {
    pub fn new(from: usize, to: usize, recurrent: bool) -> Self { 
        let mut rng = rand::rng();
        Self { 
            from,
            to,
            weight: rng.random_range(-1.0..=1.0),
            active: true,
            recurrent,
            innov_id: 0, 
        } 
    }

    pub fn assign_weight(&mut self, weight: f64) {
        self.weight = weight.clamp(-5., 5.);
    }
    pub fn assign_id(&mut self, innov_id: usize) {
        self.innov_id = innov_id;
    }
}
impl fmt::Debug for Connection {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut l = "Cn".to_string();
        l += &( ":".to_string() + &self.innov_id.to_string() + "\t" );
        l += &( "( ".to_string() + &self.from.to_string() + " - " + &self.to.to_string() + " )");
        l += &( " * ".to_string() + &format!("{:.3}", self.weight) );
        l += &( "\t :".to_string() + &self.active.to_string() );
        if self.recurrent {l += " R"}
        write!(fmt, "{}", l)
    }
}


#[derive(Clone, Serialize, Deserialize)]
pub struct NN {    
    pub nodes: Vec<Node>,
    pub connections: Vec<Connection>,
    pub layer_order: Vec<Vec<usize>>, // layers for calculating values (without input nodes), eg
                                        // which node calculate first
    pub generation: usize, // generation number, just out of curiosity
    pub size: (usize, usize), // information

    pub chances: [usize; 6], // chances for mutations to happen, sum does NOT need to be equal 100
    pub recurrence: bool,
    pub nodes_values: Vec<f64>,
    pub nodes_function: ActFunc,
}
impl fmt::Debug for NN {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut l = "Nodes: \n".to_string();
        self.nodes.iter().enumerate().for_each(|(i,n)|{
            l += &(format!("{}.\t{:?}", i, n) + "\n");
        });
        l += "Connections: \n";
        self.connections.iter().for_each(|c|{
            l += &(format!("{:?}", c) + "\n");
        });
        l += &(format!("Order: {:?}", self.layer_order) + "\n");
        l += &("Gen: ".to_string() + &self.generation.to_string() + "\n");
        l += &(format!("Chances: {:?}", self.chances) + "\n");
        l += &("Recurrence: ".to_string() + &self.recurrence.to_string() + "\n");
        write!(fmt, "{}", l)
    }
}
