use std::fmt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::NodeKey;

/// Struct representing network's gene, eg. connection between two nodes.
/// It can be recurrent and/or have gating node.
#[derive(PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Key of source node
    pub from: NodeKey, 
    /// Key of destination node
    pub to: NodeKey,
    pub weight: f32,
    pub active: bool,
    /// Uses node's old value as input
    pub recurrent: bool,
    /// Connection is attenuated according to chosen node value
    pub gater: Option<NodeKey>
}

impl Connection {
    pub fn new(from: NodeKey, to: NodeKey, recurrent: bool) -> Self { 
        let mut rng = rand::rng();
        Self { 
            from,
            to,
            weight: rng.random_range(-5.0..=5.0),
            active: true,
            recurrent,
            gater: None
        } 
    }

    pub fn assign_weight(&mut self, weight: f32) {
        self.weight = weight.clamp(-9.9, 9.9);
    }
}

impl fmt::Debug for Connection {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut l = "".to_string();
        l += &( "( ".to_string() + &format!("{:?}", self.from) + " - " + &format!("{:?}", self.to) + " )");
        l += &( "*".to_string() + &format!("{:>+7.3}", self.weight) );
        l += &( "\t :".to_string() + &self.active.to_string() );
        match &self.gater {
            Some(g) => l += &( "\t|".to_string() + &format!("{:?}", g) ),
            None => l += "\t|------",
        }
        if self.recurrent {l += " R"}
        write!(fmt, "{}", l)
    }
}
