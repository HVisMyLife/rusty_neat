use std::fmt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::NodeKey;


#[derive(PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub from: NodeKey, // key of start node
    pub to: NodeKey, // key of end node
    pub weight: f32,
    pub active: bool, // connections can be deactivated through mutations
    pub recurrent: bool, // aren't included in layer sort
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
