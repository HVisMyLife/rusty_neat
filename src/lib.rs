#![allow(clippy::map_clone, clippy::needless_range_loop)]
//#![feature(drain_filter)]

use bincode::{serialize, deserialize};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {

        let mut net = NN::new(3, 2);
        for _ in 0..32 {
            net.mutate();
        }
        let _ = net.get_chances();
        net.set_chances(&[0,0,0]);
        net.forward(&[0.5, 0.2, 0.8]);
        println!("\nOrder: \n{:?}", net.layer_order);
        println!("\nConnections: \n{:?}", net.connections);
        println!("\nNodes: \n{:?}", net.nodes);

        assert_eq!(1, 1);
    }
}


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActFunc {
    Sigmoid,
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


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    value: f64,
    pub bias: f64,
    genre: Genre, // 0 - hidden, 1 - input, 2 - output
    pub act_func: ActFunc,
    free_nodes: Vec<usize>, // vec containing nodes, to which there is free path
}
impl Node {
    // initializing to random values 
    pub fn new(genre: Genre) -> Self { 
        let mut af = match fastrand::usize(0..3) {
            0 => ActFunc::Sigmoid,
            1 => ActFunc::Tanh,
            2 => ActFunc::ReLU,
            _ => unreachable!(),
        };
        // input nodes don't have activation functions
        if genre == Genre::Input {af = ActFunc::None;}

        // input nodes don't have biases
        let b = match genre {
            Genre::Input => 0.0,
            _ => fastrand::f64() * 2.0 - 1.0,
        };

        Self { 
            value: 0.0, 
            bias: b,
            genre,
            act_func: af,
            free_nodes: vec![],
        } 
    }
}
impl Default for Node {
    fn default() -> Self {Self::new(Genre::Hidden)}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub from: usize, // idx of start node
    pub to: usize, // idx of end node
    pub weight: f64,
    pub active: bool // connections can be deactivated through mutations
}

impl Connection {
    pub fn new(from: usize, to: usize) -> Self { 
        Self { 
            from,
            to,
            weight: fastrand::f64() * 2.0 - 1.0,
            active: true,
        } 
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NN {    
    pub nodes: Vec<Node>,
    pub connections: Vec<Connection>,
    pub layer_order: Vec<Vec<usize>>, // layers for calculating values (without input nodes), eg
                                        // which node calculate first
    pub generation: usize, // generation number, just out of curiosity
    pub size: (usize, usize),

    chances: [usize; 7], // chances for mutations to happen, sum does NOT need to be equal 100
}

impl NN {
    pub fn new(input_count: usize, output_count: usize) -> Self { 
        // create input and output nodes
        let mut n = vec![];
        for _ in 0..input_count { n.push(Node::new(Genre::Input)); }
        for _ in 0..output_count { n.push(Node::new(Genre::Output)); }

        Self { 
            nodes: n, 
            connections: vec![],
            layer_order: vec![], 
            generation: 0,
            size: (input_count, output_count),
            chances: [35, 35, 10, 10, 10, 0, 0],
        } 
    }

    pub fn get_chances(&mut self) -> &[usize; 7] {
        &self.chances
    }

    pub fn set_chances(&mut self, ch: &[usize]) {
        let mut size = ch.len();
        if size > 7 {size = 7;}
        for i in 0..size {
            self.chances[i] = ch[i];
        }
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64>{
        // read inputs
        self.nodes.iter_mut().filter(|n| n.genre == Genre::Input).zip(input.iter()).for_each(|(n, v)|{
            n.value = *v;
        });

        // process network: layer -> element
        // faster method would be cool
        self.layer_order.iter().for_each(|l|{
            l.iter().for_each(|i|{
                // collect active connections pointing to current node (from ordering)
                self.connections.iter().filter(|f| f.to == *i && f.active).for_each(|c|{
                    self.nodes[*i].value += self.nodes[c.from].value * c.weight;
                });

                // activate node
                let v = self.nodes[*i].value;
                match self.nodes[*i].act_func {
                    ActFunc::Sigmoid => self.nodes[*i].value = 1.0 / (1.0 + (-v).exp()),
                    ActFunc::Tanh => self.nodes[*i].value = v.tanh(),
                    ActFunc::ReLU => self.nodes[*i].value = v.max(0.0),
                    ActFunc::None => {},
                };
            });
        });

        // return outputs
        self.nodes.iter()
            .filter(|n| n.genre == Genre::Output)
            .map(|n| n.value)
            .collect::<Vec<_>>()
    }

    pub fn mutate(&mut self) {
        // init network
        if self.generation == 0 {
            self.topological_sort();
            self.free_nodes_calc();
            self.m_connection_add();
            self.topological_sort();
            self.free_nodes_calc();
        }
        self.generation += 1; // increment generation

        // choose mutation based on chances
        let random_num = fastrand::usize(0..self.chances.iter().sum());
        let mut cumulative_prob = 0;
        let mut func_num = 99;
        for (i, prob) in self.chances.iter().enumerate() {
            cumulative_prob += prob;
            if random_num <= cumulative_prob {
                func_num = i;
                break;
            }
        }

        match func_num {
            0 => self.m_weight_mangle(),
            1 => self.m_bias_mangle(),
            2 => self.m_act_mangle(),
            3 => self.m_connection_add(),
            4 => self.m_node_add(),
            5 => self.m_connection_enable(),
            6 => self.m_connection_disable(),
            99 => {},
            _ => unreachable!(),
        }

        self.topological_sort();
        self.free_nodes_calc();
    }

// connections manipulations
    fn m_node_add(&mut self) {
        // get active connections, if none return
        let mut con_filtered = self.connections.iter_mut().filter(|c| c.active).collect::<Vec<_>>();
        if con_filtered.is_empty() {return;}

        let idx = fastrand::usize(0..con_filtered.len());
        let from = con_filtered[idx].from;
        let to = con_filtered[idx].to;

        // insert node in the middle of existing connection
        self.nodes.push(Node::default());
        con_filtered[idx].active = false;
        self.connections.push(Connection::new(from, self.nodes.len()-1));
        self.connections.push(Connection::new(self.nodes.len()-1, to));
    }

    fn m_connection_add(&mut self) {
        // randomly select node, that isn't output and have free paths, if none return (full)
        let mut from_v: Vec<usize> = vec![];
        self.layer_order.iter().for_each(|l|{
            l.iter().for_each(|p|{
                if !self.nodes[*p].free_nodes.is_empty() {
                    from_v.push(*p);
                }
            });
        });
        
        
        if from_v.is_empty(){println!("Err, no free nodes");return;}
        let from = from_v[fastrand::usize(0..from_v.len())];

        // randomly select to node
        let ll = self.nodes[from].free_nodes.len();
        let to = self.nodes[from].free_nodes[fastrand::usize(0..ll)];

        self.connections.push(Connection::new(from, to));
    }

    fn m_connection_disable(&mut self) {
        let mut con_filtered = self.connections.iter_mut().filter(|c| c.active && c.weight != 0.0).collect::<Vec<_>>();
        if con_filtered.is_empty() {return;}
        let idx = fastrand::usize(0..con_filtered.len());

        con_filtered[idx].weight = 0.0;
    }

    fn m_connection_enable(&mut self) {
        let mut con_filtered = self.connections.iter_mut().filter(|c| c.active && c.weight == 0.0).collect::<Vec<_>>();
        if con_filtered.is_empty() {return;}
        let idx = fastrand::usize(0..con_filtered.len());

        con_filtered[idx].weight = fastrand::f64() * 2.0 - 1.0;
    }

// set of mutation funcs
    fn m_weight_mangle(&mut self){
        let mut con_filtered = self.connections.iter_mut().filter(|c| c.active && c.weight != 0.0).collect::<Vec<_>>();
        if con_filtered.is_empty() {return;}
        let idx = fastrand::usize(0..con_filtered.len());

        con_filtered[idx].weight += fastrand::f64() / 2.5 - 0.2;
        con_filtered[idx].weight = con_filtered[idx].weight.clamp(-1.0, 1.0);
    }

    fn m_bias_mangle(&mut self) {
        let mut node_filtered = self.nodes.iter_mut().filter(|n| n.genre != Genre::Input).collect::<Vec<_>>();
        let idx = fastrand::usize(0..node_filtered.len());

        node_filtered[idx].bias += fastrand::f64() / 2.5 - 0.2;
        node_filtered[idx].bias = node_filtered[idx].bias.clamp(-1.0, 1.0)
    }

    fn m_act_mangle(&mut self) {
        let idx = fastrand::usize(self.size.0..self.nodes.len());
        self.nodes[idx].act_func = match fastrand::usize(0..3) {
            0 => ActFunc::Sigmoid,
            1 => ActFunc::Tanh,
            2 => ActFunc::ReLU,
            _ => unreachable!(),
        };
    }

// order funcs
    fn free_nodes_calc(&mut self) {
        // check, to which nodes is a free path from node
        self.layer_order.iter().enumerate().for_each(|(i, l)|{
            l.iter().for_each(|p|{

                let mut free: Vec<usize> = vec![];
                self.layer_order.iter().skip(i).for_each(|li|{
                    li.iter().for_each(|pi|{
                        if *p != *pi && self.nodes[*pi].genre != Genre::Input {free.push(*pi);}
                    });
                });

                self.connections.iter().filter(|c| c.from == *p).for_each(|c|{
                    //free.drain_filter(|e| c.to == *e);
                    free.retain(|e| c.to != *e);
                    // drain_filter is propably removed from rust, retain is simply inversion
                });

                if self.nodes[*p].genre != Genre::Output { 
                    self.nodes[*p].free_nodes.clear(); 
                    self.nodes[*p].free_nodes.append(&mut free); 
                }
                //else { self.nodes[*p].free_nodes = Vec::<usize>::new(); }
            });
        });
    }

    // belive or not, but that part was mostly written by chatgpt, 
    // I have very little idea how it works, but it works, so...
    fn topological_sort(&mut self) {
        let mut indegrees = vec![0; self.nodes.len()];
        let mut neighbors = vec![Vec::new(); self.nodes.len()];
        
        for connection in self.connections.as_slice() {
            indegrees[connection.to] += 1;
            neighbors[connection.from].push(connection.to);
        }
        
        let mut queue = Vec::new();
        let mut layers = Vec::new();
        
        for i in 0..self.nodes.len() {
            if indegrees[i] == 0 {
                queue.push(i);
            }
        }
        
        while !queue.is_empty() {
            let mut layer = Vec::new();
            let mut next_queue = Vec::new();
            
            for node in queue {
                layer.push(node);
                for neighbor in &neighbors[node] {
                    indegrees[*neighbor] -= 1;
                    if indegrees[*neighbor] == 0 {
                        next_queue.push(*neighbor);
                    }
                }
            }
            
            layers.push(layer);
            queue = next_queue;
        }
        
        self.layer_order = layers;
    }

    // save nn to file
    pub fn save(&self, path: &str) {        
        // convert simplified nn to Vec<u8>
        let encoded: Vec<u8> = serialize(
            &self
        ).unwrap();
    
        // open file and write whole Vec<u8>
        let mut file = File::create(path).unwrap();
        file.write_all(&encoded).unwrap();
    }

    // load nn from file
    pub fn load(&mut self, path: &str) {

        // convert readed Vec<u8> to plain nn
        let mut buffer = vec![];
        let mut file = File::open(path).unwrap();
        file.read_to_end(&mut buffer).unwrap();
        let decoded: NN = deserialize(&buffer).unwrap();

        *self = decoded;
    }
}
