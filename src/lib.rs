use rand::distr::Distribution;
use rand::Rng;
use rand::distr::weighted::WeightedIndex;
use rand::seq::{IndexedRandom, IteratorRandom};
use rand_distr::Normal;
use core::f64;
use std::fs::File;
use std::io::prelude::*;
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
pub mod svg;
pub mod types;
use types::*;



impl NN {
    pub fn new(input_count: usize, output_count: usize, recurrence: bool, af: ActFunc) -> Self { 
        // create input and output nodes
        let mut n = vec![];
        n.push(Node::new(Genre::Input, &ActFunc::None)); // bias node 
        for _ in 0..input_count { n.push(Node::new(Genre::Input, &ActFunc::None)); }
        for _ in 0..output_count { n.push(Node::new(Genre::Output, &af)); }

        let mut s = Self { 
            nodes: n, 
            connections: vec![],
            layer_order: vec![], 
            generation: 0,
            size: (input_count+1, output_count),
            chances: [200, 40, 14, 2, 0, 0],
            recurrence,
            nodes_values: vec![], // hopefully saves memory alloc time
            nodes_function: af, 
        };
        s.topological_sort();
        s.free_nodes_calc();
        for _ in 0..2 {
            s.m_connection_add();
            s.topological_sort();
            s.free_nodes_calc();
        }
        for _ in 0..1 {
            s.m_node_add();
            s.topological_sort();
            s.free_nodes_calc();
        }
        s
    }

    pub fn get_chances(&mut self) -> &[usize; 6] {
        &self.chances
    }

    pub fn set_chances(&mut self, ch: &[usize]) {
        let mut size = ch.len();
        if size > 6 {size = 6;}
        for i in 0..size {
            self.chances[i] = ch[i];
        }
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64>{
        self.nodes_values = self.nodes.iter().map(|n| n.value).collect(); // save for recurrent
        self.nodes.iter_mut().for_each(|n| n.value = 0.); // clear 

        // read inputs, skipping first, as it's a bias node
        self.nodes.iter_mut().skip(1).filter(|n| n.genre == Genre::Input).zip(input.iter()).for_each(|(n, v)|{
            n.value = *v;
        });
        self.nodes[0].value = 1.;

        // process network: layer -> element
        // faster method would be cool
        self.layer_order.iter().for_each(|l|{
            l.iter().for_each(|i|{
                // collect active connections pointing to current node (from ordering)
                self.connections.iter().filter(|f| f.to == *i && f.active).for_each(|c|{
                    if c.recurrent {self.nodes[*i].value += self.nodes_values[c.from] * c.weight;}
                    else {self.nodes[*i].value += self.nodes[c.from].value * c.weight;}
                });

                // activate node
                let v = self.nodes[*i].value;
                match self.nodes[*i].act_func {
                    ActFunc::Sigmoid => self.nodes[*i].value = 1.0 / (1.0 + (-v).exp()),
                    ActFunc::SigmoidBipolar => self.nodes[*i].value = (2.0 / (1.0 + (-v).exp()) ) - 1., // offset
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
        self.generation += 1; // increment generation

        // choose mutation based on chances
        let dist = WeightedIndex::new(&self.chances).unwrap();
        let mut rng = rand::rng();

        // TODO ID
        match dist.sample(&mut rng) {
            0 => self.m_connection_weight(),
            1 => {self.m_connection_add();},
            2 => {self.m_node_add();},
            3 => self.m_connection_enable(),
            4 => self.m_connection_disable(),
            5 => self.m_node_func(),
            _ => unreachable!(),
        }

        self.topological_sort();
        self.free_nodes_calc();
    }

// connections manipulations
    fn m_node_add(&mut self) -> Option<(usize, usize)> {
        let mut rng = rand::rng();
        // get connection to be replaced
        let (ctbr_idx, from, to) = match self.connections.iter().enumerate().filter(|(_,c)| c.active && !c.recurrent).choose(&mut rng) {
            Some((i,_)) => (i, self.connections[i].from, self.connections[i].to),
            None => {return None;}
        };

        // TODO: inserting nodes into recurrent connections, 
        // first new connection should be normal,
        // while second should be recurrent (propably)
        //
        // insert node in the middle of existing connection
        self.nodes.push(Node::new(Genre::Hidden, &self.nodes_function));
        self.connections[ctbr_idx].active = false;
        let w = self.connections[ctbr_idx].weight;

        // to new node conn receives weight 1
        self.connections.push(Connection::new(from, self.nodes.len()-1, false)); // ID
        self.connections.last_mut().unwrap().assign_weight(1.);
        // from new node conn receives weight from deleted connection
        self.connections.push(Connection::new(self.nodes.len()-1, to, false)); // ID 
        self.connections.last_mut().unwrap().assign_weight(w);
        Some((from, to))
    }

    fn m_connection_add(&mut self) -> Option<(usize, usize)> {
        let mut rng = rand::rng();
        // randomly select node index, that have free paths and isn't output, if none return (full)
        let from = match self.nodes.iter().enumerate().filter(|(_,n)| !n.free_nodes.is_empty() ).choose(&mut rng){
            Some((i,_)) => i,
            None => {return None;}
        };
        
        // randomly select to node
        let to = self.nodes[from].free_nodes.choose(&mut rng).unwrap();
        let recurrent = self.get_node_layer(from) >= self.get_node_layer(*to);

        self.connections.push(Connection::new(from, *to, recurrent)); // ID
        Some((from, *to))
    }

    fn m_connection_disable(&mut self) {
        let mut rng = rand::rng();
        let ctbd_idx = match self.connections.iter_mut().enumerate().filter(|(_,c)| c.active).choose(&mut rng){
            Some((i,_)) => i,
            None => {return;}
        };

        self.connections[ctbd_idx].active = false;
    }

    fn m_connection_enable(&mut self) {
        let mut rng = rand::rng();
        let ctbd_idx = match self.connections.iter_mut().enumerate().filter(|(_,c)| !c.active).choose(&mut rng){
            Some((i,_)) => i,
            None => {return;}
        };

        self.connections[ctbd_idx].active = true;
    }

// set of mutation funcs
    fn m_connection_weight(&mut self){
        let mut rng = rand::rng();
        let ctbd_idx = match self.connections.iter_mut().enumerate().filter(|(_,c)| c.active).choose(&mut rng){
            Some((i,_)) => i,
            None => {return;}
        };

        // small chance for new value, otherwise slight change from normal distribution
        // Variability converges to zero as generations approach infinity
        match rng.random_range(0..10) {
            0 => {self.connections[ctbd_idx].weight = rng.random_range(-5.0..=5.0);}
            _ => {let dist = Normal::new(0., 8. / (self.generation+4) as f64 ).unwrap();
                self.connections[ctbd_idx].weight = 
                    (self.connections[ctbd_idx].weight + dist.sample(&mut rng).max(f64::MIN) ).clamp(-5.0, 5.0);}
        }
    }

    fn m_node_func(&mut self) {
        let mut rng = rand::rng();
        // randomly select node index, that have free paths and isn't output, if none return (full)
        let idx = match self.nodes.iter().enumerate().filter(|(_,n)| n.genre != Genre::Input ).choose(&mut rng){
            Some((i,_)) => i,
            None => {return;}
        };
        self.nodes[idx].act_func = match rng.random_range(0..4) {
            0 => ActFunc::Sigmoid,
            1 => ActFunc::SigmoidBipolar,
            2 => ActFunc::Tanh,
            3 => ActFunc::ReLU,
            _ => unreachable!(),
        };
    }

    pub fn free_nodes_calc(&mut self) {
        // Build a mapping from node index to its outgoing connections (excluding self)
        let mut outgoing: HashMap<usize, HashSet<usize>> = HashMap::new();
        self.connections.iter().for_each(|c|{ outgoing.entry(c.from).or_insert_with(HashSet::new).insert(c.to); } );
    
        let node_count = self.nodes.len();
        let free: Vec<Vec<usize>> = (0..node_count).map(|current_node| {    
            (0..node_count).filter(|&target_node| {
                // Exclude self and nodes with existing connections
                target_node != current_node && 
                !outgoing.get(&current_node).map_or(false, |set| set.contains(&target_node)) && 
                (self.nodes[current_node].genre != Genre::Input || self.nodes[target_node].genre != Genre::Input) &&
                (self.recurrence || (self.get_node_layer(target_node) > self.get_node_layer(current_node)) ) 
            }).collect()
        }).collect();
        free.iter().zip(self.nodes.iter_mut()).for_each(|(f, n)| {n.free_nodes = f.to_vec();} );
    }

    fn get_node_layer(&self, node_idx: usize) -> usize {
        self.layer_order
            .iter()
            .position(|layer| layer.contains(&node_idx))
            .unwrap_or(0)
    }

    // we have to divide nodes into layers based on the longest possible path to the input 
    fn topological_sort(&mut self){
        let num_nodes = self.nodes.len();
        let mut reachable = vec![false; num_nodes]; // if it can be calculated
        let mut layers = vec![0; num_nodes]; // each node idx have layer number
    
        // Initialize input nodes
        for (i, node) in self.nodes.iter().enumerate() {
            if node.genre == Genre::Input {
                reachable[i] = true;
                layers[i] = 0;
            }
        }
    
        // Build adjacency lists for predecessors and in_degree
        let mut predecessors = vec![vec![]; num_nodes];
        let mut adj = vec![vec![]; num_nodes];
        let mut in_degree = vec![0; num_nodes];
    
        // skipping recurrent connections
        self.connections.iter().filter(|c| !c.recurrent ).for_each(|c|{
            predecessors[c.to].push(c.from);
            adj[c.from].push(c.to);
            in_degree[c.to] += 1;
        });
    
        // Kahn's algorithm for topological sort
        let mut queue = VecDeque::new();
        for (i, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {queue.push_back(i);}
        }
    
        let mut topo_order = Vec::with_capacity(num_nodes);
        while let Some(node) = queue.pop_front() {
            topo_order.push(node);
            for &neighbor in &adj[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {queue.push_back(neighbor);}
            }
        }
    
        // Check for cycles
        if topo_order.len() != num_nodes {panic!("Graph contains a cycle");}
    
        // Compute layers based on reachability and predecessors
        for &node in &topo_order {
            if self.nodes[node].genre == Genre::Input {
                continue; // Input node, already initialized
            }
    
            let mut max_layer = 0;
            let mut is_reachable = false;
    
            for &pred in &predecessors[node] {
                if reachable[pred] {
                    is_reachable = true;
                    if layers[pred] + 1 > max_layer {max_layer = layers[pred] + 1;}
                }
            }
    
            if is_reachable {
                reachable[node] = true;
                layers[node] = max_layer;
            } else {
                layers[node] = 0;
                reachable[node] = false;
            }
        }
    
        // ordering into layers 
        let mut layer_amount = 0;
        layers.iter().for_each(|n| if *n > layer_amount {layer_amount = *n;});
        self.layer_order.clear();
        self.layer_order.resize(layer_amount + 3, vec![]);
        layers.iter().enumerate().for_each(|(n,l)| {
            if self.nodes[n].genre == Genre::Output {self.layer_order.last_mut().unwrap().push(n);}
            else if self.nodes[n].genre == Genre::Input {self.layer_order[0].push(n);}
            else {self.layer_order[*l+1].push(n);}
        });

        // assign recurrent markers
        for i in 0..self.connections.len() {
            self.connections[i].recurrent = self.get_node_layer(self.connections[i].from) >= self.get_node_layer(self.connections[i].to);

        }
    }
    

    // save nn to file
    pub fn save(&self, path: &str) {        
        // convert simplified nn to Vec<u8>
        let toml: String = toml::to_string(
            &self
        ).unwrap();
    
        // open file and write whole Vec<u8>
        let mut file = File::create(path).unwrap();
        file.write(toml.as_bytes()).unwrap();
    }

    // load nn from file
    pub fn load(&mut self, path: &str) {

        // convert readed Vec<u8> to plain nn
        let mut toml = String::new();
        let mut file = File::open(path).unwrap();

        file.read_to_string(&mut toml).unwrap();
        let decoded: NN = toml::from_str(&toml).unwrap();

        *self = decoded;
    }
}


#[cfg(test)]
mod tests {
    use svg::svg_nn;

    use super::*;

    #[test]
    fn it_works() {

        let mut net = NN::new(3, 2, true, ActFunc::Tanh);
        for _ in 0..32 {
            net.mutate();
        }
        let _ = net.get_chances();
        //net.set_chances(&[0,0,0]);
        svg_nn(&net, true, 1);
        net.forward(&[0.5, 0.2, 0.8]);
        println!("\nOrder: \n{:?}", net.layer_order);
        println!("\nConnections: \n{:?}", net.connections);
        println!("\nNodes: \n{:?}", net.nodes);

        assert_eq!(1, 1);
    }
}
