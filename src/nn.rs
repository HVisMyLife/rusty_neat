use itertools::Itertools;
use rand::{seq::IteratorRandom, Rng};
use rand_distr::{weighted::WeightedIndex, Distribution, Normal};
use serde::{Serialize, Deserialize};
use serde_with::serde_as;
use core::f32;
use std::{collections::{HashMap, HashSet}, fmt, fs::File, io::{Read, Write}};

use crate::{connection::Connection, node::{ActFunc, Genre, Node, NodeKey}};

/// NEAT network representation.
#[serde_as]
#[derive(Clone, Serialize, Deserialize)]
pub struct NN {
    #[serde_as(as = "Vec<(_, _)>")]
    pub nodes: HashMap<NodeKey, Node>, // key is a splited connection key + doubles protection
    #[serde_as(as = "Vec<(_, _)>")]
    pub connections: HashMap<usize, Connection>, // key is an innovation number
    /// All nodes are layered for calculations.
    /// Additionally it's usefull during visualisation generation.
    pub layer_order: Vec<HashSet<NodeKey>>, // layers for calculating values
    /// orphaned nodes (without inputs)
    pub idle: HashSet<NodeKey>,
    /// Incremented after mutate.
    pub generation: usize, 
    /// Amount of input and output nodes.
    pub size: (usize, usize),
    /// leftover reserved space for inputs and outputs
    pub size_free: (usize, usize),

    outputs: Vec<f32>,

    chances: [usize; 8], // chances for mutations to happen, sum does NOT need to be equal 100
    pruning: (bool, f32),
    /// Are recurrent connections allowed, and
    /// how much new data influences node's memory.
    /// Useful if you want more stable memory value.
    /// value_old = value_old * (1. - recurrence.1) + value * recurrence.1 )
    pub recurrence: (bool, f32),
    /// default activaton function for nodes.
    pub function_io: ActFunc,
    /// Vector of functions that can appear during mutations
    pub functions_allowed: Vec<ActFunc>,
    /// Higher fitness gives higher chanceto become parent during crossover.
    pub fitness: f32,
    /// Species ID.
    pub species: usize,
    pub active: bool,
}

impl NN {
    /// add_space - how much space reserve for future add_input/output 
    pub fn new(input_count: usize, output_count: usize, add_space: Option<(usize,usize)>, 
        recurrence: bool, new_data_weight_rec: f32, function_io: ActFunc, functions_allowed: &[ActFunc]
    ) -> Self {
        let add_space = match add_space {Some(a) => a, None => (0,0) };
        // create input and output nodes
        let mut n = HashMap::new();
        n.insert(
            NodeKey::new(0, 0), 
            Node::new(Genre::Input, &ActFunc::None)); // bias node 
        for i in 1..input_count+1 { 
            n.insert(
                NodeKey::new(i, 0), 
                Node::new(Genre::Input, &ActFunc::None)); 
        }
        for i in input_count+1+add_space.0..input_count+output_count+1+add_space.0 { 
            n.insert(
                NodeKey::new(i, 0), 
                Node::new(Genre::Output, &function_io)); 
        }
        let mut s = Self { 
            nodes: n, 
            connections: HashMap::new(),
            layer_order: vec![], 
            idle: HashSet::new(),
            generation: 0,
            size: (input_count + 1, output_count),
            size_free: add_space,
            outputs: vec![0.; output_count],
            chances: [200, 20, 5, 10, 3, 0, 0, 0], // weight, ca, na, ga, gr, cn, cf, am
            pruning: (false, 0.33),
            recurrence: (recurrence, new_data_weight_rec),
            function_io,
            functions_allowed: functions_allowed.to_vec(),
            fitness: 0.,
            species: 0,
            active: true,
        };
        s.sort_layers();
        s.free_nodes_calc();
        s
    }

    /// Adds input node, returns false if there is no space left.
    pub fn add_input(&mut self) -> bool {
        if self.size_free.0 < 1 {return false}
        let k = NodeKey::new(self.size.0, 0);
        self.nodes.insert(k, Node::new(Genre::Input, &ActFunc::None));
        self.size_free.0 -= 1;
        self.size.0 += 1;
        true
    }

    /// Adds output node, returns false if there is no space left.
    pub fn add_output(&mut self, func: &ActFunc) -> bool {
        if self.size_free.1 < 1 {return false}
        let k = NodeKey::new(self.size.0 + self.size_free.0 + self.size.1, 0);
        self.nodes.insert(k, Node::new(Genre::Output, func));
        self.size_free.1 -= 1;
        self.size.1 += 1;
        self.outputs.push(0.);
        true
    }
// #########################################################################################################################################
    
    /// Cleans up network by merging doubled connections.
    pub fn post_process(&mut self) {
        self.connections.retain(|_, c| c.active && c.weight.abs() > 0.0001 ); // remove inactive 
        let keys: Vec<usize> = self.connections.keys().cloned().collect(); 
        let mut waste: Vec<usize> = vec![];
        for k in &keys {
            for g in &keys {
                if k != g && !waste.contains(k) {
                    let g_c = self.connections.get(g).unwrap().clone();
                    let k_c = self.connections.get_mut(k).unwrap();
                    if k_c.from == g_c.from && k_c.to == g_c.to && k_c.gater == g_c.gater && k_c.recurrent == g_c.recurrent {
                        k_c.weight += g_c.weight;
                        waste.push(*g);
                    }
                }
            }
        }
        self.connections.retain(|k,_| !waste.contains(k) ); // merge identical 
        // TODO removing dead-ends
    }

    /// Compares self with other network, higher return means other is more different.
    /// - c1 - excess genes weight,
    /// - c2 - disjoint genes weight,
    /// - c3 - weight difference between similar genes weight,
    /// - c4 - different activation functions in similar nodes weight,
    pub fn compare(&self, nn: &NN, c1: f32, c2: f32, c3: f32, c4: f32) -> f32 {
        let mut g_e = 0;    // excess
        let mut g_d = 0;    // disjoint 
        let mut g_wd = 0.;  // sum of weight differences 
        let mut g_m = 0;    // amount of matching genes
        let max_a = self.connections.keys().max().unwrap();
        let max_b = nn.connections.keys().max().unwrap();
        let max = max_a.max(max_b);
        for i in 0 ..= *max {
            let a = match self.connections.get(&i)  { Some(c) => c.weight, None => f32::MAX };
            let b = match nn.connections.get(&i)    { Some(c) => c.weight, None => f32::MAX };
            
            if a != f32::MAX && b != f32::MAX { g_m += 1; g_wd += (a - b).abs(); }
            else if *max_a.min(max_b) < i && (a != f32::MAX || b != f32::MAX) { g_e += 1;}
            else if a != f32::MAX || b != f32::MAX { g_d += 1; }
        }
        let amount = self.connections.len().max(nn.connections.len());
        let excess =    ( c1 * (g_e as f32) ) / ( amount as f32 );
        let disjoint =  ( c2 * (g_d as f32) ) / ( amount as f32 );
        let weight = if g_m != 0  {c3 * ((g_wd as f32) / (g_m as f32) )} else {0.};
        //println!("e:{} , d:{} , w:{}", excess, disjoint, weight);

        let mut disjoint_nodes = 0.;
        self.nodes.iter().for_each(|(s_k,s_n)|{
            let opt = nn.nodes.get(s_k);
            if let Some(o_n) = opt {
                if s_n.genre != o_n.genre {disjoint_nodes += 1.;}
            }
        });
        let nodes_max = self.nodes.len().max(nn.nodes.len());
        disjoint_nodes /= nodes_max as f32;
        disjoint_nodes *= c4;
        
        excess + disjoint + weight + disjoint_nodes // TODO: including gates
    }

    // fitness average / species size 
    // fitness adjusted / global fitness adjusted * N
    /// Clones network with higher fitness and averages similar genes.
    pub fn crossover(&self, nn: &NN) -> NN {
        let mut child = match self.fitness > nn.fitness {
            true => self.clone(),
            false => nn.clone()
        };
        let max_a = self.connections.keys().max().unwrap();
        let max_b = nn.connections.keys().max().unwrap();
        for i in 0 ..= *max_a.max(max_b) {
            let a = self.connections.get(&i);
            let b = nn.connections.get(&i);

            if a != None && b != None {
                child.connections.get_mut(&i).unwrap().weight = 
                    (a.unwrap().weight + b.unwrap().weight) / 2.; 
                // gates have to be the same as
                // in node's source, so from fittest parent
            }
        }
        child
    }

// #########################################################################################################################################
    
    /// Calculates every node output according to layer ordering.
    pub fn process_network(&mut self, inputs: &[f32]) -> &Vec<f32> {
        let mut key = NodeKey::new(0, 0);
        self.nodes.get_mut(&key).unwrap().value = 1.;
        for i in 0..self.size.0-1 {
            key.sconn = i + 1;
            self.nodes.get_mut(&key).unwrap().value = inputs[i];
        }

        // smooth memory
        self.nodes.iter_mut().for_each(
            |(_, n)| n.value_old = n.value_old * (1. - self.recurrence.1) + n.value * self.recurrence.1 );

        let layer_order = self.layer_order.clone();
        for layer in &layer_order {
            for key in layer {
                self.process_node(&key);
            }
        }
    
        let mut key = NodeKey::new(0, 0);
        // return outputs
        self.outputs.iter_mut().enumerate().for_each(|(i, v)| {
            key.sconn = self.size.0 + self.size_free.0 + i;
            *v = self.nodes.get(&key).unwrap().value;
        });
        &self.outputs
    }
    
    /// Calculate single node's output, along with gating value.
    pub fn process_node(&mut self, node_key: &NodeKey) {
        let mut sum = 0.0;

        // Iterate through incoming connections (both feed-forward and recurrent)
        for (_connection_id, connection) in &self.connections {
            if connection.to == *node_key {
                let value = 
                    if connection.recurrent { self.nodes.get(&connection.from).unwrap().value_old }
                    else { self.nodes.get(&connection.from).unwrap().value};

                let mut gated_weight = connection.weight; // Start with the base weight
                // Apply gating, if present
                if let Some(gater_key) = &connection.gater {
                    gated_weight *= self.nodes.get(gater_key).unwrap().value_gate; // Get gater value
                }
                sum += value * gated_weight;
            }
        }

        let node = self.nodes.get_mut(node_key).unwrap();

        // Apply activation function (e.g., sigmoid, tanh, ReLU)
        node.value = node.act_func.run(sum, node.value);
        node.value_gate = (0.2 * sum + 0.5).clamp(0., 1.); // faster than sigmoid 
    }


    /// Mutates network according to chances (pruning is disabled).
    /// Returns option1 if made a new connection (from node, to node).
    /// Returns option2 if made a new node (innovation number of split connection).
    /// NEAT handler needs to assign innov_id's after analyzing whole generation.
    pub fn mutate(&mut self) -> (Option<Connection>, Option<( Connection, Connection )>) {
        self.generation += 1; // increment generation

        // choose mutation based on chances
        let mut rng = rand::rng();

        let mut n_conn = None;
        let mut n_node = None;

        if ! self.pruning.0 {
            let dist = WeightedIndex::new(&self.chances).unwrap();
            match dist.sample(&mut rng) {
                0 => self.m_connection_weight(),
                1 => {n_conn = self.m_connection_add();},
                2 => {n_node = self.m_node_add();},
                3 => self.m_connection_gater_add(),
                4 => self.m_connection_gater_remove(),
                5 => self.m_connection_enable(),
                6 => self.m_connection_disable(),
                7 => self.m_node_func(),
                _ => unreachable!(),
            }
        }
        else {
            n_conn = self.prune(self.pruning.1 as f64);
        }

        self.sort_layers();
        self.free_nodes_calc();
        (n_conn, n_node)
    }

    /// Deletes nodes and connections
    fn prune(&mut self, p: f64) -> Option<Connection>  {
        let mut rng = rand::rng();
        let mut out: Option<Connection> = None;

        let mut counts: HashMap<NodeKey, (usize, usize, bool)> = self.nodes.iter()
            .map(|(k, n)| 
                (
                    k.clone(), 
                    ( if n.genre == Genre::Input {1} else {0} , if n.genre == Genre::Output {1} else {0}, n.genre == Genre::Hidden)
                ) 
            ).collect();
        self.connections.values().filter(|c| c.active && c.to != c.from ).for_each(|v| {
            counts.get_mut(&v.to).unwrap().0 += 1;
            counts.get_mut(&v.from).unwrap().1 += 1;
        } );

        let d_nodes: HashSet<NodeKey> = counts.iter()
            .filter(|(_, c)| c.0 == 1 && c.1 == 1 && c.2 ).map(|(k,_)| k.clone() ).collect();

        //let d_conn: HashSet<usize> = self.connections.iter()
        //    .filter(|(_,v)| (counts.get(&v.to).unwrap().0 > 1 && counts.get(&v.from).unwrap().1 > 1) || v.to == v.from )
        //    .map(|(k,_)| *k ).collect();

        let d_conn: HashSet<usize> = self.connections.iter().filter(|(ori_k, ori_v)|{
            ori_v.to == ori_v.from ||
            (self.connections.iter().find(|(k,v)| **k != **ori_k && v.to == ori_v.to && v.to != v.from && v.active ).is_some() &&
            self.connections.iter().find(|(k,v)| **k != **ori_k && v.from == ori_v.from && v.to != v.from && v.active ).is_some() )
        }).map(|(k,_)| *k).collect();

        if d_nodes.len() > 0 && rng.random_bool(p) {
            let key = d_nodes.iter().choose(&mut rng).unwrap();
            let _node = self.nodes.remove(&key).unwrap();

            let tbd1 = self.connections.iter()
                .find(|(_,c)| c.to == *key && c.to != c.from && c.active ).unwrap().1.clone();
            let tbd2 = self.connections.iter()
                .find(|(_,c)| c.from == *key && c.to != c.from && c.active && **c != tbd1 ).unwrap().1.clone();

            self.connections.retain(|_,c| c.to != *key && c.from != *key );

            // insert average connection, recurrent only if both removed are
            if let Some((_,e_conn)) = self.connections.iter_mut()
                .find(|(_,v)| v.from == tbd1.from && v.to == tbd2.to && (v.recurrent == (tbd1.recurrent && tbd2.recurrent)) ){
                if e_conn.active {
                    e_conn.weight = (e_conn.weight + (tbd1.weight + tbd2.weight)/2. )/2.;
                } else {
                    e_conn.active = true;
                    e_conn.weight = (tbd1.weight + tbd2.weight)/2.;
                }
            } else {
                self.connections.insert(
                    usize::MAX,
                    Connection::new(tbd1.from.clone(), tbd2.to.clone(), tbd1.recurrent && tbd2.recurrent)
                );
                let weight = (tbd1.weight + tbd2.weight) / 2.;
                self.connections.get_mut(&usize::MAX).unwrap().weight = weight;
                out = Some(self.connections.get(&usize::MAX).unwrap().clone());
            }

            // removing gating
            self.connections.values_mut().filter(|v| v.gater.is_some() ).for_each(|v| {
                if v.gater.clone().unwrap() == *key { v.gater = None; }
            });
        }
        else if d_conn.len() > 0 {
            let key = d_conn.iter().choose(&mut rng).unwrap();

            self.connections.remove(key);
            //if k.to != k.from {
            //    let _ = self.connections.values().find(|c| c.to == k.to && c.from != c.to ).unwrap();
            //    let _ = self.connections.values().find(|c| c.from == k.from && c.from != c.to ).unwrap();
            //}
        }
        out
    }

// #########################################################################################################################################
    /// Returns splitted connection innovation number, if Some() global is incremented by 2 .
    fn m_node_add(&mut self) -> Option<( Connection, Connection )> {
        // inserting nodes into recurrent connections, 
        // at the moment both are recurrent
        //
        // insert node in the middle of existing connection
        // in case it isn't first time splitting this node increment second key value

        let mut rng = rand::rng();
        // get connection to be replaced
        let c_key = match self.connections.iter_mut().filter(|(_,c)| c.active).choose(&mut rng) { // f recurrent
            Some(c) => c.0.clone(),
            None => return None,
        };
        let dup = self.nodes.keys().filter(|k| k.sconn == c_key ).count();
        let key = NodeKey::new(c_key, dup);
        let o = self.nodes.insert(
            key.clone(),
            Node::new(Genre::Hidden, &self.function_io));
        if o.is_some() {panic!("Node insert_0 failed")}
        let c = self.connections.get_mut(&c_key).unwrap();
        c.active = false;
        let c = c.clone();
        
        // to new node conn receives weight 1
        self.connections.insert(
            usize::MAX,
            Connection::new(c.from.clone(), key.clone(), c.recurrent)); // ID
        self.connections.get_mut(&usize::MAX).unwrap().assign_weight(1.);
        // from new node conn receives weight from deleted connection
        self.connections.insert(
            usize::MAX-1,
            Connection::new(key.clone(), c.to.clone(), c.recurrent)); // ID
        self.connections.get_mut(&(usize::MAX-1)).unwrap().assign_weight(c.weight);

        Some(( self.connections.get(&usize::MAX).unwrap().clone(), self.connections.get(&(usize::MAX-1)).unwrap().clone() ))
    }

    /// Returns from and to node's innovation id's.
    fn m_connection_add(&mut self) -> Option<Connection> {
        let mut rng = rand::rng();
        // randomly select node index, that have free paths and isn't output, if none return (full)
        match !(rng.random_ratio(1, 3) && self.recurrence.0) {
            true => { // feedforward
                let node_from = match self.nodes.iter().filter(|(_,n)| !n.free_nodes_f.is_empty() ).choose(&mut rng){
                    Some(n) => n,
                    None => return None,
                };
                let key_to = node_from.1.free_nodes_f.iter().choose(&mut rng).unwrap();

                self.connections.insert(
                    usize::MAX,
                    Connection::new(node_from.0.clone(), key_to.clone(), false)
                );
                Some(self.connections.get(&usize::MAX).unwrap().clone())
            },
            false => { // recurrent
                let node_from = match self.nodes.iter().filter(|(_,n)| !n.free_nodes_r.is_empty() ).choose(&mut rng){
                    Some(n) => n,
                    None => return None,
                };
                let key_to = node_from.1.free_nodes_r.iter().choose(&mut rng).unwrap();

                self.connections.insert(
                    usize::MAX,
                    Connection::new(node_from.0.clone(), key_to.clone(), true)
                );
                Some(self.connections.get(&usize::MAX).unwrap().clone())
            }
        }
    }

    /// Assings missing innovation numbers.
    pub fn correct_keys(&mut self, innov_id_1: usize, innov_id_2: usize) -> usize {
        let mut counter = 0; // tracks innovation incrementation
        if let Some(c) = self.connections.remove(&usize::MAX) {
            let a = self.connections.insert(innov_id_1, c);
            if a.is_some() {panic!("Conn insert failed")}
            counter += 1;
        }
        if let Some(c) = self.connections.remove(&(usize::MAX-1)) {
            let a = self.connections.insert(innov_id_2, c);
            if a.is_some() {panic!("Conn insert failed")}
            counter += 1;
        }
        counter
    } 

// #########################################################################################################################################

    fn m_connection_weight(&mut self){
        let mut rng = rand::rng();
        // small chance for new value, otherwise slight change from normal distribution
        // Variability converges to zero as generations approach infinity
        match self.connections.iter_mut().filter(|(_,c)| c.active).choose(&mut rng){
            Some((_,c)) => {
                match rng.random_range(0..10) {
                    0 => c.weight = rng.random_range(-5.0..=5.0),
                    _ => {let dist = Normal::new(0., 8. / (self.generation+4) as f32 ).unwrap();
                        c.weight = (c.weight + dist.sample(&mut rng).max(f32::MIN)).clamp(-9.9, 9.9);
                    }
                }
            },
            None => return,
        }
    }

    // gating uses a second activation value (sigmoid)
    fn m_connection_gater_add(&mut self) {
        let mut rng = rand::rng();
        let connection = match self.connections.iter_mut().filter(|(_,c)| c.gater == None).choose(&mut rng) {
            Some((_,c)) => c,
            None => return,
        };
        match self.nodes.iter().filter(|(k,_)| **k != connection.to && **k != connection.from ).choose(&mut rng) {
            Some((k,_)) => connection.gater = Some(k.clone()),
            None => return,
        }
    }

    fn m_connection_gater_remove(&mut self) {
        let mut rng = rand::rng();
        match self.connections.iter_mut().filter(|(_,c)| c.gater != None).choose(&mut rng) {
            Some((_,c)) => c.gater = None,
            None => return,
        }
    }

    fn m_connection_disable(&mut self) {
        let mut rng = rand::rng();
        match self.connections.iter_mut().filter(|(_,c)| c.active).choose(&mut rng) {
            Some((_,c)) => c.active = false,
            None => return,
        }
    }

    fn m_connection_enable(&mut self) {
        let mut rng = rand::rng();
        match self.connections.iter_mut().filter(|(_,c)| !c.active).choose(&mut rng) {
            Some((_,c)) => c.active = true,
            None => return,
        }
    }

    fn m_node_func(&mut self) {
        let mut rng = rand::rng();
        // randomly select node index, that have free paths and isn't output, if none return (full)
        match self.nodes.iter_mut().filter(|(_,n)| n.genre == Genre::Hidden ).choose(&mut rng){
            Some((_,n)) => n.act_func = ActFunc::random(&self.functions_allowed),
            None => return,
        };
    }

// #########################################################################################################################################

    /// For each node calculates two vectors of node key's 
    /// that can be connected through feedforward and recurrent connections.
    pub fn free_nodes_calc(&mut self) {
        // List of outgoing connections from each node  
        let mut outgoing: HashMap<NodeKey, HashSet<NodeKey>> = HashMap::new();
        self.connections.iter().filter(|(_,c)| !c.recurrent ).for_each(|c|{ 
            outgoing.entry(c.1.from.clone()).or_insert_with(HashSet::new).insert(c.1.to.clone()); 
        });

        let free: Vec<(NodeKey, HashSet<NodeKey>)> = self.nodes.iter().map(|(current_key, current_node)| {        
            (
                current_key.clone(),
                HashSet::from_iter(
                    self.nodes.iter().filter(|(target_key, target_node)| {
                    !outgoing.get(&current_key).map_or(false, |set| set.contains(&target_key)) && 
                    // excludes existing connections 
                    *target_node != current_node && 
                    // disables (recurrent) connections to itself
                    (current_node.genre != Genre::Input || target_node.genre != Genre::Input) &&
                    (current_node.genre != Genre::Output || target_node.genre != Genre::Output) &&
                    !(current_node.genre == Genre::Output && target_node.genre == Genre::Input)
                    //(self.get_node_layer(current_key) < self.get_node_layer(target_key))
                    // disable recurrence between layer elements 
                    }).map(|(k, _)| k.clone() )
                )
            )
        }).collect();
        free.iter().for_each(|(n, keys)| { self.nodes.get_mut(n).unwrap().free_nodes_f = keys.clone();} );
        
        // Same but for recurrent  
        let mut outgoing: HashMap<NodeKey, HashSet<NodeKey>> = HashMap::new();
        self.connections.iter().filter(|(_,c)| c.recurrent ).for_each(|c|{ 
            outgoing.entry(c.1.from.clone()).or_insert_with(HashSet::new).insert(c.1.to.clone()); 
        });

        let free: Vec<(NodeKey, HashSet<NodeKey>)> = self.nodes.iter().map(|(current_key, current_node)| {        
            (
                current_key.clone(),
                HashSet::from_iter(
                    self.nodes.iter().filter(|(target_key, target_node)| {
                    !outgoing.get(&current_key).map_or(false, |set| set.contains(&target_key)) && 
                    // excludes existing connections 
                    (current_node.genre != Genre::Input || target_node.genre != Genre::Input)
                    //(self.get_node_layer(current_key) >= self.get_node_layer(target_key))
                    // disable recurrence between input layer elements 
                    }).map(|(k, _)| k.clone() )
                )
            )
        }).collect();
        free.iter().for_each(|(n, keys)| { self.nodes.get_mut(n).unwrap().free_nodes_r = keys.clone();} );
    }

    // make sure that all nodes are in order
    #[allow(dead_code)]
    fn get_node_layer(&self, key: &NodeKey) -> usize {
        match self.layer_order.iter().position(|layer| layer.contains(key)) {
            Some(t) => t,
            None => panic!("Node not in layer order"),
        }
    }

    /// Layer order:
    ///  - based on a shortest non-recurrent path to input
    ///  - loops aren't a big deal in a continuous environment so they are ignored
    ///  - only recurrent nodes order are based on layer of closest node in layer order 
    /// The feedforward approach should promote more stability, while recurrent approach even if arbitrary, should guarantee visual pretteness ;)
    pub fn sort_layers(&mut self) {
        self.layer_order.clear();
        let layer_0: HashSet<NodeKey> = self.nodes.iter().filter(|(_,n)| n.genre == Genre::Input ).map(|(k,_)|k.clone()).collect();
        self.layer_order.push(layer_0.clone());
        let layer_o: HashSet<NodeKey> = self.nodes.iter().filter(|(_,n)| n.genre == Genre::Output ).map(|(k,_)|k.clone()).collect();
        self.layer_order.push(layer_o.clone());

        // Build adjacency lists for feedforward and recurrent connections.
        let mut feedforward_adj: HashMap<&NodeKey, Vec<&NodeKey>> = HashMap::new();
        self.connections.iter().filter(|(_,c)| c.active && !c.recurrent ).for_each(|(_,c)| {
            feedforward_adj.entry(&c.to).or_default().push(&c.from);
        });

        // Iteratively assign layers to nodes based on feedforward connections.
        let mut layered_nodes: HashSet<NodeKey> = self.layer_order[0].clone(); // Start with input nodes.
        layered_nodes.extend(self.layer_order[1].iter().cloned());// Start with input nodes.
        loop {
            let mut next_layer: HashSet<NodeKey> = HashSet::new();
            for (key, _node) in &self.nodes {
                if layered_nodes.contains(key) {
                    continue; // Skip already layered nodes and input nodes
                }

                // Check if all feedforward predecessors are layered.
                match feedforward_adj.get(key).map_or(false, |predecessors| {
                    predecessors.iter().all(|pred_key| layered_nodes.contains(pred_key))
                }) {
                    true => {
                        next_layer.insert(key.clone());
                        layered_nodes.insert(key.clone());
                    },
                    false => {}
                }
            }
            if next_layer.is_empty() {break;}
            self.layer_order.insert(self.layer_order.len() - 1, next_layer);
        }


        // loop through every layered node and check if there is a connection (probably only rec)
        // if yes, push to the same layer, repeat until didn't added any node
        let mut leftover_nodes: Vec<NodeKey> = self.nodes.iter()
            .filter(|(key, _node)| !layered_nodes.contains(key) )
            .map(|(key, _)| key.clone())
            .collect();

        let mut all_adj: HashMap<&NodeKey, Vec<&NodeKey>> = HashMap::new();
        self.connections.iter().filter(|(_,c)| c.active ).for_each(|(_,c)| {
            all_adj.entry(&c.to).or_default().push(&c.from);
        });

        loop {
            let mut glued: HashMap<NodeKey, usize> = HashMap::new();

            leftover_nodes.iter().for_each(|n| { // each leftover node 
                all_adj.get(n).map(|a| { // each from nodes 
                    a.iter().for_each(|a_element| { // each from node 
                        // if from node is in layer order, push leftover node to glued with layer 
                        match self.layer_order.iter().position(|l| l.contains(&a_element) ) {
                            Some(i) => {glued.insert(n.clone(), i);},
                            None => {}
                        };
                    } );
                } );
            } );

            if glued.is_empty() {break;}
            glued.iter().for_each(|(k,i)| {
                leftover_nodes.retain(|a| a != k );
                self.layer_order.get_mut(*i).unwrap().insert(k.clone());
            } );
        }
        self.idle.clear();
        leftover_nodes.iter().for_each(|k| {self.idle.insert(k.clone());} );

        // correct first and last
        if self.layer_order[0].len() > self.size.0 {
            self.layer_order[0].retain(|k| self.nodes.get(k).unwrap().genre != Genre::Input );
            self.layer_order.insert(0, layer_0);
        }
        let l = self.layer_order.len() -1;
        if self.layer_order[l].len() > self.size.1 {
            self.layer_order[l].retain(|k| self.nodes.get(k).unwrap().genre != Genre::Output );
            self.layer_order.push(layer_o);
        }
    }

// #########################################################################################################################################
    
    pub fn get_outputs(&self) -> &Vec<f32> {
        &self.outputs
    }
    
    pub fn get_pruning(&self) -> &(bool, f32) {
        &self.pruning
    }
    
    pub fn set_pruning(&mut self, enabled: bool, ratio: f32) {
        self.pruning = (enabled, ratio);
    }

    pub fn get_chances(&self) -> &[usize; 8] {
        &self.chances
    }

    pub fn set_chances(&mut self, ch: &[usize]) {
        let mut size = ch.len();
        if size > 6 {size = 6;}
        for i in 0..size {
            self.chances[i] = ch[i];
        }
    }

    /// Save nn to file.
    pub fn save(&self, path: &str) {        
        let toml: String = toml::to_string(
            &self
        ).unwrap();
    
        let mut file = File::create(path).unwrap();
        file.write(toml.as_bytes()).unwrap();
    }

    /// Load nn from file.
    pub fn load(&mut self, path: &str) {
        let mut toml = String::new();
        let mut file = File::open(path).unwrap();

        file.read_to_string(&mut toml).unwrap();
        let decoded: NN = toml::from_str(&toml).unwrap();

        *self = decoded;
    }
}


impl fmt::Debug for NN {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut l = "---NN--- \n".to_string();
        l += "Nodes: \n";
        self.nodes.iter().sorted_by_key(|(k,_)| k.sconn ).for_each(|(i,n)|{
            l += &(format!("{:?} {:?}", i, n) + "\n");
        });
        l += "Connections: \n";
        self.connections.iter().sorted_by_key(|(k,_)| **k ).for_each(|c|{
            l += &(format!("{:>4} {:?}", c.0, c.1) + "\n");
        });
        l += "Order: ";
        self.layer_order.iter().for_each(|layer| {
            l += "[ ";
            layer.iter().for_each(|k| {
                l += &k.to_string();
            } );
            l += "]";
        } );
        l += "\n";
        l += "Idle: [ ";
        self.idle.iter().for_each(|k| {
            l += &k.to_string();
        } );
        l += "]\n";

        l += &format!("Outputs: [ ");
        self.outputs.iter().for_each(|o| {
            l += &format!("{:>+.3}, ", o);
        } );
        l += "]\n";
        l += &("Gen: ".to_string() + &self.generation.to_string() + "\n");
        l += &(format!("Chances: {:?}", self.chances) + "\n");
        l += &("Recurrence: ".to_string() + &self.recurrence.0.to_string() + &format!("| {:>.2}", self.recurrence.1) + "\n");
        l += &("Species: ".to_string() + &format!("{}", self.species) + "\n" );
        l += &("Fitness: ".to_string() + &format!("{:.3}", self.fitness) + "\n" );
        write!(fmt, "{}", l)
    }
}
