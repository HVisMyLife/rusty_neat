use core::fmt;
use std::collections::{HashMap, VecDeque};
use itertools::Itertools;
use rand_distr::{weighted::WeightedIndex, Distribution};
use rayon::prelude::*;

use crate::{nn::NN, node::NodeKey, ActFunc, Connection};

// Single species data.
pub struct Species {
    fitness: f32,
    pub size: usize,
    pub offspring: usize,
    history_fitness: VecDeque<f32>,
}
impl Species {
    pub fn new(fitness: f32) -> Self {
        Self { fitness, size: 0, offspring: 0, history_fitness: VecDeque::from_iter(std::iter::repeat(0.).take(20)) }
    }
}
impl fmt::Debug for Species {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let l = format!("[{}:{}, f:{:>.1}]", self.size, self.offspring, self.fitness);
        write!(fmt, "{}", l)
    }
}

/**
Struct for handling real-time neuroevolution.
Time isn't divided by generations, instead each entity produces offspring on-the-run.
Great for ecosystem simulation.
*/
pub struct NeatContinous {
    /// HashMap of all networks.
    pub agents: HashMap<usize, NN>,
    /// First free innovation number.
    pub innov_id: usize,
    /// If connection have the same souce, destination, and recurrency, it has also same id.
    pub innov_table: HashMap<(NodeKey, NodeKey, bool), usize>,
    /// Below this threshold agents are assigned to the same species.
    pub species_threshold: f32,
    /// Desired amout of species in ecosystem.
    pub species_amount: usize,
    /// HashMap of all non-empty species.
    pub species_table: HashMap<usize, Species>
} 
impl NeatContinous {
    /// Each agent is a clone, but with it's own (random) initial genes.
    pub fn new(agent: &NN, size: usize, species_amount: usize) -> Self {
        let mut agents = HashMap::new();
        (0..size).into_iter().for_each(|k| { agents.insert(k, agent.clone()); } );
        let mut s = Self { 
            agents,
            innov_id: agent.size.0+agent.size.1+1+agent.size_free.0+agent.size_free.1, 
            innov_table: HashMap::new(),
            species_threshold: 3.,
            species_amount,
            species_table: HashMap::new()
        };

        s.agents.par_iter_mut().for_each(|(_, a)|{ a.set_chances(&[0,1,0,0,0,0,0,0]); a.recurrence.0 = false; });
        let keys: Vec<usize> = s.agents.keys().cloned().collect();
        for _ in 0..=(agent.size.0 + agent.size.1)/2 { for k in &keys {s.mutate(k);} }
        s.agents.par_iter_mut().for_each(|(_, a)| { a.set_chances(agent.get_chances()); a.recurrence = agent.recurrence; });
        s
    }
    /// Panics if any agent have no free space.
    pub fn add_input(&mut self) {
        if ! self.agents.values_mut().all(|a| a.add_input() ) {panic!("No more space for inputs")}
    }
    /// Panics if any agent have no free space.
    pub fn add_output(&mut self, func: &ActFunc) {
        if ! self.agents.values_mut().all(|a| a.add_output(func) ) {panic!("No more space for outputs")}
    }
    /// Creates new agent by crossing key's with other from the same species.
    /// Probably good idea to assign species to it right after.
    pub fn offspring(&mut self, key: &usize) -> usize {
        let agent_0 = self.agents.get(key).unwrap();
        let child_key = self.agents.keys().max().unwrap() + 1;

        let (keys, fs): (Vec<usize>, Vec<f32>) = self.agents.iter().filter(|(_, a)| a.species == agent_0.species )
            .map(|(k,a)| (k, a.fitness+1.) ).collect(); // probabilities
            
        let mut rng = rand::rng();

        let dist = WeightedIndex::new(&fs).unwrap();
        let agent_1 = self.agents.get( &keys[dist.sample(&mut rng)] ).unwrap();
        let mut child = agent_0.crossover(agent_1);
        child.active = true;
        self.agents.insert(child_key, child);
        self.mutate(&child_key);
        //self.species_assign(&child_key); // assign to species 

        child_key
    }
    /// Mutates agent and corrects innovation numbers (if needed).
    pub fn mutate(&mut self, key: &usize) {
        let agent = &mut self.agents.get_mut(key).unwrap();
        let ( n_conn, n_node) = agent.mutate();

        if n_conn.is_some() {
            let current = n_conn.unwrap();
            if !self.innov_table.contains_key(&(current.from.clone(), current.to.clone(), current.recurrent)) {
                self.innov_table.insert((current.from.clone(), current.to.clone(), current.recurrent), self.innov_id);
                self.innov_id += 1;
            }
            let correct = self.innov_table.get(&(current.from.clone(), current.to.clone(), current.recurrent)).unwrap();

            assert!(agent.correct_keys(*correct, 0) == 1);
        }
        
        if n_node.is_some() {
            let current = n_node.unwrap();
            if !self.innov_table.contains_key(&(current.0.from.clone(), current.0.to.clone(), current.0.recurrent)) {
                self.innov_table.insert((current.0.from.clone(), current.0.to.clone(), current.0.recurrent), self.innov_id);
                self.innov_id += 1;
            }
            if !self.innov_table.contains_key(&(current.1.from.clone(), current.1.to.clone(), current.1.recurrent)) {
                self.innov_table.insert((current.1.from.clone(), current.1.to.clone(), current.1.recurrent), self.innov_id);
                self.innov_id += 1;
            }
            let correct0 = self.innov_table.get(&(current.0.from.clone(), current.0.to.clone(), current.0.recurrent)).unwrap();
            let correct1 = self.innov_table.get(&(current.1.from.clone(), current.1.to.clone(), current.1.recurrent)).unwrap();

            assert!(agent.correct_keys(*correct0, *correct1) == 2);
        }
    }

    // ********************************************************************************************
    /// Assigns agent to species according to threshold. 
    pub fn species_assign(&mut self, key: &usize) -> usize {
        self.species_prune();
        let mut species = None;
        let reference = self.agents.get(key).unwrap();

        // for loop needed bc continue/break doesn't work in for_each 
        // sorted from smallest species to promote them
        for s in self.species_table.iter_mut().sorted_by_key(|(_,s)| s.size ) {
            match self.agents.values().find(|a| a.species == *s.0 ) {
                Some(a) => {
                    if a.compare(reference, 1., 1., 0.4, 1.) < self.species_threshold {
                        species = Some(a.species); break;
                    } else {
                        continue;
                    }
                }
                None => continue,
            }
        }
        if species.is_none() {
            let uuid = self.species_table.keys().max().unwrap_or(&0) + 1;
            self.species_table.insert(uuid, Species::new(1.));
            species = Some(uuid);
        }

        self.agents.get_mut(key).unwrap().species = species.unwrap();
        self.species_threshold_correct();
        species.unwrap()
    }
    /// Corrects threshold to hit target amout of species.
    /// Should be run after every offspring.
    pub fn species_threshold_correct(&mut self) { 
        // no need for more intelligent approach bc of frequent usage
        if self.species_amount > self.species_table.len() + 1 { self.species_threshold -= 0.02 }
        else if self.species_amount < self.species_table.len() - 1 { self.species_threshold += 0.02 }
        self.species_threshold = self.species_threshold.clamp(0.1, 8.);
    }
    fn species_prune(&mut self) {
        for s in &mut self.species_table {
            s.1.size = self.agents.iter().filter(|(_, a)| a.species == *s.0 ).count();
        }
        self.species_table.retain(|_, s| s.size > 0 );
    }
    /// In continous type it's used only at init.
    /// Should be run several times, if you want to hit target amount of species.
    pub fn speciate(&mut self){
        let mut refs = self.agents.iter_mut().map(|a| a).collect_vec();
        for s in &mut self.species_table {
            let tbr;
            match refs.iter().position(|(_, a)| a.species == *s.0 ) {
                Some(p) => tbr = p,
                None => {s.1.size = 0; continue;},
            }
            let f = refs.swap_remove(tbr);
            s.1.size = 1;

            let mut assigned: Vec<usize> = vec![];
            refs.iter_mut().enumerate().for_each(|(i,(_, a))| {
                let t = f.1.compare(a, 1., 1., 0.4, 1.);
                if t < self.species_threshold { 
                    assigned.push(i);
                    a.species = f.1.species;
                    s.1.size += 1;
                }
            } );
            // sorted and inverted so indexes doesn't change
            assigned.iter().sorted().rev().for_each(|i| { refs.remove(*i); } );
        }
        self.species_table.retain(|_, s| s.size > 0 );
        let mut uuid: usize = self.species_table.keys().max().unwrap_or(&0) + 1;

        // creating new species for leftovers
        while !refs.is_empty() {
            let f = refs.swap_remove(0);
            self.species_table.insert(uuid, Species::new(f.1.fitness));
            f.1.species = uuid;
            self.species_table.get_mut(&uuid).unwrap().size = 1;

            // compare every leftover to the leader and assign if matches
            let mut assigned: Vec<usize> = vec![];
            refs.iter_mut().enumerate().for_each(|(i,(_, a))| {
                let t = f.1.compare(a, 1., 1., 0.4, 1.);
                if t < self.species_threshold { 
                    assigned.push(i);
                    a.species = uuid;
                    self.species_table.get_mut(&uuid).unwrap().size += 1;
                }
            } );
            // sorted and inverted so indexes doesn't change
            assigned.iter().sorted().rev().for_each(|i| { refs.remove(*i); } );
            
            uuid += 1;
        }
        self.species_threshold_correct();
    }
    // ********************************************************************************************
    
    /// Removes agents, that do not have input vector.
    /// Should be run before every forward(), if any agent could have died.
    pub fn check_integrity(&mut self, inputs: &HashMap<usize, Vec<f32>>){
        let mut empty = vec![];
        self.agents.keys().for_each(|k|{
            match inputs.get(k) {
                Some(_) => {}
                None => empty.push(*k),
            }
        });
        empty.iter().for_each(|k| {self.agents.remove(k);} );
    }

    /// Takes inputs for each agent, and runs it.
    pub fn forward(&mut self, inputs: &HashMap<usize, Vec<f32>>) {
        self.agents.par_iter_mut().for_each(|(k,a)|{
            if a.active {a.process_network(inputs.get(k).unwrap());}
        });
    }
    /// Gets single network's output node's value's.
    pub fn get_outputs(&self, key: &usize) -> &Vec<f32>{
        self.agents.get(&key).unwrap().get_outputs()
    }
    /// If enabled each mutation deletes node or connection.
    /// Ratio of 0.1 means, that there is 10% chance of deleting node
    /// , and 90% of deleting connection.
    pub fn set_pruning(&mut self, enabled: bool, ratio: f32) {
        self.agents.par_iter_mut().for_each(|(_,a)| a.set_pruning(enabled, ratio) );
    }
}


/**
Struct for handling generation-based neuroevolution.
Time is divided in evaluation runs.
Great for training networks to do well defined tasks.
*/
pub struct NeatIntermittent {
    pub agents: Vec<NN>,
    /// First free innovation number.
    pub size: usize,
    /// First free innovation number.
    pub innov_id: usize,
    /// If connection have the same souce, destination, and recurrency, it has also same id.
    pub innov_table: HashMap<(NodeKey, NodeKey, bool), usize>,
    /// Below this threshold agents are assigned to the same species.
    pub species_threshold: f32,
    /// Desired amout of species in ecosystem.
    pub species_amount: usize, 
    /// HashMap of all non-empty species.
    pub species_table: HashMap<usize, Species>
} 

impl NeatIntermittent {
    // there need to be minimal (>0) amount of connections at the start
    // but it needs to be done through mutate function, so innovation numbers are kept
    // so for mutation procedure the chances are modified as so each mutation results in new conn
    /// Each agent is a clone, but with it’s own (random) initial genes.
    pub fn new(agent: &NN, size: usize, species_amount: usize) -> Self {
        let agents = (0..size).into_iter().map(|_| agent.clone() ).collect();
        let mut s = Self { 
            agents,
            size,
            innov_id: agent.size.0+agent.size.1+1+agent.size_free.0+agent.size_free.1, 
            innov_table: HashMap::new(),
            species_threshold: 3.,
            species_amount,
            species_table: HashMap::new()
        };

        s.agents.par_iter_mut().for_each(|a|{ a.set_chances(&[0,1,0,0,0,0,0,0]); });
        for _ in 0..=(agent.size.0 + agent.size.1)/2 {s.mutate(None);}
        s.agents.par_iter_mut().for_each(|a| { a.set_chances(agent.get_chances()); });
        s
    }

    /// Panics if any agent have no free space.
    pub fn add_input(&mut self) {
        if ! self.agents.iter_mut().all(|a| a.add_input() ) {panic!("No more space for inputs")}
    }
    /// Panics if any agent have no free space.
    pub fn add_output(&mut self, func: &ActFunc) {
        if ! self.agents.iter_mut().all(|a| a.add_output(func) ) {panic!("No more space for outputs")}
    }
    // connections are the same only if they have same addresses AND appeared in the same gen 
    // another option is to use 2d global connection lookup table that is filled with innov id's 
    // option 1 is in original neat paper 
    // at the moment using 2, bc otherwise innovation numbers explode
    /// Mutates agent and corrects innovation numbers (if needed).
    /// If "single" is provided, only agent with that index is mutated.
    pub fn mutate(&mut self, single: Option<usize>) {
        let mut all_n_conn: Vec<(usize, Connection)> = vec![]; // NN idx, added conn (from, to)
        let mut all_n_node: Vec<(usize, ( Connection, Connection ))> = vec![]; // NN idx, splitted conn innovation number

        match single {
            Some(i) => {
                let ( n_conn, n_node) = self.agents[i].mutate();
                match n_conn {
                    None => {},
                    Some(c) => {all_n_conn.push((i, c));},
                }
                match n_node {
                    None => {},
                    Some(n) => {all_n_node.push((i, n));},
                }
            }
            None => {
                let results: Vec<(usize, (Option<Connection>, Option<(Connection, Connection)>))> = self.agents
                .par_iter_mut().enumerate()
                .map(|(i, a)| (i, a.mutate()) ).collect();
                
                let conn_to_extend: Vec<_> = results.par_iter().filter_map(|(i, (n_conn, _))| {
                    n_conn.as_ref().map(|conn_ref| (*i, conn_ref.clone()))
                }).collect();
                all_n_conn.extend(conn_to_extend);
                
                let node_to_extend: Vec<_> = results.par_iter().filter_map(|(i, (_, n_node))| {
                    n_node.as_ref().map(|node_ref| (*i, node_ref.clone()))
                }).collect();
                all_n_node.extend(node_to_extend);
            }
        }
        // clones first element and extracts every other that looks like the same mutation
        // repeats until all elements are extracted
        all_n_conn.iter().for_each(|current| {
            if !self.innov_table.contains_key(&(current.1.from.clone(), current.1.to.clone(), current.1.recurrent)) {
                self.innov_table.insert((current.1.from.clone(), current.1.to.clone(), current.1.recurrent), self.innov_id);
                self.innov_id += 1;
            }
            let correct = self.innov_table.get(&(current.1.from.clone(), current.1.to.clone(), current.1.recurrent)).unwrap();

            assert!(self.agents[current.0].correct_keys(*correct, 0) == 1);
        });
        
        all_n_node.iter().for_each(|current| {
            if !self.innov_table.contains_key(&(current.1.0.from.clone(), current.1.0.to.clone(), current.1.0.recurrent)) {
                self.innov_table.insert((current.1.0.from.clone(), current.1.0.to.clone(), current.1.0.recurrent), self.innov_id);
                self.innov_id += 1;
            }
            if !self.innov_table.contains_key(&(current.1.1.from.clone(), current.1.1.to.clone(), current.1.1.recurrent)) {
                self.innov_table.insert((current.1.1.from.clone(), current.1.1.to.clone(), current.1.1.recurrent), self.innov_id);
                self.innov_id += 1;
            }
            let correct0 = self.innov_table.get(&(current.1.0.from.clone(), current.1.0.to.clone(), current.1.0.recurrent)).unwrap();
            let correct1 = self.innov_table.get(&(current.1.1.from.clone(), current.1.1.to.clone(), current.1.1.recurrent)).unwrap();

            assert!(self.agents[current.0].correct_keys(*correct0, *correct1) == 2);
        });
    }

    /// Assigns all agents to species, and corrects threshold.
    /// At init should be run few times.
    pub fn speciate(&mut self){
        let mut refs = self.agents.iter_mut().map(|a| a).collect_vec();
        for s in &mut self.species_table {
            let tbr;
            match refs.iter().position(|a| a.species == *s.0 ) {
                Some(p) => tbr = p,
                None => {s.1.size = 0; continue;},
            }
            let f = refs.swap_remove(tbr);
            s.1.size = 1;

            let mut assigned: Vec<usize> = vec![];
            refs.iter_mut().enumerate().for_each(|(i,a)| {
                let t = f.compare(a, 1., 1., 0.4, 1.);
                if t < self.species_threshold { 
                    assigned.push(i);
                    a.species = f.species;
                    s.1.size += 1;
                }
            } );
            // sorted and inverted so indexes doesn't change
            assigned.iter().sorted().rev().for_each(|i| { refs.remove(*i); } );
        }
        self.species_table.retain(|_, s| s.size > 0 );
        let mut uuid: usize = self.species_table.keys().max().unwrap_or(&0) + 1;

        // creating new species for leftovers
        while !refs.is_empty() {
            let f = refs.swap_remove(0);
            self.species_table.insert(uuid, Species::new(f.fitness));
            f.species = uuid;
            self.species_table.get_mut(&uuid).unwrap().size = 1;

            // compare every leftover to the leader and assign if matches
            let mut assigned: Vec<usize> = vec![];
            refs.iter_mut().enumerate().for_each(|(i,a)| {
                let t = f.compare(a, 1., 1., 0.4, 1.);
                if t < self.species_threshold { 
                    assigned.push(i);
                    a.species = uuid;
                    self.species_table.get_mut(&uuid).unwrap().size += 1;
                }
            } );
            // sorted and inverted so indexes doesn't change
            assigned.iter().sorted().rev().for_each(|i| { refs.remove(*i); } );
            
            uuid += 1;
        }
        let diff = 1. - (self.species_table.len() as f32) / (self.species_amount as f32);
        self.species_threshold -= diff.clamp(-2., 2.);
    }

    /// Creates new agent's generation. Each species have offspring size based on it's size and avg fitness.
    /// Inside single species, higher fitness means more chance to become parent.
    pub fn next_gen(&mut self) {

        // species fitness 
        self.agents.iter().for_each(|a| 
            self.species_table.get_mut(&a.species).unwrap().fitness += a.fitness / self.species_table.get(&a.species).unwrap().size as f32
        );
        self.species_table.par_iter_mut().for_each(|(_, s)| {
            s.fitness /= s.size as f32;
            s.history_fitness.push_front(s.fitness);
            s.history_fitness.pop_back();
        });

        let global_fitness: f32 = self.species_table.par_iter().map(|(_, s)| s.fitness ).sum::<f32>() / (self.species_table.len() as f32);
        self.species_table.par_iter_mut().for_each(|(_, s)|{
            s.offspring = (s.fitness / global_fitness * s.size as f32) as usize;
        });
        // scalling offspring so sum ~ target agents
        let size_scale = self.size as f32 / self.species_table.par_iter().map(|(_,s)| s.offspring ).sum::<usize>() as f32;
        self.species_table.par_iter_mut().for_each(|(_,s)| s.offspring = (s.offspring as f32 * size_scale) as usize );

        self.agents = self.species_table.par_iter().flat_map(|(uuid, species)|{
            let (idxs, fs): (Vec<usize>, Vec<f32>) = self.agents.iter().enumerate().filter(|(_, a)| a.species == *uuid )
                .map(|(ai,a)| (ai, a.fitness+1.) ).collect(); // probabilities
            
            let mut rng = rand::rng();
            let mut agents: Vec<NN> = vec![];

            (0..species.offspring).into_iter().for_each(|_|{
                let dist = WeightedIndex::new(&fs).unwrap();
                let idx0 = idxs[dist.sample(&mut rng)];  // index of chosen parent
                let idx1 = idxs[dist.sample(&mut rng)];  // index of chosen parent
                let mut child = self.agents.get(idx0).unwrap().crossover(self.agents.get(idx1).unwrap());
                child.active = true;
                agents.push(child);
            });
            agents
        }).collect::<Vec<NN>>();
    }

    /// Forwards inputs to all agents
    pub fn forward(&mut self, inputs: &Vec<Vec<f32>>) {
        self.agents.par_iter_mut().zip_eq(inputs.par_iter()).for_each(|(a, i)|{
            if a.active {a.process_network(i);}
        });
    }

    /// Gets output node's value's of single agent.
    pub fn get_outputs(&self, id: usize) -> &Vec<f32>{
        self.agents[id].get_outputs()
    }
    /// If enabled each mutation deletes node or connection. 
    /// Ratio of 0.1 means, that there is 10% chance of deleting node , and 90% of deleting connection.
    pub fn set_pruning(&mut self, enabled: bool, ratio: f32) {
        self.agents.par_iter_mut().for_each(|a| a.set_pruning(enabled, ratio) );
    }
}

impl fmt::Debug for NeatIntermittent {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut l = "<<<Neat Handler>>>\n".to_string();
        self.agents.iter().for_each(|a| {
            l += &(format!("{:?}\n", *a));
        } );
        let a = self.agents.first().unwrap();
        l += &( "innov_id: ".to_string() + &(self.innov_id - a.size.0-a.size.1-1).to_string() + "\n" );
        l += &( "s_threshold: ".to_string() + &self.species_threshold.to_string() );
        write!(fmt, "{}", l)
    }
}
