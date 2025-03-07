use core::fmt;
use std::collections::{HashMap, VecDeque};
use itertools::Itertools;
use rand_distr::{weighted::WeightedIndex, Distribution};
use rayon::prelude::*;

use crate::{nn::NN, node::NodeKey, Connection};

pub struct Species {
    fitness: f32,
    size: usize,
    offspring: usize,
    history_fitness: VecDeque<f32>
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

pub struct NeatContinous {
    pub agents: HashMap<usize, NN>,
    pub innov_id: usize,
    pub innov_table: HashMap<(NodeKey, NodeKey, bool), usize>, // recurrent flag
    species_threshold: f32,  // difference threshold between species 
    pub species_amount: usize, // desired amount of speciec
    pub species_table: HashMap<usize, Species>
} 

impl NeatContinous {
    pub fn new(agent: &NN, size: usize) -> Self {
        //let agents = (0..size).into_iter().map(|_| agent.clone() ).collect();
        let mut agents = HashMap::new();
        (0..size).into_iter().for_each(|k| { agents.insert(k, agent.clone()); } );
        let mut s = Self { 
            agents,
            innov_id: agent.size.0+agent.size.1+1+agent.size_free.0+agent.size_free.1, 
            innov_table: HashMap::new(),
            species_threshold: 3.,
            species_amount: 5,
            species_table: HashMap::new()
        };

        s.agents.par_iter_mut().for_each(|(_, a)|{ a.set_chances(&[0,1,0,0,0,0,0,0]); a.recurrence = false; });
        let keys: Vec<usize> = s.agents.keys().cloned().collect();
        for _ in 0..=(agent.size.0 + agent.size.1)/2 { for k in &keys {s.mutate(k);} }
        s.agents.par_iter_mut().for_each(|(_, a)| { a.set_chances(agent.get_chances()); a.recurrence = agent.recurrence; });
        s
    }
    pub fn add_input(&mut self) {
        if ! self.agents.values_mut().all(|a| a.add_input() ) {panic!("No more space for inputs")}
    }
    pub fn add_output(&mut self) {
        if ! self.agents.values_mut().all(|a| a.add_output() ) {panic!("No more space for outputs")}
    }
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

        child_key
    }
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
                let t = f.1.compare(a, 1., 1., 0.4);
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
                let t = f.1.compare(a, 1., 1., 0.4);
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

    pub fn forward(&mut self, inputs: &HashMap<usize, Vec<f32>>) {
        self.agents.par_iter_mut().for_each(|(k,a)|{
            if a.active {a.process_network(inputs.get(k).unwrap());}
        });
    }
    pub fn get_outputs(&self, key: &usize) -> &Vec<f32>{
        self.agents.get(&key).unwrap().get_outputs()
    }
    pub fn set_pruning(&mut self, enabled: bool, ratio: f32) {
        self.agents.par_iter_mut().for_each(|(_,a)| a.set_pruning(enabled, ratio) );
    }
}

pub struct NeatIntermittent {
    pub agents: Vec<NN>,
    pub size: usize, // desired agents amount 
    pub innov_id: usize,
    pub innov_table: HashMap<(NodeKey, NodeKey, bool), usize>, // recurrent flag
    species_threshold: f32,  // difference threshold between species 
    pub species_amount: usize, // desired amount of speciec
    pub species_table: HashMap<usize, Species>
} 

impl NeatIntermittent {
    // there need to be minimal (>0) amount of connections at the start
    // but it needs to be done through mutate function, so innovation numbers are kept
    // so for mutation procedure the chances are modified as so each mutation results in new conn 
    pub fn new(agent: &NN, size: usize) -> Self {
        let agents = (0..size).into_iter().map(|_| agent.clone() ).collect();
        let mut s = Self { 
            agents,
            size,
            innov_id: agent.size.0+agent.size.1+1+agent.size_free.0+agent.size_free.1, 
            innov_table: HashMap::new(),
            species_threshold: 3.,
            species_amount: 5,
            species_table: HashMap::new()
        };

        s.agents.par_iter_mut().for_each(|a|{ a.set_chances(&[0,1,0,0,0,0,0,0]); });
        for _ in 0..=(agent.size.0 + agent.size.1)/2 {s.mutate(None);}
        s.agents.par_iter_mut().for_each(|a| { a.set_chances(agent.get_chances()); });
        s
    }

    pub fn add_input(&mut self) {
        if ! self.agents.iter_mut().all(|a| a.add_input() ) {panic!("No more space for inputs")}
    }
    pub fn add_output(&mut self) {
        if ! self.agents.iter_mut().all(|a| a.add_output() ) {panic!("No more space for outputs")}
    }
    // connections are the same only if they have same addresses AND appeared in the same gen 
    // another option is to use 2d global connection lookup table that is filled with innov id's 
    // option 1 is in original neat paper 
    // at the moment using 2, bc otherwise innovation numbers explode
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
                let t = f.compare(a, 1., 1., 0.4);
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
                let t = f.compare(a, 1., 1., 0.4);
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

    pub fn forward(&mut self, inputs: &Vec<Vec<f32>>) {
        self.agents.par_iter_mut().zip_eq(inputs.par_iter()).for_each(|(a, i)|{
            if a.active {a.process_network(i);}
        });
    }

    pub fn get_outputs(&self, id: usize) -> &Vec<f32>{
        self.agents[id].get_outputs()
    }
    
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
