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

pub struct NeatHandler {
    pub agents: Vec<NN>,
    pub size: usize, // desired agents amount 
    pub innov_id: usize,
    pub innov_table: HashMap<(NodeKey, NodeKey, bool), usize>, // recurrent flag
    pub species_threshold: f32,  // difference threshold between species 
    pub species_amount: usize, // desired amount of speciec
    pub species_table: HashMap<usize, Species>
} 

impl NeatHandler {
    // there need to be minimal (>0) amount of connections at the start
    // but it needs to be done through mutate function, so innovation numbers are kept
    // so for mutation procedure the chances are modified as so each mutation results in new conn 
    pub fn new(agent: &NN, size: usize) -> Self {
        let agents = (0..size).into_iter().map(|_| agent.clone() ).collect();
        let mut s = Self { 
            agents,
            size,
            innov_id: agent.size.0+agent.size.1+1, 
            innov_table: HashMap::new(),
            species_threshold: 3.,
            species_amount: 5,
            species_table: HashMap::new()
        };

        let ch: Vec<[usize;8]> = s.agents.iter().map(|a| a.get_chances().clone() ).collect();
        s.agents.par_iter_mut().for_each(|a|{ a.set_chances(&[0,1,0,0,0,0,0,0]); });
        for _ in 0..=(s.agents.first().unwrap().size.0 + s.agents.first().unwrap().size.1)/2 {s.mutate();}
        //for _ in 0..2 {s.mutate();}
        s.agents.par_iter_mut().zip_eq(ch.par_iter()).for_each(|(a, ch)| { a.set_chances(ch);});
        s
    }

    // connections are the same only if they have same addresses AND appeared in the same gen 
    // another option is to use 2d global connection lookup table that is filled with innov id's 
    // option 1 is in original neat paper 
    // at the moment using 2, bc otherwise innovation numbers explode
    pub fn mutate(&mut self) {
        let mut all_n_conn: Vec<(usize, Connection)> = vec![]; // NN idx, added conn (from, to)
        let mut all_n_node: Vec<(usize, ( Connection, Connection ))> = vec![]; // NN idx, splitted conn innovation number

        self.agents.iter_mut().enumerate().for_each(|(i, a)| {
            let ( n_conn, n_node) = a.mutate();
            match n_conn {
                None => {},
                Some(c) => {all_n_conn.push((i, c));},
            }
            match n_node {
                None => {},
                Some(n) => {all_n_node.push((i, n));},
            }
        });
        // clones first element and extracts every other that looks like the same mutation
        // repeats until all elements are extracted
        while !all_n_conn.is_empty() {
            let current = all_n_conn.first().unwrap().clone();

            match self.innov_table.get(&(current.1.from.clone(), current.1.to.clone(), current.1.recurrent)) {
                Some(_) => {},
                None => {
                    self.innov_table.insert((current.1.from.clone(), current.1.to.clone(), current.1.recurrent), self.innov_id);
                    self.innov_id += 1;
                }
            }
            let correct = self.innov_table.get(&(current.1.from.clone(), current.1.to.clone(), current.1.recurrent)).unwrap();

            all_n_conn.extract_if(.., |nc| 
                nc.1.to == current.1.to && nc.1.from == current.1.from && nc.1.recurrent == current.1.recurrent
            ).for_each(|nc|{
                assert!(self.agents[nc.0].correct_keys(*correct, 0) == 1);
            });
        }
        
        while !all_n_node.is_empty() {
            let current = all_n_node.first().unwrap().clone();

            match self.innov_table.get(&(current.1.0.from.clone(), current.1.0.to.clone(), current.1.0.recurrent)) {
                Some(_) => {},
                None => {
                    self.innov_table.insert((current.1.0.from.clone(), current.1.0.to.clone(), current.1.0.recurrent), self.innov_id);
                    self.innov_id += 1;
                }
            }
            match self.innov_table.get(&(current.1.1.from.clone(), current.1.1.to.clone(), current.1.1.recurrent)) {
                Some(_) => {},
                None => {
                    self.innov_table.insert((current.1.1.from.clone(), current.1.1.to.clone(), current.1.1.recurrent), self.innov_id);
                    self.innov_id += 1;
                }
            }
            let correct0 = self.innov_table.get(&(current.1.0.from.clone(), current.1.0.to.clone(), current.1.0.recurrent)).unwrap();
            let correct1 = self.innov_table.get(&(current.1.1.from.clone(), current.1.1.to.clone(), current.1.1.recurrent)).unwrap();

            all_n_node.extract_if(.., |nc| 
                nc.1.0.to == current.1.0.to && nc.1.0.from == current.1.0.from && nc.1.0.recurrent == current.1.0.recurrent &&
                nc.1.1.to == current.1.1.to && nc.1.1.from == current.1.1.from && nc.1.1.recurrent == current.1.1.recurrent
            ).for_each(|nc|{
                assert!(self.agents[nc.0].correct_keys(*correct0, *correct1) == 2);
            });
        }
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
        let mut rng = rand::rng();

        // species fitness 
        self.agents.iter().for_each(|a| 
            self.species_table.get_mut(&a.species).unwrap().fitness += a.fitness / self.species_table.get(&a.species).unwrap().size as f32
        );
        self.species_table.iter_mut().for_each(|(_, s)| {
            s.fitness /= s.size as f32;
            s.history_fitness.push_front(s.fitness);
            s.history_fitness.pop_back();
        });

        let global_fitness: f32 = self.species_table.values().map(|s| s.fitness ).sum::<f32>() / (self.species_table.len() as f32);
        self.species_table.values_mut().for_each(|s|{
            s.offspring = (s.fitness / global_fitness * s.size as f32) as usize;
        });
        // scalling offspring so sum ~ target agents
        let size_scale = self.size as f32 / self.species_table.values().map(|s| s.offspring ).sum::<usize>() as f32;
        self.species_table.values_mut().for_each(|s| s.offspring = (s.offspring as f32 * size_scale) as usize );

        let mut agents_new: Vec<NN> = vec![];
        self.species_table.iter().for_each(|(uuid, species)|{
            let (idxs, fs): (Vec<usize>, Vec<f32>) = self.agents.iter().enumerate().filter(|(_, a)| a.species == *uuid )
                .map(|(ai,a)| (ai, a.fitness) ).collect(); // probabilities

            (0..species.offspring).into_iter().for_each(|_|{
                let dist = WeightedIndex::new(&fs).unwrap();
                let idx0 = idxs[dist.sample(&mut rng)];  // index of chosen parent
                let idx1 = idxs[dist.sample(&mut rng)];  // index of chosen parent
                let mut child = self.agents.get(idx0).unwrap().crossover(self.agents.get(idx1).unwrap());
                child.active = true;
                agents_new.push(child);
            });
        });
        self.agents = agents_new;
    }

    pub fn forward(&mut self, inputs: Vec<Vec<f32>>) {
        self.agents.par_iter_mut().zip_eq(inputs.par_iter()).for_each(|(a, i)|{
            if a.active {a.process_network(i);}
        });
    }

    pub fn get_outputs(&self, id: usize) -> &Vec<f32>{
        self.agents[id].get_outputs()
    }
}

impl fmt::Debug for NeatHandler {
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
