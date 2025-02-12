use core::fmt;
use std::collections::HashMap;
use rayon::prelude::*;

use crate::{node::NodeKey, nn::NN};


pub struct NeatHandler {
    pub agents: Vec<NN>,
    pub innov_id: usize,
    pub innov_table: HashMap<NodeKey, HashMap<NodeKey, usize>>
}

impl NeatHandler {
    // there need to be minimal (>0) amount of connections at the start
    // but it needs to be done through mutate function, so innovation numbers are kept
    // so for mutation procedure the chances are modified as so each mutation results in new conn 
    pub fn new(agent: &NN, amount: usize) -> Self {
        let agents = (0..amount).into_iter().map(|_| agent.clone() ).collect();
        let mut s = Self { 
            agents, 
            innov_id: agent.size.0+agent.size.1+1, 
            innov_table: HashMap::new()
        };

        let ch: Vec<[usize;8]> = s.agents.iter().map(|a| a.get_chances().clone() ).collect();
        s.agents.par_iter_mut().for_each(|a|{ a.set_chances(&[0,1,0,0,0,0,0,0]); });
        for _ in 0..=(s.agents.first().unwrap().size.0 + s.agents.first().unwrap().size.1)/4 {s.mutate();}
        s.agents.par_iter_mut().zip_eq(ch.par_iter()).for_each(|(a, ch)| { a.set_chances(ch); });
        s
    }

    // connections are the same only if they have same addresses AND appeared in the same gen 
    // another option is to use 2d global connection lookup table that is filled with innov id's 
    // option 1 is in original neat paper 
    // at the moment using 2, bc otherwise innovation numbers explode
    pub fn mutate(&mut self) {
        let mut all_n_conn: Vec<(usize, (NodeKey, NodeKey))> = vec![]; // NN idx, added conn (from, to)
        let mut all_n_node: Vec<(usize, ( (NodeKey,NodeKey), (NodeKey, NodeKey) ))> = vec![]; // NN idx, splitted conn innovation number

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

            match self.innov_table.get(&current.1.0) {
                Some(_) => {},
                None => {self.innov_table.insert(current.1.0.clone(), HashMap::new());},
            }
            match self.innov_table.get(&current.1.0).unwrap().get(&current.1.1) {
                Some(_) => {},
                None => {
                    self.innov_table.get_mut(&current.1.0).unwrap().insert(current.1.1.clone(), self.innov_id);
                    self.innov_id += 1;
                },
            }
            let correct = self.innov_table.get(&current.1.0).unwrap().get(&current.1.1).unwrap();

            all_n_conn.extract_if(.., |nc| nc.1 == current.1).for_each(|nc|{
                assert!(self.agents[nc.0].correct_keys(*correct, 0) == 1);
            });
        }

        while !all_n_node.is_empty() {
            let current = all_n_node.first().unwrap().clone();

            match self.innov_table.get(&current.1.0.0) {
                Some(_) => {},
                None => {self.innov_table.insert(current.1.0.0.clone(), HashMap::new());},
            }
            match self.innov_table.get(&current.1.0.0).unwrap().get(&current.1.0.1) {
                Some(_) => {},
                None => {
                    self.innov_table.get_mut(&current.1.0.0).unwrap().insert(current.1.0.1.clone(), self.innov_id);
                    self.innov_id += 1;
                },
            }
            let correct_1 = self.innov_table.get(&current.1.0.0).unwrap().get(&current.1.0.1).unwrap().clone();
            
            match self.innov_table.get(&current.1.1.0) {
                Some(_) => {},
                None => {self.innov_table.insert(current.1.1.0.clone(), HashMap::new());},
            }
            match self.innov_table.get(&current.1.1.0).unwrap().get(&current.1.1.1) {
                Some(_) => {},
                None => {
                    self.innov_table.get_mut(&current.1.1.0).unwrap().insert(current.1.1.1.clone(), self.innov_id);
                    self.innov_id += 1;
                },
            }
            let correct_2 = self.innov_table.get(&current.1.1.0).unwrap().get(&current.1.1.1).unwrap();

            all_n_node.extract_if(.., |nc| nc.1 == current.1).for_each(|nc|{
                assert!(self.agents[nc.0].correct_keys(correct_1, *correct_2) == 2);
            });
        }
        //while !all_n_node.is_empty() {
        //    let current = all_n_node.first().unwrap().clone();
        //    all_n_node.extract_if(.., |nn| nn.1 == current.1).for_each(|nn|{
        //        assert!(self.agents[nn.0].correct_keys(self.innov_id) == 2);
        //    });
        //    self.innov_id += 2; // new node creates two new connections 
        //}
    }

    //pub fn speciate(&mut self){}

    pub fn forward(&mut self, inputs: Vec<&[f32]>) {
        self.agents.par_iter_mut().zip_eq(inputs.par_iter()).for_each(|(a, i)|{
            a.process_network(i);
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
        l += &( "innov_id: ".to_string() + &(self.innov_id - a.size.0-a.size.1-1).to_string() );
        write!(fmt, "{}", l)
    }
}
