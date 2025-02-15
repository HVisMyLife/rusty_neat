# rusty_neat
Simple lib for handling Neuroevolution of augmenting topologies.
It may not be exact implementation of that alghoritm, but it's simple, fast, and easy to use. Moreover, it just works for my use cases, so that's that.
Documentation and lib still in progress

### NN visual representation 
Network can be saved as .svg: blue connections are recurrent, thickness represents weight, and node 0 is a bias.
![example](https://github.com/HVisMyLife/rusty_neat/blob/master/data/nn.png)

### What it does?
In short, you start by creating input and output nodes, without any connections, etc.
Then, through randomly evolving, there are added intermediate nodes and connections, which characteristics are randomly changed.
NEW: each connection has an optional gating node.
NEW: Speciation and crossovers

### How to use it?
Create large amount of agents, each with each own neural networks. Let simulation run ( or whatever you are using it for ) and after set amount of time choose best of them to be parents of next generation.
Alternatively, you can make agents spawn children after, for example, surviving and getting set amount of points ( what are they depends on use case ).
Next generation should be created (for now) by copying neural network of it's predecessor and mutating it.
Here you have example project that uses it to train "cars" ride along random track: https://github.com/HVisMyLife/neat_race

### Exaple usage:

```rust
    use rusty_neat::{NN, ActFunc};

    fn init() {

        let gens = 10;
        let size = 50;
        let nn = NN::new(2, 2, true, ActFunc::SigmoidBipolar, 0);
        let mut handler = NeatHandler::new(&nn, size);

        handler.forward(vec![vec![1.;2]; size]);
        for _ in 0..5 {
            handler.forward(vec![vec![1.;2]; size]);
        }
        handler.agents.iter_mut().enumerate().for_each(|(_,a)| a.fitness = 100. );
        for _ in 0..5 {
            //while handler.species_table.len().abs_diff(handler.species_amount) < 1 {handler.speciate();}
            handler.speciate();
            handler.next_gen();
            handler.mutate();
        }
    }
```

Possible mutations it's order, and default chances:

```rust
    200 => modify one of connections weight,
    20 => add new random connection,
    5 => add new random node,
    10 => add gating node to connection,
    3 => romove gating node from connection,
    0 => connection_enable,
    0 => connection_disable,
    0 => change one of nodes activation function,
```

Struct NN supports serialization and deserialization through serde.
