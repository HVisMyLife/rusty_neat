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
ALMOST DONE: Speciation and crossovers

### How to use it?
Create large amount of agents, each with each own neural networks. Let simulation run ( or whatever you are using it for ) and after set amount of time choose best of them to be parents of next generation.
Alternatively, you can make agents spawn children after, for example, surviving and getting set amount of points ( what are they depends on use case ).
Next generation should be created (for now) by copying neural network of it's predecessor and mutating it.
Here you have example project that uses it to train "cars" ride along random track: https://github.com/HVisMyLife/neat_race

### Exaple usage:

```rust
    use rusty_neat::{NN, ActFunc};

    fn init() {

    // init new network with amount of (input, output) nodes, recurrent connections, default activation function
        let mut net = NN::new(3, 2, true, ActFunc::Tanh);

    // set diffrent ( than default ) chances of mutations, sum (eg. 100%) doesn't matter
        net.set_chances(&[20, 20, 20, 0, 0, 0])

    // evolve network, mutations are chosen randomly, according to above settings,
        for _ in 0..32 {
            net.mutate();
        }
    
    // forward inputs through net and return outputs
        let outputs = net.forward(&[0.5, 0.2, 0.8]);

    // access internal structure
        // calculation order, may be used for creating graph
        println!("\nOrder: \n{:?}", net.layer_order);
        // list of connections
        println!("\nConnections: \n{:?}", net.connections);
        // list of nodes
        println!("\nNodes: \n{:?}", net.nodes);
        // pretty debug
        println!("\n{:?}", net);

    // save network to file
        net.save("path");

    // load network from file
        net.load("path");
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
