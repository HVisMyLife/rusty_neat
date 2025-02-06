# rusty_neat
Simple lib for handling Neuroevolution of augmenting topologies.
It may not be exact implementation of that alghoritm, but it's simple, fast, and easy to use. Moreover, it just works for my use cases, so that's that.
Documentation and lib still in progress

### What it does?
In short, you start by creating input and output nodes, without any connections, etc.
Then, through randomly evolving, there are added intermediate nodes and connections, which characteristics are randomly changed.
Speciation and crossovers should be in next release

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
    40 => add new random connection,
    14 => add new random node,
    2% => connection_enable,
    0% => connection_disable,
    10% => change one of nodes activation function,
```

Struct NN supports serialization and deserialization through serde.
