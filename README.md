# rusty_neat
Simple lib for handling Neuroevolution of augmenting topologies.
Documentation and program still in progress

### Exaple usage:

    fn init() {

    // init new network with amount of (input, output) nodes
        let mut net = NN::new(3, 2);

    // evolve network, mutations are chosen randomly,
    // in future there will be an interface for choosing types and chances
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

    // save network to file
        net.save("path");

    // load network from file
        net.load("path");
    }

Possible mutations and it's hardcoded propabilities:
    35% => modify one of connections weight,
    35% => modify one of nodes bias,
    10% => change one of nodes activation function,
    10% => add new random connection,
    10% => add new random node,
    0% => connection_enable,
    0% => connection_disable,

Soon I will add interface for modyfing those propabilities
