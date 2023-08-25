# rusty_neat
Simple lib for handling Neuroevolution of augmenting topologies.
Documentation and program still in progress

### Exaple usage:

    fn init() {

    // init new network with amount of (input, output) nodes
        let mut net = NN::new(3, 2);

    // evolve network, mutation are chosen randomly,
    // in future there will be an interface for choosing types and chances
        for _ in 0..32 {
            net.mutate();
        }
    
    // forward inputs through net, returns outputs
        let outputs = net.forward(&[0.5, 0.2, 0.8]);
        println!("\nOrder: \n{:?}", net.layer_order);
        println!("\nConnections: \n{:?}", net.connections);
        println!("\nNodes: \n{:?}", net.nodes);

    }
