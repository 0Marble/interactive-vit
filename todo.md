# Critical

- keep track of sizes on the client
- ability to save/share graphs
    1. A (super-) user designs a graph locally, then it is possible to save and run it on the server.
        - How can we design a very expensive graph locally?
    2. It should also be possible to design a graph non-interactively (json file), so the system operators can create expensive networks.
    3. Gradual implementation: it is much easier to make a server-side node than a client-side node, since client side nodes need custom WebGPU kernels, and on the server we may just use pytorch.

- models: implement a model using server-side nodes
    1. VGG16: looks simple, will get me familiar with the concepts in the field

# Non-critical

- server-side graph: whole sections of the computation graph can be evaluated on the server, without data going back and forth

# Log

- 26.11.2025: 
    1. async eval: the `eval()` function should be async, since it relies on `fetch()` and other async functions.
    Done, but the implementation still feels raw. The issue is that IO events (image loaded, user input, ...) can come at any time, how to ensure that the async stuff is in the correct state?
