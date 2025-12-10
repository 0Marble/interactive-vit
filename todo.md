# Critical

- Zeros node: create a new empty tensor of given shape.
- Stack node: `out = [a, b, c, d, ...]`. This is tricky, because we only support a pre-defined number of ports.
    1. "Fake" ports: create ports within the content div of the node. Hacky...
    2. Proper dynamic ports support: sounds possible, but there is a large surface for unforseen issues
- Multi-view node: view all slices of a CHW tensor

# Non-critical

- backend Node class: make it easier to implement nodes on the backend, by abstracting away all the http stuff.
- `net_node` scripting: is it possible to send over logic together with `net_node` contents? 
- keep track of sizes on the client
- server-side graph: whole sections of the computation graph can be evaluated on the server, without data going back and forth
- Admin panel: ability to upload custom client-desgined graphs

# Log

- 10.12.2025:
    1. Worked on frontend design. I am bad at it lol. Now the main workspace is draggable, and displays a grid to find yourself more easily. 
    I also removed the ugly toolbar at the top, replacing it with an ugly hamburger menu in the corner.

- 08.12.2025:
    1. Node picker: create a node at the pointer location.

- 03.12.2025:
    1. Shuffle node: to reorder dimensions
    2. Fix remote file access in `load_model`. Before I used to just use python `open` function to read the file located at `BASE_DIR/static/models/{name}/graph.json` (name is a query parameter) and send this file. This would enable people to do something like `fetch("load_model?name=../../secrets")`. Now I use django's `django.views.static.serve` which has a `parent_dir` parameter, and the function does not allow to get files outside of this dir.
    3. Implemented a VGG16 model. It is actually pretty simple, since pytorch allows you to run sub-modules of a network separately, by name. I create a `class Model` in the model dir, and then call its interface functions to do computation. 
    However, it is quite difficult to tell what is going on. I get a (for example) [512, 30, 30] tensor in the middle, and half the outputs are just black, the rest just have a vague outline of the original image.

- 02.12.2025:
    1. Slice node: `out = in[:, 10, 20]`. In addition, the img source node now outputs a CHW shaped tensor.

- 29.11.2025:
    1. Support tensor strides+offsets

- 28.11.2025:
    1. Save/Share graphs part 1: save/load on the client
    2. Ability to load graphs from the server

- 27.11.2025:
    1. Experimented with different node APIs. 
    Doing a "true" dynamic re-evaluation seems too messy: we need a lot of complicated (=> buggy) synchronization (in a singlethreaded environment...). 
    A much simpler solution is to separate scheduling for evaluation and running the evaluation. We can later just disable all inputs if we are in evaluation mode (spawn a semi-transparent non-clicktrough modal to cover the whole screen)
    Should graph operations be async and wait for eval to end, or do we just assert? 
    Becomes complicated since constructors can not be async. 
    In addition, what if there is a whole queue of actions? 
    It seems the most robust approach is to disallow all input during eval, and emit hard errors if some input does get through.

- 26.11.2025: 
    1. async eval: the `eval()` function should be async, since it relies on `fetch()` and other async functions.
    Done, but the implementation still feels raw. The issue is that IO events (image loaded, user input, ...) can come at any time, how to ensure that the async stuff is in the correct state?
