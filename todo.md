# Critical

- async eval: the `eval()` function should be async, since it relies on `fetch()` and other async functions
- models: implement a model using server-side nodes

# Non-critical

- server-side graph: whole sections of the computation graph can be evaluated on the server, without data going back and forth
