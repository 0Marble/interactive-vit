export class Port {
	/**
	 *
	 * @param {string} channel 
	 * @param {Node} node 
	 * @param {"in"|"out"} dir 
	 * @this {Port}
	 */
	constructor(node, channel, dir) {
		this.node = node;
		this.channel = channel;
		this.dir = dir;
	}
}

export class Edge {
	static counter = 0;
	/**
	 *
	 * @param {Port} in_port 
	 * @param {Port} out_port 
	 */
	constructor(in_port, out_port) {
		Edge.counter += 1;

		/**
		 * @type {number}
		 */
		this.index = Edge.counter;
		console.debug(`connect(${in_port.node.index}@${in_port.channel}, ${out_port.node.index}@${out_port.channel}) -> edge_${this.index}`);
		this.in_port = in_port;
		this.out_port = out_port;
	}

	/**
	 * Removes the edge, possibly triggering a calculation downstream
	 * @fires Node#on_upstream_change
	 */
	disconnect() {
		if (!Context.can_edit()) return;

		console.debug(`disconnect(${this.index})`);
		this.in_port.node.outs.get(this.in_port.channel).delete(this);
		this.out_port.node.ins.get(this.out_port.channel).delete(this);
		this.out_port.node.on_upstream_change();
	}
}

/**
 *
 * @implements {Iterable<Edge>}
 */
class EdgeIter {
	/**
	 * @param {Node} node 
	 * @param {"in" | "out"} dir 
	 * @param {string | undefined} channel 
	 */
	constructor(node, dir, channel) {
		let map = node.ins;
		if (dir === "out") {
			map = node.outs;
		}
		if (channel) {
			this.kind = "single_channel";
			this.iter = map.get(channel).values();
		} else {
			this.kind = "multi_channel";
			this.iter1 = map.values();
			const cur_set = this.iter1.next().value;
			if (!cur_set) {
				this.iter2 = [].values();
			} else {
				this.iter2 = cur_set.values();
			}
		}
	}

	/**
	 *@returns {{done: boolean, value: Edge | undefined}}
	 */
	next() {
		if (this.kind === "single_channel") {
			return this.iter.next();
		}
		while (true) {
			const cur = this.iter2.next().value;
			if (!cur) {
				const next_set = this.iter1.next().value;
				if (!next_set) {
					return { done: true };
				}
				this.iter2 = next_set.values();
			} else {
				return { done: false, value: cur };
			}
		}
	}
	[Symbol.iterator]() {
		return this;
	}
}

export class NodeFunction {
	/**
	 * @virtual
	 * @returns {Iterable<string>}
	 */
	in_channel_names() {
		return [];
	}

	/**
	 * @virtual
	 * @returns {Iterable<string>}
	 */
	out_channel_names() {
		return [];
	}

	/**
	 * @virtual
	 */
	on_upstream_change() { }

	/**
	 * @virtual
	 * @returns {boolean}
	 */
	eval() {
		return false;
	}

	/**
	 * @virtual
	 * @returns {boolean}
	 */
	verify_io() {
		return true;
	}

	/**
	 * Returns `eval`'ed packet, `undefined` if couldn't `eval`
	 * @param {string} channel 
	 * @returns {undefined | any }
	 */
	read_packet(_channel) {
		return undefined;
	}
}

export class Context {
	static instance = new Context();

	constructor() {
		this.edit_lock = 0;
		this.eval_lock = false;
		/**
		 * @type {Set<Node>}
		 */
		this.eval_later = new Set();
	}

	static can_edit() {
		return Context.instance.edit_lock === 0;
	}
	static acquire_edit_lock() {
		Context.instance.edit_lock++;
	}
	static release_edit_lock() {
		Context.instance.edit_lock--;
		if (Context.instance.edit_lock < 0) {
			Context.instance.edit_lock = 0;
			console.warn("edit_lock went below 0");
		}
		if (Context.instance.edit_lock === 0) {
			console.debug("edit_lock fully released");
		}
	}

	static lock_eval() {
		Context.instance.eval_lock = true;
	}
	static unlock_eval() {
		Context.instance.eval_lock = false;
		for (const node of Context.instance.eval_later.values()) {
			node.on_upstream_change();
		}
		Context.instance.eval_later.clear();
	}
}

export class Node {
	static counter = 0;
	/**
	 * @type {Set<Node>}
	 */
	static all_nodes = new Set();

	/**
	 * @param {NodeFunction} impl 
	 */
	constructor(impl) {
		Node.counter += 1;
		/**
		 *@type {number}
		 */
		this.index = Node.counter;
		/**
		 *@type {Map<string, Set<Edge>>}
		 */
		this.ins = new Map();
		/**
		 *@type {Map<string, Set<Edge>>}
		 */
		this.outs = new Map();

		this.impl = impl;

		for (const channel of this.impl.in_channel_names()) {
			this.ins.set(channel, new Set());
		}
		for (const channel of this.impl.out_channel_names()) {
			this.outs.set(channel, new Set());
		}

		Node.all_nodes.add(this);
	}

	/**
	 *
	 * @param {string | undefined} port 
	 * @returns {Iterable<Edge>}
	 */
	inputs(port) {
		return new EdgeIter(this, "in", port);
	}

	/**
	 * @returns {Iterable<string>}
	 */
	in_channel_names() {
		return this.impl.in_channel_names();
	}

	/**
	 * @returns {Iterable<string>}
	 */
	out_channel_names() {
		return this.impl.out_channel_names();
	}

	/**
	 *
	 * @param {string | undefined} port 
	 * @returns {Iterable<Edge>}
	 */
	outputs(port) {
		return new EdgeIter(this, "out", port);
	}

	/**
	 * Destroys this node, disconnects it from every parent/child.
	 * Causes a `on_upstream_change` on all children of `this` node
	 * @fires Node#on_upstream_change
	 */
	destroy() {
		if (!Context.can_edit()) return;

		Node.all_nodes.delete(this);

		for (const edge of this.inputs()) {
			edge.in_port.node.outs.get(edge.in_port.channel).delete(edge);
		}

		/**
		 * @type {Set<Node>}
		 */
		const recalc = new Set();
		for (const edge of this.outputs()) {
			edge.out_port.node.ins.get(edge.out_port.channel).delete(edge);
			recalc.add(edge.out_port.node);
		}

		for (const n of recalc) {
			n.on_upstream_change();
		}
	}

	/**
	 * Main way to trigger a calculation.
	 * Notifies `this` node something has changed in the upstream.
	 * `this` node is a root of a sub-DAG of the current expression
	 * 1. Call `on_upstream_change()` on all `impl`s in the sub-DAG (used to invalidate caches)
	 * 2. Find leaf nodes, try to calculate them.
	 * 
	 * Calculation is handled by the `impl.eval()` method.
	 * The method should:
	 * 1. Do any neccessary checks to see if the configuration is correct
	 * 2. Cache its' results to avoid needless recalculations
	 * 3. Recursively call `eval` on `impl`'s parents
	 */
	on_upstream_change() {
		if (Context.instance.eval_lock) {
			Context.instance.eval_later.add(this);
			return;
		}

		console.debug(`on_upstream_change(${this.index})`);
		/**
		 * @type {Set<Node>}
		 */
		const leaves = new Set();
		Node.dfa(this, (n) => {
			if (!n.outputs().next().value) {
				leaves.add(n);
			}
			n.impl.on_upstream_change();
			return true;
		});

		for (const node of leaves) {
			node.impl.eval();
		}
	}

	on_this_changed() {
		console.debug(`on_this_changed(${this.index})`);
		for (const edge of this.outputs()) {
			edge.out_port.node.on_upstream_change();
		}
	}

	/**
	 * Connect nodes, causes calculation in the `out_port.node` sub-DAG.
	 * Returns `true` if the connection actually happened.
	 * 1. Loops are illegal
	 * 2. `in_port.dir` should be `"out"` 
	 * 3. `out_port.dir` should be `"in"` 
	 * 4. An edge is added but then instantly removed if any of the node's `impl`s 
	 *		fail a `impl.verify_io()` custom check.
	 *
	 * @param {Port} in_port 
	 * @param {Port} out_port 
	 * @fires Node#on_upstream_change
	 */
	static connect(in_port, out_port) {
		if (!Context.can_edit()) return;

		if (in_port.node == out_port.node || in_port.dir !== "out" || out_port.dir !== "in") {
			return undefined;
		}

		console.debug(`Connecting: ${in_port.node.index}@${in_port.channel} -> ${out_port.node.index}@${out_port.channel}...`);
		const connected = new Set();
		Node.dfa(out_port.node, (n) => {
			connected.add(n); return true;
		});
		if (connected.has(in_port.node)) {
			console.warn("Loop detected, cancelled connection");
			return undefined;
		}

		const edge = new Edge(in_port, out_port);
		in_port.node.outs.get(in_port.channel).add(edge);
		out_port.node.ins.get(out_port.channel).add(edge);

		if (!in_port.node.impl.verify_io() || !out_port.node.impl.verify_io()) {
			console.warn("New edge failed IO verification, cancelled connection");
			edge.disconnect();
			return undefined;
		}
		console.debug("Connection successfull");

		out_port.node.on_upstream_change();

		return edge;
	}

	/**
	 * @param {Node} node 
	 */
	static dfa(node, visitor) {
		const stack = [node];
		const visited = new Set(stack);

		while (stack.length > 0) {
			const cur = stack.pop();
			if (!visitor(cur)) {
				continue;
			}

			for (const egde of cur.outputs()) {
				const next = egde.out_port.node;
				if (!visited.has(next)) {
					visited.add(next);
					stack.push(next);
				}
			}
		}
	}
}

