import * as gpu from "./gpu.js";
import { CallbackPromise } from "./promise.js";

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

	/**
	 * @returns {string}
	 */
	format() {
		return `${this.node.format()}@${this.dir}_${this.channel}`;
	}
}

export class Edge {
	static counter = 0;
	/**
	 * @param {Port} in_port 
	 * @param {Port} out_port 
	 * Dont call directly
	 */
	constructor(in_port, out_port) {
		Edge.counter += 1;

		/**
		 * @type {number}
		 */
		this.index = Edge.counter;
		this.in_port = in_port;
		this.out_port = out_port;
		/**
		 * @type {gpu.Tensor | null}
		 */
		this.packet = null;

		console.debug(`new ${this.format()}: ${in_port.format()} -> ${out_port.format()}`);
	}

	/**
	 * @returns {string}
	 */
	format() {
		return `Edge(${this.index})`;
	}

	/**
	 * Removes the edge. Does not trigger re-evaluation
	 */
	disconnect() {
		if (!Context.can_edit()) return;

		console.debug(`${this.format()}: disconnect()`);
		this.in_port.node.outs.get(this.in_port.channel).delete(this);
		this.out_port.node.ins.get(this.out_port.channel).delete(this);

		this.out_port.node.invalidate_with_descendants();
	}

	/**
	 * Main way to get packets from the system
	 * If there is no packet, attempts to re-eval
	 * @returns {Promise<gpu.Tensor | null>}
	 */
	async read_packet() {
		if (!this.packet) await this.in_port.node.eval();
		return this.packet;
	}

	clear_packet() {
		this.packet = null;
	}

	/**
	 * @param {gpu.Tensor} packet 
	 */
	write_packet(packet) {
		this.packet = packet;
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
		this.to_update = new Set();

		/**
		 * @type {CallbackPromise[]}
		 */
		this.eval_promises = [];
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
	static async ensure_can_eval() {
		if (!Context.instance.eval_lock) {
			return await Promise.resolve();
		}

		const promise = new CallbackPromise();
		Context.instance.eval_promises.push(promise);
		await promise;
	}
	static unlock_eval() {
		Context.instance.eval_lock = false;
		for (const p of Context.instance.eval_promises) {
			p.trigger();
		}
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

export class Node {
	static counter = 0;
	/**
	 * @type {Set<Node>}
	 */
	static all_nodes = new Set();

	constructor() {
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

		for (const channel of this.in_channel_names()) {
			this.ins.set(channel, new Set());
		}
		for (const channel of this.out_channel_names()) {
			this.outs.set(channel, new Set());
		}

		Node.all_nodes.add(this);
	}

	//// ================ CUSTOM IMPLEMENTATION START ================ 

	/**
	 * @returns {Iterable<string>}
	 */
	in_channel_names() {
		return [];
	}

	/**
	 * @returns {Iterable<string>}
	 */
	out_channel_names() {
		return [];
	}

	/*
	 * Main evaluation entry point
	 * Returns false if couldnt evaluate
	 * Puts all result packets into the corresponding edges
	 * using `Node.emit_result`.
	 * if eval is disabled by the Context this does not get called.
	 */
	async eval_impl() {
		return false;
	}

	/**
	 * Called when anything has changed upstream
	 * Intended to clear caches 
	 */
	invalidate_impl() { }

	/**
	 * @param {string} _channel 
	 * @param {"in"|"out"} _dir 
	 * Verifies if all edges connected to this port are correct
	 * (size ok, ...)
	 */
	verify(_dir, _channel) {
		return true;
	}

	/**
	 * returns an object to be turned into a JSON string later
	 */
	serialize() {
		return {};
	}

	//// ================ CUSTOM IMPLEMENTATION END ================ 

	/**
	 * @param {string} channel 
	 * @param {gpu.Tensor} packet 
	 */
	emit_result(channel, packet) {
		for (const edge of this.outputs(channel)) {
			edge.write_packet(packet);
		}
	}

	/**
	 *
	 * @param {string | undefined} channel 
	 * @returns {Iterable<Edge>}
	 */
	inputs(channel) {
		return new EdgeIter(this, "in", channel);
	}

	/**
	 * @param {string | undefined} port 
	 * @returns {Iterable<Edge>}
	 */
	outputs(channel) {
		return new EdgeIter(this, "out", channel);
	}

	/**
	 * @returns {string}
	 */
	format() {
		return `Node(${this.index})`;
	}

	/**
	 * Calls `invalidate_impl()` and clears packets on output edges
	 */
	invalidate() {
		console.debug(`${this.format()}: invalidate()`);
		this.invalidate_impl();
		for (const edge of this.outputs()) edge.clear_packet();
	}

	/**
	 * @returns {Promise<boolean>}
	 */
	async eval() {
		console.debug(`${this.format()}: eval: started`);
		await Context.ensure_can_eval();
		const res = await this.eval_impl();
		console.debug(`${this.format()}: eval: ${res}`);
		return res;
	}

	/**
	 * @param {Port} in_port 
	 * @param {Port} out_port 
	 * @returns {Promise<Edge | null>}
	 *
	 * Connect two ports, and trigger evaluation if the connection is accepted by `Node.verify`
	 * Only DAG's are allowed
	 * Returns the new edge if everything is ok
	 */
	static async connect(in_port, out_port) {
		const e = new Edge(in_port, out_port);

		in_port.node.outs.get(in_port.channel).add(e);
		out_port.node.ins.get(out_port.channel).add(e);

		/**
		 * @type {Set<Node>}
		 */
		const reachable = new Set();
		const no_loop = Node.dfa(out_port.node, node => {
			if (reachable.has(node)) return false;
			reachable.add(node);
			return true;
		});

		if (!no_loop) {
			console.warn(`illegal connection ${e.format()}: only DAGs allowed`);
			in_port.node.outs.get(in_port.channel).delete(e);
			out_port.node.ins.get(out_port.channel).delete(e);
			return null;
		}

		if (!in_port.node.verify("out", in_port.channel) || !out_port.node.verify("in", out_port.channel)) {
			console.warn(`illegal connection ${e.format()}: Node.verify()`);
			in_port.node.outs.get(in_port.channel).delete(e);
			out_port.node.ins.get(out_port.channel).delete(e);
			return null;
		}

		await in_port.node.eval_with_descendants();

		return e;
	}

	async destroy() {
		/**
		 * @type {Set<Node>}
		 */
		console.debug(`${this.format()}: destroy()`);
		const to_eval = new Set();
		for (const edge of this.outputs()) {
			to_eval.add(edge.out_port.node);
			edge.disconnect();
		}

		const promises = [];
		for (const node of to_eval) promises.push(node.eval_with_descendants());
		await Promise.all(promises);
	}

	/**
	 * Causes a recursive invalidation
	 */
	invalidate_with_descendants() {
		Node.dfa(this, (node) => { node.invalidate(); return true; });
	}

	/**
	 * Causes a recursive evaluation of all nodes in `to_eval` 
	 * as well as their descendants
	 * @param {Iterable<Node>} to_eval 
	 */
	static async eval_all_with_descendants(to_eval) {
		await Context.ensure_can_eval();

		/**
		 * @type {Map<number, Promise<{node:Node,status:boolean}>>}
		 */
		const work_list = new Map();
		for (const node of to_eval) {
			work_list.set(node.index, node.eval().then(ok => {
				return { node, status: ok }
			}));
		}

		while (work_list.size != 0) {
			const { node, status } = await Promise.race(work_list.values());
			work_list.delete(node.index);
			if (!status) continue;

			for (const edge of node.outputs()) {
				const next = edge.out_port.node;
				if (!work_list.has(next.index)) {
					work_list.set(next.index, next.eval().then(ok => {
						return { node: next, status: ok }
					}));
				}
			}
		}
	}

	async eval_with_descendants() {
		await Node.eval_all_with_descendants([this]);
	}

	async eval_only_descendants() {
		const nodes = new Set();
		for (const edge of this.outputs()) nodes.add(edge.out_port.node);
		await Node.eval_all_with_descendants(nodes);
	}

	/**
	 * @param {Node} start 
	 * @param {(node:Node)=>boolean} visitor 
	 * @returns {boolean}
	 */
	static dfa(start, visitor) {
		const visited = new Set([start]);
		const stack = [start];
		while (stack.length != 0) {
			const node = stack.pop();

			if (!visitor(node)) return false;

			for (const edge of node.outputs()) {
				const next = edge.out_port.node;
				if (!visited.has(next)) {
					visited.add(next);
					stack.push(next);
				}
			}
		}

		return true;
	}
}

