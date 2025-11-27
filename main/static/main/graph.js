import * as gpu from "./gpu.js";
import { AsyncLock, CallbackPromise } from "./promise.js";

export class Port {
	/**
	 * @param {Node} node 
	 * @param {"in"|"out"} dir 
	 * @param {string} channel 
	 */
	constructor(node, dir, channel) {
		this.node = node;
		this.dir = dir;
		this.channel = channel;
	}

	toString() {
		return `${this.node}@${this.dir}_${this.channel}`;
	}
}

export class Edge {

	/**
	 * @private
	 * @param {Port} in_port 
	 * @param {Port} out_port 
	 */
	constructor(in_port, out_port) {
		this.index = Context.new_index();
		Context.all_edges.add(this);

		this.in_port = in_port;
		this.out_port = out_port;

	}

	/**
	 * @param {Port} in_port 
	 * @param {Port} out_port 
	 * @returns {Promise<Edge|null>}
	 */
	static async connect(in_port, out_port) {
		await Context.start_no_eval_section("Edge.connect");
		if (in_port.dir !== "out" || out_port.dir !== "in") {
			console.warn(`Could not connect ${in_port} -> ${out_port}: must connect out to in`);
			Context.end_no_eval_section("Edge.connect");
			return null;
		}

		if (!Node.dfa(out_port.node, (node) => node !== in_port.node)) {
			console.warn(`Could not connect ${in_port} -> ${out_port}: loop detected`);
			Context.end_no_eval_section("Edge.connect");
			return null;
		}

		const e = new Edge(in_port, out_port);
		in_port.node.outs.get(in_port.channel).add(e);
		out_port.node.ins.get(out_port.channel).add(e);

		Context.end_no_eval_section("Edge.connect");

		await e.out_node().invalidate_with_descendants();
		await e.in_node().eval_with_descendants();

		console.debug(`new ${e}: ${in_port} -> ${out_port}`);

		return e;
	}

	async disconnect() {
		await Context.start_no_eval_section("Edge.disconnect");
		console.debug(`${this}.disconnect()`);

		this.in_node().outs.get(this.in_channel()).delete(this);
		this.out_node().ins.get(this.out_channel()).delete(this);
		Context.all_edges.delete(this);

		Context.end_no_eval_section("Edge.disconnect");

		await this.out_node().invalidate_with_descendants();
		await this.out_node().eval_with_descendants();
	}

	/**
	 * @returns {Node}
	 */
	out_node() {
		return this.out_port.node;
	}

	/**
	 * @returns {Node}
	 */
	in_node() {
		return this.in_port.node;
	}

	/**
	 * @returns {string}
	 */
	in_channel() {
		return this.in_port.channel;
	}
	/**
	 * @returns {string}
	 */
	out_channel() {
		return this.out_port.channel;
	}

	toString() {
		return `Edge(${this.index})`;
	}
}

class Pinout {

	constructor() {
		/**
		 * @type {Map<string, gpu.Tensor>}
		 */
		this.values = new Map();
	}

	clear() {
		this.values.clear();
	}

	/**
	 * @param {string} channel 
	 * @returns {gpu.Tensor}
	 */
	get(channel) {
		return this.values.get(channel);
	}

	/**
	 * @param {string} channel 
	 * @param {gpu.Tensor} value 
	 */
	set(channel, value) {
		this.values.set(channel, value);
	}
}

export class Context {

	static eval_count = 0;
	static eval_lock = new AsyncLock(true);
	/**
	 * @type {"open" | "eval" | "locked"}
	 */
	static eval_state = "open";

	/**
	 * @returns {Promise<void>}
	 */
	static async start_no_eval_section(loc) {
		if (Context.eval_state !== "locked") {
			await Context.eval_lock.acquire();
		}
		Context.eval_count++;
		Context.eval_state = "locked";
		console.debug(`Context.start_no_eval_section (${Context.eval_count}, ${Context.eval_state}, ${loc})`);
	}

	static end_no_eval_section(loc) {
		console.debug(`Context.end_no_eval_section (${Context.eval_count}, ${Context.eval_state}, ${loc})`);
		console.assert(Context.eval_state === "locked");
		console.assert(Context.eval_count > 0);
		Context.eval_count--;

		if (Context.eval_count === 0) {
			Context.eval_lock.release();
			Context.eval_state = "open";
		}
	}

	/**
	 * @returns {Promise<void>}
	 */
	static async start_eval_section(loc) {
		if (Context.eval_state !== "eval") {
			await Context.eval_lock.acquire();
		}
		Context.eval_count++;
		Context.eval_state = "eval";
		console.debug(`Context.start_eval_section (${Context.eval_count}, ${Context.eval_state}, ${loc})`);
	}

	static end_eval_section(loc) {
		console.debug(`Context.end_eval_section (${Context.eval_count}, ${Context.eval_state}, ${loc})`);
		console.assert(Context.eval_state === "eval");
		console.assert(Context.eval_count > 0);
		Context.eval_count--;

		if (Context.eval_count === 0) {
			Context.eval_lock.release();
			Context.eval_state = "open";
		}
	}

	static counter = 0;
	/**
	 * @type {Set<Node>}
	 */
	static all_nodes = new Set();
	/**
	 * @type {Set<Edge>}
	 */
	static all_edges = new Set();

	/**
	 * @returns {number}
	 */
	static new_index() {
		Context.counter++;
		return Context.counter;
	}
}

export class Node {
	// ============== SUB-CLASS METHODS START ============== 

	/**
	 * @returns {Iterable<string>}
	 */
	output_channels() {
		return [];
	}

	/**
	 * @returns {Iterable<string>}
	 */
	input_channels() {
		return [];
	}

	/**
	 * Just implementation, caching is handled by the parent class
	 * @returns {Promise<Pinout | null>}
	 */
	async eval_impl() {
		throw new Error(`${this}.eval_impl(): unimplemented`);
	}

	render_impl() {
		throw new Error(`${this}.render_impl(): unimplemented`);
	}

	serialize() {
		throw new Error(`${this}.serialize(): unimplemented`);
	}

	// ============== SUB-CLASS METHODS END ============== 

	/**
	 * @private
	 */
	constructor() {
		this.index = Context.new_index();
		Context.all_nodes.add(this);

		/**
		 * @type {null | Promise<Pinout | null>}
		 */
		this.eval_sate = null

		/**
		 * @type {Map<string, Set<Edge>>}
		 */
		this.ins = new Map();
		/**
		 * @type {Map<string, Set<Edge>>}
		 */
		this.outs = new Map();

		for (const ch of this.input_channels()) {
			this.ins.set(ch, new Set());
		}
		for (const ch of this.output_channels()) {
			this.outs.set(ch, new Set());
		}

		console.debug(`new ${this}`);
	}

	static async create(typ, ...args) {
		await Context.start_no_eval_section("Node.create");
		const n = new typ(...args);
		Context.end_no_eval_section("Node.create");
		return n;
	}

	async destroy() {
		const promises = [];

		await Context.start_no_eval_section("Node.destroy");

		for (const e of this.outputs()) {
			promises.push(e.disconnect());
		}
		for (const e of this.inputs()) {
			promises.push(e.disconnect());
		}
		Context.all_nodes.delete(this);

		Context.end_no_eval_section("Node.destroy");

		await Promise.all(promises);
		console.debug(`${this}.destroy()`);
	}

	/**
	 * @param {string} channel 
	 * @returns {Promise<gpu.Tensor | null>}
	 */
	async get(channel) {
		const pinout = await this.eval();
		if (!pinout) return null;
		return pinout.get(channel);
	}

	/**
	 * @returns {Iterable<Edge>}
	 * @param {string} channel 
	 */
	outputs(channel) {
		if (channel) return this.outs.get(channel).values();

		const res = [];
		for (const set of this.outs.values()) {
			for (const e of set) {
				res.push(e);
			}
		}
		return res;
	}

	/**
	 * @returns {Iterable<Edge>}
	 * @param {string | undefined} channel 
	 */
	inputs(channel) {
		if (channel) return this.ins.get(channel).values();

		const res = [];
		for (const set of this.ins.values()) {
			for (const e of set) {
				res.push(e);
			}
		}
		return res;
	}

	/**
	 * @param {string} channel 
	 * @returns {Edge | null}
	 */
	single_input(channel) {
		const set = this.ins.get(channel);
		if (set.size !== 1) return null;
		return set.values().next().value;
	}

	/**
	 * @returns {Promise<Pinout | null>}
	 */
	async eval() {
		await Context.start_eval_section("Node.eval");

		if (!this.eval_sate) {
			console.debug(`${this}.eval()`);
			this.eval_sate = this.eval_impl();
		}
		const res = await this.eval_sate;

		Context.end_eval_section("Node.eval");

		return res;
	}

	/**
	 * @param {Iterable<Node>} nodes 
	 */
	static async eval_starting_from(nodes) {
		/**
		 * @type {Map<Node, Promise<{node:Node, ok: boolean}>>}
		 */
		const worklist = new Map();
		for (const node of nodes) {
			worklist.set(node, node.eval().then((ok) => { return { node, ok } }));
		}

		while (worklist.size > 0) {
			const { node, ok } = await Promise.race(worklist.values());
			worklist.delete(node);

			if (!ok) continue;

			for (const e of node.outputs()) {
				const next = e.out_node();
				worklist.set(next, next.eval().then((ok) => { return { node: next, ok } }));
			}
		}
	}

	async eval_with_descendants() {
		await Node.eval_starting_from([this]);
	}

	async invalidate_with_descendants() {
		await Context.start_no_eval_section("Node.invalidate_with_descendants");
		Node.dfa(this, (node) => { node.eval_sate = null; return true; });
		Context.end_no_eval_section("Node.invalidate_with_descendants");
	}


	/**
	 * @param {Node} start 
	 * @param {(node: Node) => boolean} visitor 
	 * @returns {boolean}
	 */
	static dfa(start, visitor) {
		const stack = [start];
		const visited = new Set([start]);

		while (stack.length > 0) {
			const node = stack.pop();
			if (!visitor(node)) return false;
			for (const e of node.outputs()) {
				if (!visited.has(e.out_node())) {
					visited.add(e.out_node());
					stack.push(e.out_node());
				}
			}
		}
		return true;
	}

	toString() {
		return `Node(${this.index})`;
	}
}

class TimerNode extends Node {
	constructor(seconds) {
		super();
		this.seconds = seconds;
	}

	output_channels() {
		return ["o"];
	}

	input_channels() {
		return ["o"];
	}

	async eval_impl() {
		const promise = new CallbackPromise();
		setTimeout(promise.trigger, this.seconds * 1000);
		await promise;

		const pinout = new Pinout();
		const e = this.single_input("o");
		if (!e) {
			return null;
		}
		const o_input = await e.in_node().get(e.in_channel());
		pinout.set("o", o_input);

		return pinout;
	}
}

class SourceNode extends Node {
	constructor() {
		super();
		this.value = null;
	}

	async set_value(value) {
		await Context.start_no_eval_section("SourceNode.set_value");
		this.value = value;
		Context.end_no_eval_section("SourceNode.set_value");

		await this.invalidate_with_descendants();
		await this.eval_with_descendants();
	}

	output_channels() {
		return ["o"];
	}

	input_channels() {
		return [];
	}

	async eval_impl() {
		if (this.value) {
			const pinout = new Pinout();
			pinout.set("o", this.value);
			return pinout;
		} else return null;
	}
}

class DrainNode extends Node {
	constructor() {
		super();
		this.value = null;
	}

	output_channels() {
		return [];
	}

	input_channels() {
		return ["o"];
	}

	async eval_impl() {
		this.value = null;
		const e = this.single_input("o");
		if (!e) return null;
		const value = await e.in_node().get(e.in_channel());
		this.value = value;
		return null;
	}
}

export async function test() {
	const tests = [
		{ name: "test_basic", fn: test_basic },
	];
	const max_timeout = async (obj) => {
		const promise = new CallbackPromise();
		setTimeout(() => obj.ok || promise.trigger(), 2 * 1000);
		await promise;
		throw new Error("max_timeout hit!");
	};

	let passed = 0;
	for (const { name, fn } of tests) {
		try {
			const obj = { ok: false };
			await Promise.race([fn().finally(() => obj.ok = true), max_timeout(obj)]);
			passed++;
		} catch (e) {
			console.error(`[graph::${name}] failed: `, e);
		}
	}

	if (passed === tests.length) {
		console.log("graph: all passed!");
	}
}

async function test_basic() {
	/**
	 * @type {SourceNode}
	 */
	const a = await Node.create(SourceNode);
	const b = await Node.create(DrainNode);

	await Edge.connect(new Port(a, "out", "o"), new Port(b, "in", "o"));

	console.assert(b.value === null);
	await a.set_value(69);
	console.assert(b.value === 69);

	await a.destroy();

	console.assert(b.value === null);
	await b.destroy();
}

