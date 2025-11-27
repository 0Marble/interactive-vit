import * as gpu from "./gpu.js";
import { CallbackPromise } from "./promise.js";

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
	 * @param {Port} in_port 
	 * @param {Port} out_port 
	 * @returns {Edge | null}
	 */
	static connect(in_port, out_port) {
		console.debug(`Edge.connect(${in_port}, ${out_port})`);
		Context.ensure_not_eval();

		if (in_port.dir !== "out" || out_port.dir !== "in") {
			console.warn("new edge: must connect out -> in");
			return null;
		}

		if (!Node.dfa(out_port.node, (node) => node !== in_port.node)) {
			console.warn("new edge: loop detected");
			return null;
		}

		const e = new Edge(in_port, out_port);
		console.debug(`new ${e}: ${in_port} -> ${out_port}`);

		Context.schedule_eval(in_port.node);

		return e;
	}

	disconnect() {
		console.debug(`${this}.disconnect()`);
		Context.ensure_not_eval();

		this.in_port.node.outs.get(this.in_port.channel).delete(this);
		this.out_port.node.ins.get(this.out_port.channel).delete(this);

		Context.schedule_eval(this.out_port.node);
		Context.all_edges.delete(this);
	}

	/**
	 * @param {Port} in_port 
	 * @param {Port} out_port 
	 */
	constructor(in_port, out_port) {
		this.in_port = in_port;
		this.out_port = out_port;
		this.index = Context.new_index();

		Context.all_edges.add(this);

		in_port.node.outs.get(in_port.channel).add(this);
		out_port.node.ins.get(out_port.channel).add(this);
	}

	toString() {
		return `Edge(${this.index})`;
	}
}

class Pinout {
	constructor() {
		this.values = new Map();
	}
	/**
	 * @param {string} channel 
	 */
	get(channel) {
		return this.values.get(channel);
	}
	/**
	 * @param {string} channel 
	 */
	set(channel, value) {
		this.values.set(channel, value);
	}
	clear() {
		this.values.clear();
	}
}

export class Node {
	/* ========= IMPLEMENTORD METHODS START ========= */

	/**
	 * @returns{Iterable<string>}
	 */
	input_names() {
		return [];
	}

	/**
	 * @returns{Iterable<string>}
	 */
	output_names() {
		return [];
	}

	/**
	 * @returns {Promise<Pinout | null>}
	 */
	async eval() {
		throw new Error(`${this}.eval(): unimplemented`);
	}

	/* ========== IMPLEMENTORD METHODS END ========== */

	constructor() {
		Context.ensure_not_eval();

		this.index = Context.new_index();
		Context.all_nodes.add(this);
		Context.schedule_eval(this);

		/**
		 * Do not use this!
		 * @type{null|Promise<Pinout|null>}
		 */
		this.eval_state = null;

		/**
		 * @type {Map<string, Set<Edge>>}
		 */
		this.ins = new Map();
		/**
		 * @type {Map<string, Set<Edge>>}
		 */
		this.outs = new Map();

		for (const ch of this.input_names()) {
			this.ins.set(ch, new Set());
		}
		for (const ch of this.output_names()) {
			this.outs.set(ch, new Set());
		}

		console.debug(`new ${this}`);
	}

	destroy() {
		console.debug(`${this}.destroy()`);
		Context.ensure_not_eval();

		for (const e of this.outputs()) {
			e.disconnect();
		}
		for (const e of this.inputs()) {
			e.disconnect();
		}

		Context.all_nodes.delete(this);

		console.debug(`${this}.destroy()`);
	}

	/**
	 * @param {string|undefined} channel 
	 * @returns {Iterable<Edge>}
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
	 * @param {string|undefined} channel 
	 * @returns {Iterable<Edge>}
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
		return this.inputs(channel).next().value;
	}

	/**
	 * @param {string} channel 
	 */
	async get(channel) {
		const pinout = await this.do_eval();
		if (!pinout) return null;
		return pinout.get(channel);
	}

	/**
	 * Dont use this!
	 * @returns {Promise<Pinout|null>}
	 */
	async do_eval() {
		Context.ensure_in_eval();
		if (this.eval_state === null) {
			console.debug(`${this}.eval()`);
			this.eval_state = this.eval();
		}
		return await this.eval_state;
	}

	invalidate() {
		console.debug(`${this}.invalidate()`);
		Context.ensure_not_eval();
		this.eval_state = null;
	}
	invalidate_with_descendants() {
		Node.dfa(this, (node) => { node.invalidate(); return true; });
	}

	toString() {
		return `Node(${this.index})`;
	}

	/**
	 * @param {Node} start 
	 * @param {(node:Node)=>boolean} visitor 
	 * @returns {boolean}
	 */
	static dfa(start, visitor) {
		const stack = [start];
		const visited = new Set([start]);

		while (stack.length > 0) {
			const node = stack.pop();
			if (!visitor(node)) return false;
			for (const e of node.outputs()) {
				const next = e.out_port.node;
				if (visited.has(next)) continue;
				visited.add(next);
				stack.push(next);
			}
		}
		return true;
	}
}

export class Context {
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
		Context.counter += 1;
		return Context.counter;
	}

	static is_eval = false;

	/**
	 * @type {Set<Node>}
	 */
	static nodes_to_eval = new Set();

	static ensure_not_eval() {
		if (Context.is_eval) {
			throw new Error("Context.ensure_not_eval() failed!");
		}
	}

	static ensure_in_eval() {
		if (!Context.is_eval) {
			throw new Error("Context.ensure_in_eval() failed!");
		}
	}

	/**
	 * @param {Node} node 
	 */
	static schedule_eval(node) {
		Context.ensure_not_eval();
		Context.nodes_to_eval.add(node)
	}

	static async do_eval() {
		console.debug("Context.do_eval(): start");
		for (const node of Context.nodes_to_eval) node.invalidate_with_descendants();

		Context.is_eval = true;

		/**
		 * @type {Map<Node,Promise<{node:Node,res:(Pinout|null)}>>}
		 */
		const worklist = new Map();
		for (const node of Context.nodes_to_eval) {
			worklist.set(node, node.do_eval().then(res => { return { node, res } }));
		}

		while (worklist.size > 0) {
			const { node, res } = await Promise.race(worklist.values());
			worklist.delete(node);

			if (!res) continue;

			for (const e of node.outputs()) {
				const node = e.out_port.node;

				if (!worklist.has(node)) {
					worklist.set(node, node.do_eval().then(res => { return { node, res } }));
				}
			}
		}

		Context.is_eval = false;
		console.debug("Context.do_eval(): end");
	}

	static clear() {
		console.debug("Context.clear()");
		Context.ensure_not_eval();

		Context.counter = 0;

		for (const n of Context.all_nodes) {
			n.destroy();
		}

		Context.nodes_to_eval.clear();
	}
}

class SrcNode extends Node {
	/**
	 * @returns{Iterable<string>}
	 */
	input_names() {
		return [];
	}

	/**
	 * @returns{Iterable<string>}
	 */
	output_names() {
		return ["o"];
	}

	/**
	 * @returns {Promise<Pinout | null>}
	 */
	async eval() {
		if (this.value) {
			const pinout = new Pinout();
			pinout.set("o", this.value);
			return pinout;
		} else return null;
	}

	constructor() {
		super();
		this.value = null;
	}

	set_value(value) {
		this.value = value;
		Context.schedule_eval(this);
	}
}

class DstNode extends Node {
	/**
	 * @returns{Iterable<string>}
	 */
	input_names() {
		return ["o"];
	}

	/**
	 * @returns{Iterable<string>}
	 */
	output_names() {
		return [];
	}

	/**
	 * @returns {Promise<Pinout | null>}
	 */
	async eval() {
		this.value = null;
		const e = this.single_input("o");
		if (!e) return null;
		this.value = await e.in_port.node.get(e.in_port.channel);
		return null;
	}

	constructor() {
		super();
		this.value = null;
	}
}

export async function test() {
	const tests = [
		{ fn: test_basic_usage, name: "basic" }
	];

	const max_timeout = 10;
	const timeout = async (test_state) => {
		const promise = new CallbackPromise();
		setTimeout(() => test_state.ok || promise.trigger(), max_timeout * 1000);
		await promise;
		throw new Error("testing max_timeout reached");
	};

	let passed = 0;
	for (const { name, fn } of tests) {
		try {
			const test_state = { ok: false };
			await Promise.race([fn().then(() => test_state.ok = true), timeout(test_state)]);
			passed++;
			Context.clear();
		} catch (e) {
			console.error(`[graph::${name}]: failed:`, e);
		}
	}

	if (passed === tests.length) {
		console.error(`[graph]: all passed!`);
	}
}

async function test_basic_usage() {
	const a = new SrcNode();
	const b = new DstNode();
	Edge.connect(new Port(a, "out", "o"), new Port(b, "in", "o"));

	await Context.do_eval();
	console.assert(b.value === null);

	a.set_value(69);
	await Context.do_eval();
	console.assert(b.value === 69);
}
