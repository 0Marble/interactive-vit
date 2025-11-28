import { CallbackPromise } from "./promise.js";

const graph_div = document.getElementById("graph_div");

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

	button() {
		const id = `node_${this.node.index}_${this.dir}_${this.channel}`;
		return document.getElementById(id);
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
			console.warn("new edge cancelled: must connect out -> in");
			return null;
		}

		if (!Node.dfa(out_port.node, (node) => node !== in_port.node)) {
			console.warn("new edge cancelled: loop detected");
			return null;
		}

		const e = new Edge(in_port, out_port);
		console.debug(`new ${e}: ${in_port} -> ${out_port}`);

		Context.schedule_eval(out_port.node);

		return e;
	}

	disconnect() {
		console.debug(`${this}.disconnect()`);
		Context.ensure_not_eval();

		this.in_port.node.outs.get(this.in_port.channel).delete(this);
		this.out_port.node.ins.get(this.out_port.channel).delete(this);

		if (this.hitbox_line) this.hitbox_line.remove();
		if (this.edge_line) this.edge_line.remove();

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

		this.edge_line = null;
		this.hitbox_line = null;

		this.draw_edge();
	}

	draw_edge() {
		if (this.edge_line) this.edge_line.remove();
		if (this.hitbox_line) this.hitbox_line.remove();

		const edges_svg = document.getElementById("edges_svg");
		const edges_rect = edges_svg.getBoundingClientRect();
		const in_rect = this.in_port.button().getBoundingClientRect();
		const out_rect = this.out_port.button().getBoundingClientRect();

		const x1 = in_rect.left + in_rect.width / 2 - edges_rect.left;
		const y1 = in_rect.top + in_rect.height / 2 - edges_rect.top;
		const x2 = out_rect.left + out_rect.width / 2 - edges_rect.left;
		const y2 = out_rect.top + out_rect.height / 2 - edges_rect.top;


		this.edge_line = document.createElementNS("http://www.w3.org/2000/svg", "line");
		this.edge_line.setAttribute("x1", x1);
		this.edge_line.setAttribute("y1", y1);
		this.edge_line.setAttribute("x2", x2);
		this.edge_line.setAttribute("y2", y2);
		this.edge_line.setAttribute("stroke", "black");
		this.edge_line.setAttribute("stroke-width", 2);

		this.hitbox_line = document.createElementNS("http://www.w3.org/2000/svg", "line");
		this.hitbox_line.setAttribute("x1", x1);
		this.hitbox_line.setAttribute("y1", y1);
		this.hitbox_line.setAttribute("x2", x2);
		this.hitbox_line.setAttribute("y2", y2);
		this.hitbox_line.setAttribute("stroke", "red");
		this.hitbox_line.setAttribute("stroke-width", 5);
		this.hitbox_line.setAttribute("stroke-opacity", "0%");

		let is_mouseover = false;
		this.hitbox_line.addEventListener("mouseenter", () => {
			is_mouseover = true;
			this.hitbox_line.setAttribute("stroke-opacity", "100%");
		});
		this.hitbox_line.addEventListener("mouseleave", () => {
			is_mouseover = false;
			this.hitbox_line.setAttribute("stroke-opacity", "0%");
		});
		this.hitbox_line.addEventListener("click", async () => {
			if (is_mouseover) {
				await Context.wait_for_not_in_eval();

				this.disconnect();

				await Context.do_eval();
				is_mouseover = false;
			}
		});

		edges_svg.appendChild(this.edge_line);
		edges_svg.appendChild(this.hitbox_line);
	}

	toString() {
		return `Edge(${this.index})`;
	}

	async read_packet() {
		return this.in_port.node.get(this.in_port.channel);
	}
}

export class Pinout {
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
	/* ========= IMPLEMENTOR METHODS START ========= */

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

	serialize() {
		throw new Error(`${this}.serialize(): unimplemented`);
	}

	draw_content() { }

	/* ========== IMPLEMENTOR METHODS END ========== */

	constructor() { }

	pre_init() {
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

		this.pos = { x: 0, y: 0 };
		this.div = document.createElement("div");
		this.div.className = "graph_node";
		this.div.id = `node_${this.index}`;
		this.content_div = document.createElement("div");
		this.content_div.className = "node_content";
	}

	post_init() {
		for (const ch of this.input_names()) {
			this.ins.set(ch, new Set());
		}
		for (const ch of this.output_names()) {
			this.outs.set(ch, new Set());
		}

		this.div.appendChild(this.init_header());
		this.div.appendChild(this.init_content());
		this.div.appendChild(this.init_footer());
		this.init_drag();
		graph_div.appendChild(this.div);

		console.debug(`new ${this}`);
	}

	init_drag() {
		this.div.style = `top: ${this.pos.y}px; left: ${this.pos.x}px;`;
		let delta = { x: 0, y: 0 };
		this.div.addEventListener("dragstart", (event) => {
			delta.x = this.pos.x - event.x;
			delta.y = this.pos.y - event.y;
			event.dataTransfer.effectAllowed = "move";
		});
		this.div.draggable = true;
		this.div.addEventListener("dragend", (event) => {
			this.move_to(event.x + delta.x, event.y + delta.y);
		});
	}

	init_header() {
		const header = document.createElement("div");
		header.className = "node_header";

		const left = document.createElement("div");
		left.className = "node_header_left";
		left.appendChild(this.init_port_group("in", this.input_names()));
		header.appendChild(left);

		const right = document.createElement("div");
		right.className = "node_header_right";
		const remove_button = document.createElement("button");
		remove_button.textContent = "x";
		remove_button.addEventListener("click", async () => {
			await Context.wait_for_not_in_eval();
			this.destroy();
			await Context.do_eval();
		});
		right.appendChild(remove_button);
		header.appendChild(right);

		return header;
	}

	init_content() {
		this.draw_content();
		return this.content_div;
	}

	init_footer() {
		const footer = document.createElement("div");
		footer.className = "node_footer";

		const right = document.createElement("div");
		right.className = "node_footer_right";
		right.appendChild(this.init_port_group("out", this.output_names()));
		footer.appendChild(right);

		return footer;
	}

	static current_connect_port = null;
	/**
	 * @param {"in"|"out"} dir 
	 * @param {Iterable<string>} channels 
	 */
	init_port_group(dir, channels) {
		const div = document.createElement("div");

		for (const ch of channels) {
			const button = document.createElement("button");
			button.id = `node_${this.index}_${dir}_${ch}`;
			button.textContent = ch;
			button.className = "non_selected_port";
			div.appendChild(button);

			button.addEventListener("click", async () => {
				await Context.wait_for_not_in_eval();

				const this_port = new Port(this, dir, ch);
				if (Node.current_connect_port === null) {
					Node.current_connect_port = this_port;
					this_port.button().className = "selected_port";
				} else {
					const a = Node.current_connect_port;
					const b = this_port;
					let edge = null;
					if (a.dir === "in" && b.dir === "out") {
						edge = Edge.connect(b, a);
					} else {
						edge = Edge.connect(a, b);
					}
					a.button().className = "non_selected_port";
					Node.current_connect_port = null;

					await Context.do_eval();
				}
			});
		}

		return div;
	}

	/**
	 * @param {number} x 
	 * @param {number} y 
	 */
	move_to(x, y) {
		this.pos = { x, y };
		this.div.style = `top: ${this.pos.y}px; left: ${this.pos.x}px;`;

		this.on_visual_update();
	}

	on_visual_update() {
		for (const e of this.inputs()) e.draw_edge();
		for (const e of this.outputs()) e.draw_edge();
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
		if (Context.nodes_to_eval.has(this)) {
			Context.nodes_to_eval.delete(this);
		}

		this.div.remove();
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
		const set = this.ins.get(channel);
		return set.values().next().value || null
	}

	/**
	 * @param {string} channel 
	 */
	async get(channel) {
		const pinout = await this.eval_internal();
		if (!pinout) return null;
		return pinout.get(channel);
	}

	/**
	 * Dont use this!
	 * @returns {Promise<Pinout|null>}
	 */
	async eval_internal() {
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
	static eval_signal = new CallbackPromise(true);

	static start_eval() {
		Context.is_eval = true;
		Context.eval_state = new CallbackPromise();
	}

	static end_eval() {
		Context.is_eval = false;
		Context.eval_signal.trigger();
	}

	static async wait_for_not_in_eval() {
		await Context.eval_signal;
	}

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
	 * @type {Set<Node>}
	 */
	static nodes_to_eval = new Set();

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

		Context.start_eval();

		/**
		 * @type {Map<Node,Promise<{node:Node,res:(Pinout|null)}>>}
		 */
		const worklist = new Map();
		for (const node of Context.nodes_to_eval) {
			worklist.set(node, node.eval_internal().then(res => { return { node, res } }));
		}
		Context.nodes_to_eval.clear();

		while (worklist.size > 0) {
			const { node, res } = await Promise.race(worklist.values());
			worklist.delete(node);

			if (!res) continue;

			for (const e of node.outputs()) {
				const node = e.out_port.node;

				if (!worklist.has(node)) {
					worklist.set(node, node.eval_internal().then(res => { return { node, res } }));
				}
			}
		}

		Context.end_eval();
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

