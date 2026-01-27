import { Hover } from "./hover.js";
import { CallbackPromise } from "./promise.js";

const nodes_div = document.getElementById("nodes_div");

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

		if (!Node.dfs(out_port.node, (node) => node !== in_port.node)) {
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

		this.in_port.button().className = "non_selected_port";
		this.out_port.button().className = "non_selected_port";

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
			this.in_port.button().className = "selected_port";
			this.out_port.button().className = "selected_port";
		});
		this.hitbox_line.addEventListener("mouseleave", () => {
			is_mouseover = false;
			this.hitbox_line.setAttribute("stroke-opacity", "0%");
			this.in_port.button().className = "non_selected_port";
			this.out_port.button().className = "non_selected_port";
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
	 * Throw for errors
	 * @returns {Promise<Pinout>}
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
		 * @type{null|Promise<Pinout>}
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

		this.status_text = document.createElement("span");
		this.status_text.textContent = "waiting for input";
		this.status_hover = new Hover();
		this.status_hover.attatch(this.status_text);

		this.div.appendChild(this.init_header());
		this.div.appendChild(this.init_content());
		this.div.appendChild(this.init_footer());
		this.init_drag();
		nodes_div.appendChild(this.div);

		/**
		 * @type {Map<string, Hover>}
		 */
		this.port_hovers = new Map();
		for (const ch of this.output_names()) {
			const hover = new Hover();
			this.port_hovers.set(ch, hover);
			const port = new Port(this, "out", ch);
			const button = port.button();
			hover.attatch(button);
		}

		console.debug(`new ${this}`);
	}

	init_drag() {
		this.div.style = `top: ${this.pos.y}px; left: ${this.pos.x}px;`;
		let delta = { x: 0, y: 0 };

		let in_drag = false;
		this.div.addEventListener("mousedown", (event) => {
			if (!in_drag) {
				event.stopPropagation();
				delta.x = this.pos.x - event.x;
				delta.y = this.pos.y - event.y;
				in_drag = true;
			}
		});

		this.div.addEventListener("mousemove", (event) => {
			if (in_drag) {
				event.stopPropagation();
				this.move_to(event.x + delta.x, event.y + delta.y);
			}
		});

		this.div.addEventListener("mouseup", (event) => {
			if (in_drag) {
				event.stopPropagation();
				in_drag = false;
			}
		});
	}

	init_header() {
		const header = document.createElement("div");
		header.className = "node_header";

		const left = document.createElement("div");
		left.className = "node_header_left";
		left.appendChild(this.init_port_group("in", this.input_names()));
		left.append(`Node ${this.index}`);
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
		footer.append(this.status_text, right);

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
	 * @returns {Promise<Pinout | null>}
	 */
	async eval_internal() {
		Context.ensure_in_eval();

		this.status_hover.clear_content();
		if (this.eval_state === null) {
			console.debug(`${this}.eval()`);
			this.eval_state = this.eval();
			this.status_text.textContent = "evaluating...";
			this.status_text.style = "color: black;";
			for (const ch of this.output_names()) {
				const hover = this.port_hovers.get(ch);
				hover.clear_content();
			}
		}

		try {
			const res = await this.eval_state;
			this.status_text.textContent = "Eval ok!";
			this.status_text.style = "color: LawnGreen;";
			for (const ch of this.output_names()) {
				const tensor = res.get(ch);
				const hover = this.port_hovers.get(ch);
				hover.set_content(`[${tensor.dims}]`);
			}
			return res;
		} catch (err) {
			this.status_text.textContent = "Error!";
			this.status_text.style = "color: red;";
			this.status_hover.set_content(err);
			console.error(err);
		}
		return null;
	}

	invalidate() {
		this.status_hover.clear_content();
		this.status_text.textContent = "waiting for input";
		this.status_text.style = "color: black;";

		console.debug(`${this}.invalidate()`);
		Context.ensure_not_eval();
		this.eval_state = null;
	}

	invalidate_with_descendants() {
		Node.dfs(this, (node) => { node.invalidate(); return true; });
	}

	toString() {
		return `Node(${this.index})`;
	}

	serialize_internal() {
		const instance = this.serialize();
		return {
			instance,
			pos: this.pos,
		};
	}

	/**
	 * @param {Node} start 
	 * @param {(node:Node)=>boolean} visitor 
	 * @returns {boolean}
	 */
	static dfs(start, visitor) {
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
		Context.eval_signal = new CallbackPromise();
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
		/**
		 * @param {Node} node
		 */
		const run_work = async (node) => {
			try {
				const res = await node.eval_internal();
				return { node, res };
			} catch (err) {
				node.status_text.textContent = "eval error!";
				node.status_text.style = "color: red;";
				node.status_hover.set_content(err);
				console.error(err);
				return { node, res: null };
			}
		};
		for (const node of Context.nodes_to_eval) {
			worklist.set(node, run_work(node));
		}
		Context.nodes_to_eval.clear();

		while (worklist.size > 0) {
			const { node, res } = await Promise.race(worklist.values());
			worklist.delete(node);

			for (const e of node.outputs()) {
				const node = e.out_port.node;

				if (!worklist.has(node)) {
					worklist.set(node, run_work(node));
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

	/**
	 * @type {Map<string, {(obj:any) => Promise<Node>}
	 */
	static deserializers = new Map();

	/**
	 * @param {string} kind 
	 * @param {(obj:any)=>Promise<Node>} fptr 
	 */
	static register_deserializer(kind, fptr) {
		Context.deserializers.set(kind, fptr);
	}

	static serialize() {
		Context.ensure_not_eval();

		/**
		 * @type {Map<Node,number>}
		 */
		const index_map = new Map();
		const obj = {
			nodes: [],
			edges: [],
		};

		for (const node of Context.all_nodes) {
			const data = node.serialize_internal();
			index_map.set(node, obj.nodes.length);
			obj.nodes.push(data);
		}
		for (const edge of Context.all_edges) {
			const data = {
				in_port: {
					node: index_map.get(edge.in_port.node),
					channel: edge.in_port.channel,
				},
				out_port: {
					node: index_map.get(edge.out_port.node),
					channel: edge.out_port.channel,
				},
			};
			obj.edges.push(data);
		}

		return obj;
	}

	static async deserialize(obj) {
		/**
		 * @type{Promise<Node>[]}
		 */
		const promises = obj.nodes.map(async (data) => {
			const { pos, instance } = data;
			if (!Context.deserializers.has(instance.kind)) {
				console.warn("unknown node type: ", instance);
				return Promise.resolve(null);
			} else {
				const fptr = Context.deserializers.get(instance.kind);
				const node = await fptr(instance);
				node.move_to(pos.x, pos.y);
				return node;
			}
		});

		const nodes = await Promise.all(promises);

		for (const { in_port, out_port } of obj.edges) {
			const a = new Port(nodes[in_port.node], "out", in_port.channel);
			const b = new Port(nodes[out_port.node], "in", out_port.channel);
			Edge.connect(a, b);
		}
	}
}

