import * as dataflow from "./dataflow.js";

const graph_div = document.getElementById("graph_div");

class Edge {
	static counter = 0;
	/**
	 * @param {Port} output 
	 * @param {Port} input 
	 * @param {dataflow.Edge} e 
	 */
	constructor(input, output, e) {
		Edge.counter += 1;
		this.index = Edge.counter;
		this.input = input;
		this.output = output;
		this.e = e;
	}

	render() {
		if (this.edge) this.edge.remove();
		if (this.hitbox) this.hitbox.remove();

		console.log(`Edge.render(${this.index})`);
		const edges = document.getElementById("edges_svg");
		const container = edges.getBoundingClientRect();
		const src = this.input.button.getBoundingClientRect();
		const dst = this.output.button.getBoundingClientRect();

		const x1 = src.left + src.width / 2 - container.left;
		const y1 = src.top + src.height / 2 - container.top;
		const x2 = dst.left + dst.width / 2 - container.left;
		const y2 = dst.top + dst.height / 2 - container.top;

		this.edge = document.createElementNS("http://www.w3.org/2000/svg", "line");
		this.edge.id = `edge_${this.index}`
		this.edge.setAttribute("x1", x1);
		this.edge.setAttribute("y1", y1);
		this.edge.setAttribute("x2", x2);
		this.edge.setAttribute("y2", y2);
		this.edge.setAttribute("stroke", "black");
		this.edge.setAttribute("stroke-width", "2");

		this.hitbox = document.createElementNS("http://www.w3.org/2000/svg", "line");
		this.hitbox.setAttribute("x1", x1);
		this.hitbox.setAttribute("y1", y1);
		this.hitbox.setAttribute("x2", x2);
		this.hitbox.setAttribute("y2", y2);
		this.hitbox.setAttribute("stroke-width", "5");
		this.hitbox.setAttribute("stroke", "red");
		this.hitbox.setAttribute("stroke-opacity", "0%");
		let is_mouseover = false;
		this.hitbox.addEventListener("mouseover", () => {
			this.hitbox.setAttribute("stroke-opacity", "100%");
			this.input.button.className = "selected_port";
			this.output.button.className = "selected_port";
			is_mouseover = true;
		});
		this.hitbox.addEventListener("mouseleave", () => {
			is_mouseover = false;
			this.hitbox.setAttribute("stroke-opacity", "0%");
			this.input.button.removeAttribute("class");
			this.output.button.removeAttribute("class");
		})
		this.hitbox.addEventListener("click", () => {
			if (is_mouseover) {
				this.remove();
				is_mouseover = false;
			}
		});

		edges.appendChild(this.edge);
		edges.appendChild(this.hitbox);
	}

	remove() {
		if (!dataflow.Context.can_edit()) return;

		this.edge.remove();
		this.hitbox.remove();

		this.e.disconnect();
		this.input.node.outs.get(this.input.name).delete(this);
		this.output.node.ins.get(this.output.name).delete(this);
		this.input.button.removeAttribute("class");
		this.output.button.removeAttribute("class");
	}
}

export class Port {

	/**
	 * 
	 * @param {Node} node 
	 * @param {string} name 
	 * @param {"in"|"out"} kind 
	 */
	constructor(node, name, kind) {
		this.node = node;
		this.name = name;
		this.kind = kind;

		this.button = document.createElement("button");
		this.button.id = `port_${node.index}_${name}_${kind}`;
		this.button.textContent = name;
	}

	/**
	 * @returns {dataflow.Port}
	 */
	to_dataflow() {
		return new dataflow.Port(this.node.n, this.name, this.kind);
	}
}

export class Node {
	static counter = 0;
	/**
	 * @type {Set<Node>}
	 */
	static all_nodes = new Set();
	static current_selected_port = null;

	/**
	 * @param {dataflow.Node} n 
	 */
	constructor(n) {
		Node.counter += 1;
		Node.all_nodes.add(this);

		this.index = Node.counter;
		this.n = n;

		this.div = document.createElement("div");
		this.div.className = "graph_node";
		this.x = 0;
		this.y = 0;

		/**
		 * @type {Map<string, Set<Edge>>}
		 */
		this.ins = new Map();
		/**
		 * @type {Map<string, Set<Edge>>}
		 */
		this.outs = new Map();

		this.div.appendChild(this.init_header());
		this.content_div = document.createElement("div");
		this.div.appendChild(this.content_div);
		this.content_div.className = "node_content";
		this.div.appendChild(this.init_footer());
		this.init_drag();

		graph_div.appendChild(this.div);
	}

	init_drag() {
		this.div.style = `top: ${this.y}; left: ${this.x};`;
		let delta = { x: 0, y: 0 };
		this.div.addEventListener("dragstart", (event) => {
			delta.x = this.x - event.x;
			delta.y = this.y - event.y;
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
		const left_group = document.createElement("div");
		left_group.className = "node_header_left";
		const right_group = document.createElement("div");
		right_group.className = "node_header_right";
		header.appendChild(left_group);
		header.appendChild(right_group);

		left_group.appendChild(this.init_port_group("in", this.n.in_channel_names()));
		const label = document.createElement("span");
		label.textContent = `Node ${this.index}`;
		left_group.appendChild(label);

		const remove = document.createElement("button");
		remove.textContent = "x";
		remove.addEventListener("click", () => {
			if (!dataflow.Context.can_edit()) return;

			for (const port of this.ins.values()) {
				for (const edge of port) edge.remove();
			}
			for (const port of this.outs.values()) {
				for (const edge of port) edge.remove();
			}
			this.div.remove();
			Node.all_nodes.delete(this);

			dataflow.Context.acquire_edit_lock();
			this.n.destroy().then(() => dataflow.Context.release_edit_lock());
		});

		right_group.appendChild(remove);

		return header;
	}

	init_footer() {
		const footer = document.createElement("div");
		footer.className = "node_footer";
		const left_group = document.createElement("div");
		left_group.className = "node_footer_left";
		const right_group = document.createElement("div");
		right_group.className = "node_footer_right";
		footer.appendChild(left_group);
		footer.appendChild(right_group);

		right_group.appendChild(this.init_port_group("out", this.n.out_channel_names()));

		return footer;
	}

	/**
	 * @param {"in" | "out"} kind 
	 * @param {Iterable<string>} names 
	 */
	init_port_group(kind, names) {
		const ports_div = document.createElement("div");
		for (const name of names) {
			const info = new Port(this, name, kind);
			ports_div.appendChild(info.button);

			info.button.addEventListener("click", () => {
				const other = Node.current_selected_port;
				if (Node.current_selected_port === null) {
					Node.current_selected_port = info;
					info.button.className = "selected_port";
				} else if (Node.current_selected_port.node === info.node) {
					other.button.removeAttribute("class");
					Node.current_selected_port = null;
				} else if (Node.current_selected_port.kind !== info.kind) {
					other.button.removeAttribute("class");
					Node.current_selected_port = null;
					if (info.kind == "in") {
						Node.connect(other, info);
					} else {
						Node.connect(info, other);
					}
					other.button.removeAttribute("class");
				} else {
					// dont connect 
				}
			});

		}
		return ports_div;
	}

	/**
	 *
	 * @param {number} x 
	 * @param {number} y 
	 */
	move_to(x, y) {
		this.x = x;
		this.y = y;

		this.render();
	}

	render() {
		console.debug(`render(Node.${this.index})`);
		this.div.style = `top: ${this.y}px; left: ${this.x}px;`;

		for (const port of this.ins.values()) {
			for (const edge of port.values()) {
				edge.render();
			}
		}
		for (const port of this.outs.values()) {
			for (const edge of port.values()) {
				edge.render();
			}
		}
	}

	/**
	 * @param {Port} input 
	 * @param {Port} output 
	 */
	static async connect(input, output) {
		if (!dataflow.Context.can_edit()) return;

		const df_edge = await dataflow.Node.connect(input.to_dataflow(), output.to_dataflow());
		if (!df_edge) {
			return;
		}
		const edge = new Edge(input, output, df_edge);

		if (!input.node.outs.has(input.name)) input.node.outs.set(input.name, new Set());
		input.node.outs.get(input.name).add(edge);

		if (!output.node.ins.has(output.name)) output.node.ins.set(output.name, new Set());
		output.node.ins.get(output.name).add(edge);

		input.node.render();
		output.node.render();
	}
}
