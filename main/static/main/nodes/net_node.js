
import * as graph from "../graph.js";
import * as gpu from "../gpu.js";
import * as csrf from "../csrf.js";
import { Workspace } from "../workspace.js";


class Request {
	/**
	 * @param {Set<NetworkNode>} nodes 
	 */
	constructor(nodes) {
		this.nodes = nodes;
		/**
		 * @type {Map<NetworkNode, number>}
		 */
		this.mapping = new Map();
	}

	/**
	 * @returns {Promise<Response>}
	 */
	async process() {
		const buf = await this.encode();
		const resp = await fetch("compute", {
			body: buf,
			method: "POST",
			headers: {
				'Content-Type': 'application/octet-stream',
				'X-CSRFToken': csrf.get_csrf_token(),
			}
		});
		if (!resp.ok) {
			throw new Error(await resp.text())
		}

		return new Response(this.mapping, (await resp.bytes()).buffer);
	}

	/**
	 * @returns {Promise<ArrayBuffer>}
	 * [ header | utf-8 json | data block 0 | ... ]
	 * header: 
	 *   - byte size: u32
	 *   - magic: u32
	 *   - block cnt: u32
	 *   - json block byte size: u32
	 * json: {
	 *   nodes: [{endpoint: string, params: obj}],
	 *   edges: [{
	 *      // (tensor xor node) present
	 *      in_port: {?node: number, ?tensor: number, channel: string},  
	 *      out_port: {node: number, channel: string},
	 *   }],
	 * }
	 * data block: (same as Response.decode)
	 *   - byte size: u32
	 *   - dim cnt: u32,
	 *   - dims: [u32],
	 *   - data: [f32],
	 */
	async encode() { }
}

class Response {
	/**
	 * @param {Map<NetworkNode, number>} mapping 
	 * @param {ArrayBuffer} buf 
	 */
	constructor(mapping, buf) {
		this.mapping = mapping;
		this.decode(buf);
	}

	/**
	 * @param {NetworkNode} node 
	 * @param {string} channel 
	 * @returns {gpu.Tensor}
	 */
	get_output(node, channel) { }

	/**
	 * [header | json | data block 0 | ...]
	 * header:
	 *   - byte size: u32,
	 *   - magic: u32,
	 *   - data block count: u32,
	 *   - json block byte size: u32,
	 * json: {[{node: number, channel: number}]} // element with idx i is tensor in block i
	 * data block: (same as Request.encode)
	 *   - byte size: u32
	 *   - dim cnt: u32,
	 *   - dims: [u32],
	 *   - data: [f32],
	 */
	decode() { }
}

class Context {
	constructor() {
		/**
		 * @type {Map<NetworkNode, Promise<graph.Pinout>>}
		 */
		this.pending = new Map();
	}

	/**
	 * @param {NetworkNode} node
	 */
	ensure_pending(node) {
		if (this.pending.has(node)) return;

		const nodes = Context.subgraph(node);
		const req = new Request(nodes);
		const pending = req.process();

		for (const node of nodes) {
			console.assert(!this.pending.has(node));
			this.pending.set(node, Context.eval_impl(node, pending));
		}
	}

	/**
	 * @param {NetworkNode} node 
	 * @param {Promise<Response>} pending 
	 */
	static async eval_impl(node, pending) {
		/**
		 * @type {Response}
		 */
		const response = await pending;
		const res = new graph.Pinout();
		for (const e of node.outputs()) {
			const tensor = response.get_output(node, e.in_port.channel);
			res.set(e.in_port.channel, tensor);
		}
		return res;
	}

	/**
	 * @param {NetworkNode} start 
	 * @returns {Set<NetworkNode>}
	 */
	static subgraph(start) {
		const res = new Set();
		res.add(start);
		const stack = [start];

		while (stack.length != 0) {
			const node = stack.pop();
			for (const e of node.inputs()) {
				const next = e.in_port.node;
				if (!(next instanceof NetworkNode) || res.has(next)) continue;
				res.add(next);
				stack.push(next);
			}
			for (const e of node.outputs()) {
				const next = e.out_port.node;
				if (!(next instanceof NetworkNode) || res.has(next)) continue;
				res.add(next);
				stack.push(next);
			}
		}

		return res;
	}
}

const context = new Context();

export class NetworkNode extends graph.Node {
	/**
	 * @param {string} endpoint 
	 * @param {IODescription} io 
	 */
	constructor(endpoint, io, params_obj) {
		super();
		this.pre_init();

		this.endpoint = endpoint;
		this.net_div = document.createElement("div");
		/**
		 * @type {IODescription}
		 */
		this.io = io;
		this.params_obj = params_obj;

		this.post_init();
	}

	async fetch_node() {
		while (this.net_div.firstChild) this.net_div.firstChild.remove();

		this.net_div.innerHTML = "<p>Loading...</p>"

		try {
			let url = `node/${this.endpoint}/contents`;
			if (this.params_obj) {
				url = url + "?" + new URLSearchParams(this.params_obj).toString();
			}

			const resp = await fetch(url, { method: "GET" });
			if (!resp.ok) throw new Error("response not ok");
			this.net_div.innerHTML = await resp.text();
			this.on_visual_update();
		} catch (err) {
			console.warn("Invalid IO description response:", err);
			this.init_retry();
		}
	}

	init_retry() {
		while (this.net_div.firstChild) this.net_div.firstChild.remove();
		const button = document.createElement("button");
		button.textContent = "Retry"
		button.addEventListener("click", async () => await this.fetch_node());
		this.net_div.appendChild(button);
	}

	draw_content() {
		this.content_div.appendChild(this.net_div);
	}

	/**
	 * @virtual
	 * @returns {Iterable<string>}
	 */
	input_names() {
		return this.io.in_names();
	}

	/**
	 * @virtual
	 * @returns {Iterable<string>}
	 */
	output_names() {
		return this.io.out_names();
	}
	async eval() {
		context.ensure_pending(this);
		const res = await context.pending.get(this);
		context.pending.delete(this);
		return res;
	}

	static async register_factory() {
		graph.Context.register_deserializer("net_node", NetworkNode.deserialize);

		Workspace.register_tool("Cos*", async (x, y) => {
			const node = await NetworkNode.create("cos", { b: 1.0 });
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.width * 0.5, y - rect.height * 0.5);
		});
	}

	static async create(endpoint, params_obj) {
		await graph.Context.wait_for_not_in_eval();

		let url = `node/${endpoint}/description`;
		if (params_obj) {
			url = url + "?" + new URLSearchParams(params_obj).toString();
		}

		const resp = await fetch(url, { method: "GET" });
		const json = await resp.json();
		const io = new IODescription(json);

		const node = new NetworkNode(endpoint, io, params_obj);
		await node.fetch_node();
		return node;
	}

	serialize() {
		return {
			kind: "net_node",
			endpoint: this.endpoint,
			params: this.params_obj,
		};
	}

	static async deserialize(obj) {
		return await NetworkNode.create(obj.endpoint, obj.params);
	}
}

class IODescription {
	constructor(json) {
		this.ins = json.ins;
		this.outs = json.outs;
	}

	in_names() {
		return this.ins;
	}

	out_names() {
		return this.outs;
	}
}
