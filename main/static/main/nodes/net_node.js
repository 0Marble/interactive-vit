
import * as graph from "../graph.js";
import * as gpu from "../gpu.js";
import * as csrf from "../csrf.js";
import { Workspace } from "../workspace.js";


class TargettedError extends Error {
	/**
	 * @param {NetworkNode} target 
	 * @param {string} message 
	 */
	constructor(target, message) {
		super(message);
		this.target = target;
	}

	toString() {
		return this.message;
	}
}

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
		console.debug(buf);
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
	 * also fills out this.mapping
	 *
	 * [ header | utf-8 json | padding_to_4 | data block 0 | ... ]
	 * header: 
	 *   - byte size: u32
	 *   - magic: u32
	 *   - block cnt: u32
	 *   - json block byte size: u32
	 * json: {
	 *   nodes: [{endpoint: string, params: obj}],
	 *   edges: [{
	 *      // (tensor xor in_port) present
	 *      ?tensor: number,
	 *      ?in_port: {node: number, channel: string},  
	 *      out_port: {node: number, channel: string},
	 *   }],
	 * }
	 * data block: (same as Response.decode)
	 *   - byte size: u32
	 *   - dim cnt: u32,
	 *   - dims: [u32],
	 *   - data: [f32],
	 */
	async encode() {
		const obj = { nodes: [], edges: [] };
		/**
		 * @type {[graph.Edge]}
		 */
		const input_edges = [];

		for (const node of this.nodes) {
			this.mapping.set(node, obj.nodes.length);

			obj.nodes.push({
				endpoint: node.endpoint,
				params: node.params_obj,
			});
		}

		for (const node of this.nodes) {
			for (const ch of node.input_names()) {
				let has_input = false;
				for (const e of node.inputs(ch)) {
					if (has_input) throw new TargettedError(
						node,
						`too many inputs on ${ch}, only 1 expected`,
					);
					has_input = true;

					const prev = e.in_port.node;
					if (prev instanceof NetworkNode) {
						obj.edges.push({
							in_port: {
								node: this.mapping.get(prev),
								channel: e.in_port.channel,
							},
							out_port: {
								node: this.mapping.get(node),
								channel: e.out_port.channel,
							},
						});
					} else {
						obj.edges.push({
							tensor: input_edges.length,
							out_port: {
								node: this.mapping.get(node),
								channel: e.out_port.channel,
							},
						});
						input_edges.push(e);
					}
				}

				if (!has_input) throw new TargettedError(node, `missing input ${ch}`);
			}
		}

		const tensors = await Promise.all(input_edges.map(async (e) => {
			/**
			 * @type {gpu.Tensor}
			 */
			const tensor = await e.read_packet();
			if (!tensor) throw new TargettedError(
				e.out_port.node,
				`could not compute ${e.out_port.channel}`,
			);
			return Request.encode_tensor(tensor.contiguous());
		}));

		const json = new TextEncoder().encode(JSON.stringify(obj));
		let byte_size = json.length + 4 * 4;
		const padding = align_next(byte_size, 4) - byte_size;
		byte_size += padding;
		let tensor_size = 0;
		for (const t of tensors) tensor_size += t.byteLength;
		byte_size += tensor_size;

		const buf = new ArrayBuffer(byte_size);
		const view = new DataView(buf);
		view.setUint32(0, byte_size, true);
		view.setUint32(4, 0x69babe69, true);
		view.setUint32(8, tensors.length, true);
		view.setUint32(12, json.length, true);

		new Uint8Array(buf, 16, json.length).set(json);

		let byte_offset = 16 + json.length + padding;
		for (const t of tensors) {
			new Uint8Array(buf, byte_offset, t.byteLength).set(new Uint8Array(t));
			byte_offset += t.byteLength;
		}

		console.debug("request:");
		console.debug(`size=${byte_size} (16 | ${json.length} | ${padding} | ${tensor_size})`);
		console.debug("json:", obj);

		return buf;
	}

	/**
	 * @param {gpu.Tensor} tensor 
	 * @returns {Promise<ArrayBuffer>}
	 * - byte size: u32
	 * - dim cnt: u32,
	 * - dims: [u32],
	 * - data: [f32],
	 */
	static async encode_tensor(tensor) {
		const data = await tensor.to_cpu();
		const dims = new Uint32Array(tensor.dims)
		const size = 4 + 4 + dims.byteLength + data.byteLength;
		const buf = new ArrayBuffer(size);
		const view = new DataView(buf);
		view.setUint32(0, size, true);
		view.setUint32(4, tensor.dims.length, true);
		new Uint8Array(buf, 8, dims.byteLength).set(new Uint8Array(dims.buffer));
		new Uint8Array(buf, 8 + dims.byteLength, data.byteLength).set(new Uint8Array(data));

		return buf;
	}
}

function align_next(offset, align) {
	const m = offset % align;
	if (m === 0) return offset;
	return offset + align - m;
}

class Response {
	/**
	 * @param {Map<NetworkNode, number>} mapping 
	 * @param {ArrayBuffer} buf 
	 */
	constructor(mapping, buf) {
		this.mapping = mapping;
		/**
		 * @type {[Map<string, gpu.Tensor>]}
		 */
		this.outputs = [];
		for (const _ of this.mapping) this.outputs.push(new Map());

		this.decode(buf);
	}

	/**
	 * @param {NetworkNode} node 
	 * @param {string} channel 
	 * @returns {gpu.Tensor}
	 */
	get_output(node, channel) {
		const idx = this.mapping.get(node);
		if (idx === undefined) throw new TargettedError(node, "no such node");
		const res = this.outputs[idx].get(channel)
		if (res === undefined) throw new TargettedError(node, `could not compute ${channel}`);
		return res;
	}

	/**
	 * [header | json | data block 0 | ...]
	 * header:
	 *   - byte size: u32,
	 *   - magic: u32 = 0xdeadbeef,
	 *   - data block count: u32,
	 *   - json block byte size: u32,
	 * json: {[{node: number, channel: string}]} // element with idx i is tensor in block i
	 * data block: (same as Request.encode)
	 *   - byte size: u32
	 *   - dim cnt: u32,
	 *   - dims: [u32],
	 *   - data: [f32],
	 *
	 * @param {ArrayBuffer} buf
	 */
	decode(buf) {
		const view = new DataView(buf)
		let offset = 0;
		const byte_size = view.getUint32((offset += 4) - 4, true);
		const magic = view.getUint32((offset += 4) - 4, true);
		if (magic != 0xdeadbeef) throw new Error("invalid response magic");
		const data_cnt = view.getUint32((offset += 4) - 4, true);
		const json_size = view.getUint32((offset += 4) - 4, true);

		const json_utf8 = new Uint8Array(buf, offset, json_size);
		offset += json_size;
		const json_str = new TextDecoder().decode(json_utf8);
		/**
		 * @type {[{node: number, channel: string}]}
		 */
		const arr = JSON.parse(json_str);
		const padding = align_next(offset, 4) - offset;
		offset += padding;

		console.debug("response:");
		console.debug(`size=${byte_size} (16 | ${data_cnt} | ${json_size} | ${padding})`)
		console.debug("json:", arr);

		for (let i = 0; i < data_cnt; i++) {
			const start = offset;
			const block_size = view.getUint32((offset += 4) - 4, true);
			const dim_cnt = view.getUint32((offset += 4) - 4, true);
			const dims = new Uint32Array(buf, offset, dim_cnt)
			offset += dim_cnt * 4;

			let elem_cnt = 1;
			for (const x of dims) elem_cnt *= x;
			const data = new Float32Array(buf, offset, elem_cnt);
			offset += elem_cnt * 4;
			console.debug(`tensor[${i}]: start=${start}, byte_size=${block_size}, dims=${dims}`);

			const tensor = gpu.Tensor.from_dims_and_data(4, dims, data);

			const { node, channel } = arr[i];
			this.outputs[node].set(channel, tensor);

			console.assert(block_size + start === offset);
		}
		console.assert(byte_size === offset);

		console.debug("outputs: ", this.outputs);
	}
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
			if (this.pending.has(node)) {
				console.error("node", node, " is already pending! this should be unreachable");
			}
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
		for (const ch of node.output_names()) {
			const tensor = response.get_output(node, ch);
			res.set(ch, tensor);
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
		let eval_error = null;
		let res = null;
		try {
			res = await context.pending.get(this);
		} catch (e) {
			if (e instanceof TargettedError && e.target === this) {
				eval_error = e;
			} else {
				eval_error = new Error("eval error upstream");
			}
		}
		context.pending.delete(this);

		if (eval_error !== null) throw eval_error;
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
