
import * as graph from "./graph.js";
import * as gpu from "./gpu.js";
import * as csrf from "./csrf.js";

/*
 * This node will do opaque calculations on the server
 *
 * APIs:
 * /{endpoint}/description - get IO description
 * /{endpoint}/contents    - get displayed html
 * /{endpoint}/compute     - run eval
 *
 */

class IOPort {
	/**
	 * @param {string} channel 
	 * @param {"in"|"out"} kind 
	 * @param {"1"|"1+"|"*"} access 
	 */
	constructor(kind, channel, access) {
		this.kind = kind;
		this.channel = channel;
		this.access = access;
	}

	static parse(raw_obj) {
		if (typeof raw_obj !== "object") throw new TypeError("IOPort should be an object");

		if (!raw_obj.kind) throw new TypeError("IOPort should have a field 'kind'")
		const kind = raw_obj.kind;
		if (kind !== "in" && kind !== "out") throw new TypeError("IOPort.kind should be \"in\" or \"out\"");

		if (!raw_obj.channel) throw new TypeError("IOPort should have a field 'channel'");
		const channel = raw_obj.channel;
		if (typeof channel !== "string") throw new TypeError("IOPort.channel should be a string");

		if (!raw_obj.access) throw new TypeError("IOPort should have a field 'access'")
		const access = raw_obj.access;
		if (access !== "1" && access !== "1+" && access != "*") throw new TypeError("IOPort.access should be \"1\" or \"1+\" or \"*\"");

		return new IOPort(kind, channel, access);
	}
}

class IODescription {
	/**
	 *
	 * @param {IOPort[]} ports 
	 */
	constructor(ports) {
		this.ports = ports;
	}

	in_names() {
		const res = [];
		for (const port of this.ports) {
			if (port.kind === "in") res.push(port.channel);
		}
		return res;
	}

	out_names() {
		const res = [];
		for (const port of this.ports) {
			if (port.kind === "out") res.push(port.channel);
		}
		return res;
	}

	/**
	 * @param {"in"|"out"} kind 
	 * @param {string} name 
	 * @param {number} count 
	 */
	channel_access_valid(kind, name, count) {
		for (const port of this.ports) {
			if (port.kind === kind && port.channel === name) {
				switch (port.access) {
					case "1": return count === 1;
					case "1+": return count >= 1;
					case "*": return true;
				}
			}
		}
		return false;
	}

	static parse(raw_obj) {
		if (typeof raw_obj !== "object") throw new TypeError("IODescription should be an IOPort[]");
		const ports = [];
		for (const raw_port of raw_obj) ports.push(IOPort.parse(raw_port));
		return new IODescription(ports);
	}

}

class Message {
	constructor() {
		/**
		 * @type {{tensor:gpu.Tensor, channel:string}[]}
		 */
		this.all_tensors = [];
	}

	/**
	 * @param {string} channel 
	 * @param {gpu.Tensor} tensor 
	 */
	add_tensor(channel, tensor) {
		this.all_tensors.push({ tensor, channel });
	}

	/**
	 * @returns {number}
	 */
	get_tensor_count() {
		return this.all_tensors.length;
	}

	/**
	 * @param {number} n 
	 * @returns {gpu.Tensor}
	 */
	get_nth_tensor(n) {
		return this.all_tensors[n].tensor;
	}

	/**
	 * @param {number} n 
	 * @returns {string}
	 */
	get_nth_channel(n) {
		return this.all_tensors[n].channel;
	}

	/**
	 * @param {string} endpoint 
	 */
	async send(endpoint) {
		const buf = await this.encode();

		const resp = await fetch(`${endpoint}/compute`, {
			method: "POST",
			body: buf,
			headers: {
				'Content-Type': 'application/octet-stream',
				'X-CSRFToken': csrf.get_csrf_token(),
			}
		});
		if (!resp.ok) {
			throw new Error(await resp.text())
		}

		const result = await resp.bytes()
		this.all_tensors = [];
		await this.decode(result.buffer);
	}

	/**
	 * @param {ArrayBuffer} buffer 
	 */
	async decode(buffer) {
		const view = new DataView(buffer);
		let offset = 0;

		const num_packets = view.getUint32(offset);
		offset += 4;
		this.all_tensors = [];

		for (let i = 0; i < num_packets; i++) {
			const channel_len = view.getUint32(offset);
			offset += 4;
			const str = new TextDecoder().decode(new Uint8Array(buffer, offset, channel_len));
			offset += channel_len;

			offset = add_padding(offset, 4);
			const dim_cnt = view.getUint32(offset);
			offset += 4;
			const dims_array = new Uint32Array(buffer, offset, dim_cnt);
			const dims = []
			let elem_cnt = 1;
			for (const d of dims_array) {
				dims.push(d);
				elem_cnt *= d;
			}
			offset += 4 * dim_cnt;
			const data = new Float32Array(buffer, offset, elem_cnt);
			offset += 4 * elem_cnt;

			const tensor = new gpu.Tensor(dims, 4);
			tensor.from_cpu(data);
			this.all_tensors.push({ channel: str, tensor });

		}
	}

	async encode() {
		// [num_packets | packet1 | packet2 .... ]
		// [ channel_len (u32, utf8) | channel (utf8) | n_dims (u32) | dims (u32) | data (f32) ]

		let byte_size = 4;
		const offsets = [];
		const promises = []
		for (let i = 0; i < this.all_tensors.length; i++) {
			offsets.push(byte_size);
			const t = this.all_tensors[i];

			const enc = new TextEncoder();
			byte_size += 4;
			byte_size += enc.encode(t.channel).length;
			byte_size = add_padding(byte_size, 4);
			byte_size += 4;
			byte_size += 4 * t.tensor.dims.length;
			byte_size += 4 * t.tensor.elem_cnt;
			promises.push(t.tensor.to_cpu().then(buf => { return { buf, i }; }));
		}
		offsets.push(byte_size);

		const buffer = new ArrayBuffer(byte_size);
		const view = new DataView(buffer);
		const buffers = await Promise.all(promises);
		view.setUint32(0, this.all_tensors.length);

		for (const t of buffers) {
			let offset = offsets[t.i];
			const dims = this.all_tensors[t.i].tensor.dims;
			const channel = this.all_tensors[t.i].channel;
			const elem_cnt = this.all_tensors[t.i].tensor.elem_cnt;

			const enc = new TextEncoder().encode(channel);
			view.setUint32(offset, enc.length);
			offset += 4;
			new Uint8Array(buffer, offset, enc.length).set(enc);
			offset += enc.length;

			offset = add_padding(offset, 4);
			view.setUint32(offset, dims.length);
			offset += 4;
			new Uint32Array(buffer, offset, dims.length).set(dims);

			offset += 4 * dims.length;
			new Float32Array(buffer, offset, elem_cnt).set(new Float32Array(t.buf));
			offset += 4 * elem_cnt;

			console.assert(offset === offsets[t.i + 1]);
		}

		return buffer;
	}
}

function add_padding(offset, align) {
	const m = offset % align;
	if (m === 0) return offset;
	return offset + align - m;
}

export class NetworkNode extends graph.Node {
	static async register_factory(endpoint) {
		if (!NetworkNode.io_cache.has(endpoint)) {
			const resp = await fetch(`${endpoint}/description`, { method: "GET" });
			const json = await resp.json();
			const io = IODescription.parse(json);
			NetworkNode.io_cache.set(endpoint, io);
		}
		const io = NetworkNode.io_cache.get(endpoint);

		const node_button = document.createElement("button");
		node_button.textContent = `New Net-${endpoint} Node`;
		node_button.addEventListener("click", async () => {
			const node = new NetworkNode(endpoint, io);
			await node.fetch_node();
		});

		return node_button;
	}

	/**
	 * @type {Map<string, IODescription>
	 */
	static io_cache = new Map();

	/**
	 * @param {string} endpoint 
	 * @param {IODescription} io 
	 */
	constructor(endpoint, io) {
		super();
		this.pre_init();

		this.endpoint = endpoint;
		this.net_div = document.createElement("div");
		/**
		 * @type {IODescription}
		 */
		this.io = io;

		this.post_init();
	}

	async fetch_node() {
		while (this.net_div.firstChild) this.net_div.firstChild.remove();

		this.net_div.innerHTML = "<p>Loading...</p>"

		try {
			const resp = await fetch(`${this.endpoint}/contents`, { method: "GET" });
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

	/**
	 * @virtual
	 * @returns {Promise<graph.Pinout | null>}
	 */
	async eval() {
		const pinout = new graph.Pinout();

		const msg = new Message();
		for (const ch of this.input_names()) {
			let cnt = 0;
			for (const edge of this.inputs(ch)) {
				const tensor = await edge.read_packet();
				if (!tensor) return false;
				msg.add_tensor(edge.out_port.channel, tensor);
				cnt++;
			}
			if (!this.io.channel_access_valid("in", ch, cnt)) {
				console.error(`${this}: invalid number of inputs on input "${ch}"`);
				return null;
			}
		}

		await msg.send(this.endpoint);

		for (let i = 0; i < msg.get_tensor_count(); i++) {
			const t = msg.get_nth_tensor(i);
			const ch = msg.get_nth_channel(i);
			pinout.set(ch, t);
		}

		return pinout;
	}

	serialize() {
		return {
			kind: "net_node",
			endpoint: this.endpoint,
			pos: this.pos,
		};
	}
}
