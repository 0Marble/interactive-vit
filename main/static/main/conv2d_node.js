import * as dataflow from "./dataflow.js";
import * as gpu from "./gpu.js";

const conv2d_src = `
@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read> kernel: array<f32>;
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

struct Config {
	input_w: u32,
	input_h: u32,
	kernel_w: u32,
	kernel_h: u32,
}
@group(0) @binding(3)
var<uniform> cfg: Config;

override WRK_SIZE = 64;
@compute @workgroup_size(WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	var output_w = cfg.input_w - 2 * (cfg.kernel_w / 2);
	var output_h = cfg.input_h - 2 * (cfg.kernel_h / 2);
	var x = id.x % output_w;
	var y = id.x / output_w;
	if (y >= output_h) {
		return;
	}
	
	var sum: f32 = 0.0;
	for (var j: u32 = 0; j < cfg.kernel_h; j++) {
		for (var i: u32 = 0; i < cfg.kernel_w; i++) {
			var a = input[(y + j) * cfg.input_w + (x + i)];
			var b = kernel[j * cfg.kernel_w + i];
			sum += a * b;
		}
	}

	output[id.x] = sum;
}
`;

export class Conv2dNode extends dataflow.NodeFunction {
	/**
	 * @param {number|undefined} w 
	 * @param {number|undefined} h 
	 * @param {number[]|undefined} matrix 
	 */
	constructor(w, h, matrix) {
		super();

		this.w = w || 3;
		this.h = h || 3;
		this.matrix = matrix || new Float32Array(this.w * this.h);
		this.kernel = new gpu.Kernel(conv2d_src, [new gpu.UniformInfo("cfg", 4 * 4, 3)]);
		this.matrix_tensor = new gpu.Tensor([this.h, this.w], 4, new Uint8Array(this.matrix.buffer));
	}

	/**
	 *
	 * @param {dataflow.Node} df_node 
	 */
	post_init(df_node) {
		this.df_node = df_node;
	}

	/**
	 * @override
	 * @returns {Iterable<string>}
	 */
	in_channel_names() {
		return ["o"];
	}

	/**
	 * @override
	 * @returns {Iterable<string>}
	 */
	out_channel_names() {
		return ["o"];
	}

	/**
	 * @override
	 */
	on_upstream_change() {
		this.buf = undefined;
	}

	/**
	 * @override
	 * @returns {boolean}
	 */
	eval() {
		console.log(`Conv2dNode.eval(${this.df_node.index})`);
		if (this.buf) {
			return true;
		}
		/**
		 * @type {dataflow.Edge}
		 */
		const edge = this.df_node.inputs().next().value;
		if (!edge) return false;

		const prev = edge.in_port.node;
		if (!prev.impl.eval()) return false;
		/**
		 * @type {gpu.Tensor}
		 */
		const packet = prev.impl.read_packet(edge.in_port.channel);
		if (!packet) return false;
		const size = packet.as_2d_size();
		if (!size) {
			console.error(`Only 2d convolutions supported (node ${this.df_node.index})`);
		}
		this.buf = new gpu.Tensor(
			[
				size.h - 2 * Math.floor(this.h / 2),
				size.w - 2 * Math.floor(this.w / 2),
			],
			4,
		);
		this.kernel.set_uniform("cfg", new Uint32Array([size.w, size.h, this.w, this.h]).buffer);

		this.kernel.run([
			{ binding: 0, tensor: packet },
			{ binding: 1, tensor: this.matrix_tensor },
			{ binding: 2, tensor: this.buf },
		], Math.ceil(this.buf.elem_cnt / 64));


		return true;
	}

	/**
	 * @override
	 * @returns {boolean}
	 */
	verify_io() {
		let len = 0;
		for (const _ of this.df_node.inputs()) {
			len += 1;
		}
		if (len > 1) return false;
		return true;
	}

	/**
	 * @override
	 * @param {string} channel 
	 * @returns {undefined | any }
	 */
	read_packet(channel) {
		return this.buf;
	}
}


