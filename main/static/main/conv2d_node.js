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
		this.matrix_changed = false;

		this.init_matrix();
	}

	init_matrix() {
		this.div = document.createElement("div");
		const config_div = document.createElement("div");
		config_div.className = "conv_config";

		const label = document.createElement("label");
		label.textContent = "Size:";
		config_div.appendChild(label);
		const inputs = []
		const dim_count = 2;
		const dims = [this.w, this.h];
		for (let i = 0; i < dim_count; i++) {
			const input = document.createElement("input");
			input.type = "number";
			config_div.appendChild(input);
			input.value = `${dims[i]}`;
			input.addEventListener("change", () => {
				const width = +inputs[0].value;
				const height = +inputs[1].value;
				this.resize_matrix(width, height);
			});

			if (i + 1 !== dim_count) {
				const x = document.createElement("span");
				x.textContent = "x";
				config_div.appendChild(x);
			}
			inputs.push(input);
		}

		this.matrix_div = document.createElement("div");
		this.matrix_div.className = "conv_matrix";
		this.div.appendChild(config_div);
		this.div.appendChild(this.matrix_div);
	}

	resize_matrix(w, h) {
		this.w = w;
		this.h = h;

		this.matrix = new Float32Array(w * h);
		this.matrix_changed = true;

		while (this.matrix_div.firstChild) this.matrix_div.firstChild.remove();

		const table = document.createElement("table");
		for (let j = 0; j < h; j++) {
			const row = document.createElement("tr");
			table.appendChild(row);
			for (let i = 0; i < w; i++) {
				const data = document.createElement("td");
				row.appendChild(data);
				const input = document.createElement("input");
				input.className = "matrix_entry";
				input.type = "number";
				input.value = this.matrix[j * w + i];
				input.addEventListener("change", () => {
					this.set_matrix_element(i, j, +input.value);
				});
				data.appendChild(input);
			}
		}
		this.matrix_div.appendChild(table);
	}

	set_matrix_element(i, j, x) {
		this.matrix_changed = true;
		this.matrix[j * this.w + i] = x;
	}

	/**
	 *
	 * @param {dataflow.Node} df_node 
	 */
	post_init(df_node, parent_div) {
		this.df_node = df_node;
		parent_div.appendChild(this.div);
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

		if (this.matrix_changed) {
			this.matrix_changed = false;
			this.matrix_tensor.from_cpu(this.matrix.buffer);
		}

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


