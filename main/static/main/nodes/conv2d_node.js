import * as graph from "../graph.js";
import * as gpu from "../gpu.js";

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

export class Conv2dNode extends graph.Node {
	/**
	 * @param {number|undefined} w 
	 * @param {number|undefined} h 
	 * @param {Float32Array|undefined} matrix 
	 */
	constructor(w, h, matrix) {
		super();
		this.pre_init();

		this.w = w || 3;
		this.h = h || 3;
		this.matrix = new Float32Array(this.w * this.h);
		if (matrix) this.matrix.set(matrix);

		this.kernel = new gpu.Kernel(conv2d_src, [new gpu.UniformInfo("cfg", 4 * 4, 3)]);
		this.matrix_tensor = new gpu.Tensor([this.h, this.w], 4, new Uint8Array(this.matrix.buffer));
		this.matrix_changed = false;
		this.matrix_div = document.createElement("div");

		this.post_init();
	}

	draw_content() {
		const div = document.createElement("div");
		div.appendChild(this.draw_dim_section());
		div.appendChild(this.draw_matrix_section());
		this.content_div.appendChild(div);
	}

	draw_dim_section() {
		const dim_inputs = [];
		const dim_div = document.createElement("div");
		const dim_change = async () => {
			await graph.Context.wait_for_not_in_eval();

			this.w = +dim_inputs[0].value;
			this.h = +dim_inputs[1].value;
			this.matrix = new Float32Array(this.w * this.h);
			this.matrix_changed = true;
			this.draw_matrix_section();
			this.on_visual_update();

			await graph.Context.do_eval();
		};

		const w_input = document.createElement("input");
		w_input.type = "number";
		w_input.className = "conv_config";
		w_input.value = this.w;
		w_input.addEventListener("change", dim_change);

		const h_input = document.createElement("input");
		h_input.type = "number";
		h_input.className = "conv_config";
		h_input.value = this.h;
		h_input.addEventListener("change", dim_change);

		dim_div.appendChild(w_input);
		const x = document.createElement("span");
		x.textContent = "x";
		dim_div.appendChild(x);
		dim_div.appendChild(h_input);

		dim_inputs.push(w_input);
		dim_inputs.push(h_input);

		return dim_div;
	}

	draw_matrix_section() {
		while (this.matrix_div.firstChild) this.matrix_div.firstChild.remove();

		const table = document.createElement("table");

		for (let i = 0; i < this.h; i++) {
			const row = document.createElement("tr");
			table.appendChild(row);
			for (let j = 0; j < this.w; j++) {
				const cell = document.createElement("td");
				row.appendChild(cell);
				const idx = i * this.w + j;

				const input = document.createElement("input");
				input.type = "number";
				input.className = "matrix_entry";
				input.value = this.matrix[idx];

				input.addEventListener("change", async () => {
					await graph.Context.wait_for_not_in_eval();
					this.matrix_changed = true;
					this.matrix[idx] = +input.value;
					graph.Context.schedule_eval(this);
					await graph.Context.do_eval();
				});

				cell.appendChild(input);
			}
		}

		this.matrix_div.appendChild(table);
		return this.matrix_div;
	}

	/**
	 * @override
	 * @returns {Iterable<string>}
	 */
	input_names() {
		return ["o"];
	}

	/**
	 * @override
	 * @returns {Iterable<string>}
	 */
	output_names() {
		return ["o"];
	}

	/**
	 * @override
	 * @returns {Promise<graph.Pinout | null>}
	 */
	async eval() {
		const edge = this.single_input("o");
		if (edge === null) return null;

		/**
		 * @type {gpu.Tensor | null}
		 */
		const input = await edge.read_packet();
		if (input === null) return null;
		const size = input.as_2d_size();
		if (!size) {
			console.error(`Only 2d convolutions supported (node ${this.format()})`);
			return null;
		}

		const output = new gpu.Tensor(
			[
				size.h - 2 * Math.floor(this.h / 2),
				size.w - 2 * Math.floor(this.w / 2),
			],
			4,
		);
		this.kernel.set_uniform("cfg", new Uint32Array([size.w, size.h, this.w, this.h]).buffer);

		if (this.matrix_changed) {
			this.matrix_changed = false;
			this.matrix_tensor = new gpu.Tensor([this.h, this.w], 4, new Uint8Array(this.matrix.buffer));
		}

		this.kernel.run([
			{ binding: 0, tensor: input },
			{ binding: 1, tensor: this.matrix_tensor },
			{ binding: 2, tensor: output },
		], Math.ceil(output.elem_cnt / 64));


		const pinout = new graph.Pinout();
		pinout.set("o", output);
		return pinout;
	}

	static async register_factory() {
		const node_button = document.createElement("button");
		node_button.textContent = "New Conv2d Node";
		node_button.addEventListener("click", async () => { await Conv2dNode.create(3, 3) });

		graph.Context.register_deserializer("conv2d", Conv2dNode.deserialize);

		return node_button;
	}

	static async create(w, h, matrix) {
		await graph.Context.wait_for_not_in_eval();
		return new Conv2dNode(w, h, matrix);
	}

	serialize() {
		return {
			kind: "conv2d",
			dim: [this.h, this.w],
			data: base64_encode(this.matrix.buffer),
		};
	}

	static async deserialize(obj) {
		const [h, w] = obj.dim;
		const matrix = new Float32Array(base64_decode(obj.data));
		const node = await Conv2dNode.create(w, h, matrix);
		return node;
	}
}

/**
 * @param {ArrayBuffer} data 
 */
function base64_encode(data) {
	const bytes = new Uint8Array(data);
	if (bytes.toBase64) {
		return bytes.toBase64();
	} else {
		throw new Error("browser doesnt support Uint8Array.toBase64()");
	}
}

/**
 * @param {string} str 
 */
function base64_decode(base64) {
	if (Uint8Array.fromBase64) {
		return Uint8Array.fromBase64(base64).buffer;
	} else {
		throw new Error("browser doesnt support Uint8Array.fromBase64()");
	}
}


