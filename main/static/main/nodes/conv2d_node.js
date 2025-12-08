import * as graph from "../graph.js";
import * as gpu from "../gpu.js";
import * as workspace from "../workspace.js";

const WRK_SIZE = 16;
const conv2d_src = `
${gpu.shader_tesnor_def(0, 0, "read", "input", "f32", 2)}
${gpu.shader_tesnor_def(1, 0, "read", "weight", "f32", 2)}
${gpu.shader_tesnor_def(2, 0, "read_write", "output", "f32", 2)}

override WRK_SIZE = ${WRK_SIZE};
@compute @workgroup_size(WRK_SIZE, WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	var x = id.x;
	var y = id.y;
	if (!output_in_bounds(array(y, x))){
		return;
	}
	
	var sum: f32 = 0.0;
	for (var j: u32 = 0; j < weight_cfg.size[0].x; j++) {
		for (var i: u32 = 0; i < weight_cfg.size[1].x; i++) {
			var idx_1 = input_idx(array(y + j, x + i));
			var idx_2 = weight_idx(array(j, i));
			sum += input[idx_1] * weight[idx_2];
		}
	}

	output[output_idx(array(y, x))] = sum;
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

		this.kernel = new gpu.Kernel(conv2d_src);

		this.matrix_tensor = gpu.Tensor.from_dims_and_data(4, [this.h, this.w], this.matrix.buffer);
		this.matrix_changed = false;
		this.kernel.set_tensor(1, 0, this.matrix_tensor);
		this.matrix_tensor.to_cpu().then(buf => console.log("conv2d_matrix:", buf));

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

		const out_size = {
			h: size.h - 2 * Math.floor(this.h / 2),
			w: size.w - 2 * Math.floor(this.w / 2),
		};
		const output = gpu.Tensor.from_dims_and_data(4, [out_size.h, out_size.w]);

		if (this.matrix_changed) {
			this.matrix_changed = false;
			this.matrix_tensor = gpu.Tensor.from_dims_and_data(4, [this.h, this.w], this.matrix.buffer);
			this.kernel.set_tensor(1, 0, this.matrix_tensor);
		}

		this.kernel.set_tensor(0, 0, input);
		this.kernel.set_tensor(2, 0, output);

		this.kernel.run([
			Math.ceil(output.dims[1] / WRK_SIZE),
			Math.ceil(output.dims[0] / WRK_SIZE),
		]);

		const pinout = new graph.Pinout();
		pinout.set("o", output);
		return pinout;
	}

	static async register_factory() {
		graph.Context.register_deserializer("conv2d", Conv2dNode.deserialize);
		workspace.register_tool("Conv2d", async (x, y) => {
			const node = await Conv2dNode.create();
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.width * 0.5, y - rect.height * 0.5);
		});
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


