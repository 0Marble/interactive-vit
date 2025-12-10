import * as graph from "../graph.js";
import * as gpu from "../gpu.js";
import { Workspace } from "../workspace.js";
import { InputFmt } from "../input_fmt.js";

class IndexNode extends graph.Node {
	constructor() {
		super();
		this.pre_init();

		/**
		 * @type {Map<number,number>}
		 */
		this.fixed_dims = new Map();
		/**
		 * @type {Map<number,number>}
		 */
		this.free_dims = new Map();

		this.dim_cnt = 0;

		this.post_init();
	}

	async eval() {
		const e = this.single_input("o");
		if (!e) return null;
		/**
		 * @type {gpu.Tensor}
		 */
		const input = await e.read_packet();
		if (!input) return null;
		if (!input.is_Nd(this.dim_cnt)) {
			console.error(
				`IndexNode(${this}): invalid input size, got`,
				input.dims.length,
				"expected",
				this.dim_cnt,
			);
			return null;
		}

		const output = new gpu.Tensor(4, 0);
		output.dims = [];
		output.strides = [];
		output.data_buffer = input.data_buffer;
		output.offset = input.offset;
		for (let i = 0; i < this.free_dims.size; i++) {
			output.dims.push(0);
			output.strides.push(0);
		}

		for (const [dim, val] of this.fixed_dims) {
			output.offset += input.strides[dim] * val;
		}

		for (const [dim, order] of this.free_dims) {
			output.dims[order] = input.dims[dim];
			output.strides[order] = input.strides[dim];
		}

		const pinout = new graph.Pinout();
		pinout.set("o", output);
		return pinout;
	}

	input_names() {
		return ["o"];
	}

	output_names() {
		return ["o"];
	}

	serialize() {
		const fixed = [];
		for (const [dim, val] of this.fixed_dims) {
			fixed.push({ dim, val });
		}
		const free = [];
		for (const [in_dim, out_dim] of this.free_dims) {
			free.push({ in_dim, out_dim });
		}

		return {
			kind: "index",
			fixed,
			free,
		};
	}
}

export class SliceNode extends IndexNode {
	constructor() {
		super();
	}

	draw_content() {
		while (this.content_div.firstChild) this.content_div.firstChild.remove();

		const div = document.createElement("div");
		div.className = "index_node_dims";

		const text_start = document.createElement("span");
		text_start.textContent = "input[";
		const text_end = document.createElement("span");
		text_end.textContent = "]";

		const dim_inputs = [];
		for (let i = 0; i < this.dim_cnt; i++) {
			const input = document.createElement("input");
			input.className = "index_node_dim_input";
			input.type = "text";
			dim_inputs.push(input);

			if (this.fixed_dims.has(i)) {
				input.value = this.fixed_dims.get(i);
			}
			if (this.free_dims.has(i)) {
				input.value = ":";
			}

			input.addEventListener("change", async () => {
				if (input.value !== ":" && +input.value === NaN) {
					return;
				}
				await graph.Context.wait_for_not_in_eval();
				graph.Context.schedule_eval(this);
				this.read_inputs(dim_inputs);
				await graph.Context.do_eval();
			});
		}

		div.appendChild(text_start);
		for (const input of dim_inputs) {
			div.appendChild(input);
			const coma = document.createElement("span");
			coma.textContent = ",";
			div.appendChild(coma);
		}

		const plus_button = document.createElement("button");
		plus_button.textContent = "+";
		plus_button.addEventListener("click", async () => {
			await graph.Context.wait_for_not_in_eval();
			graph.Context.schedule_eval(this);

			this.dim_cnt++;
			this.draw_content();

			await graph.Context.do_eval();
		});

		div.appendChild(plus_button);

		div.appendChild(text_end);

		this.content_div.appendChild(div);
	}

	read_inputs(dim_inputs) {
		this.free_dims.clear();
		this.fixed_dims.clear();

		let out_idx = 0;
		for (let i = 0; i < this.dim_cnt; i++) {
			const val = dim_inputs[i].value;
			if (val === ":") {
				this.free_dims.set(i, out_idx);
				out_idx++;
			} else if (+val !== NaN) {
				this.fixed_dims.set(i, +val);
			}
		}
	}

	static async register_factory() {
		graph.Context.register_deserializer("slice", SliceNode.deserialize);
		Workspace.register_tool("Slice", async (x, y) => {
			const node = await SliceNode.create();
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.width * 0.5, y - rect.height * 0.5);
		});
	}

	/**
	 * @returns{Promise<SliceNode>}
	 */
	static async create() {
		await graph.Context.wait_for_not_in_eval();
		return new SliceNode();
	}

	serialize() {
		const obj = super.serialize();
		obj.kind = "slice";
		return obj;
	}

	static async deserialize(obj) {
		const node = await SliceNode.create();
		for (const { dim, val } of obj.fixed) {
			node.fixed_dims.set(dim, val);
		}
		for (const { in_dim, out_dim } of obj.free) {
			node.free_dims.set(in_dim, out_dim);
		}
		node.dim_cnt = node.free_dims.size + node.fixed_dims.size;
		node.draw_content();

		return node;
	}
}

export class ShuffleNode extends IndexNode {
	constructor() {
		super();
	}

	draw_content() {
		while (this.content_div.firstChild) this.content_div.firstChild.remove();

		const div = document.createElement("div");
		const fmt = new InputFmt();

		fmt.push_text("shuffle([");
		for (let i = 0; i < this.dim_cnt; i++) {
			let value = "";
			if (this.free_dims.has(i)) value = this.free_dims.get(i);
			fmt.push_input("input_" + i, value, async (value) => this.on_input(i, value));
			fmt.push_text(", ");
		}
		fmt.push_text("])");

		const button = document.createElement("button");
		button.textContent = "+";
		button.addEventListener("click", async () => {
			await graph.Context.wait_for_not_in_eval();

			const i = this.dim_cnt;
			fmt.pop();
			fmt.push_input("input_" + i, "", async (value) => this.on_input(i, value));
			fmt.push_text(", ");
			fmt.push_text("])");

			this.dim_cnt += 1;

			graph.Context.schedule_eval(this);
			await graph.Context.do_eval();
		});

		div.append(fmt.div);
		div.append(button);
		this.content_div.appendChild(div);
	}

	async on_input(i, value) {
		await graph.Context.wait_for_not_in_eval();

		this.free_dims.set(i, +value);

		graph.Context.schedule_eval(this);
		await graph.Context.do_eval();
	}

	static async register_factory() {
		graph.Context.register_deserializer("shuffle", ShuffleNode.deserialize);
		Workspace.register_tool("Shuffle", async (x, y) => {
			const node = await ShuffleNode.create();
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.width * 0.5, y - rect.height * 0.5);
		});
	}

	/**
	 * @returns{Promise<ShuffleNode>}
	 */
	static async create() {
		await graph.Context.wait_for_not_in_eval();
		return new ShuffleNode();
	}

	serialize() {
		const obj = super.serialize();
		obj.kind = "shuffle";
		return obj;
	}

	static async deserialize(obj) {
		const node = await ShuffleNode.create();
		for (const { dim, val } of obj.fixed) {
			node.fixed_dims.set(dim, val);
		}
		for (const { in_dim, out_dim } of obj.free) {
			node.free_dims.set(in_dim, out_dim);
		}
		node.dim_cnt = node.free_dims.size + node.fixed_dims.size;
		node.draw_content();

		return node;
	}
}
