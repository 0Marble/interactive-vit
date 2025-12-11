import * as gpu from "../gpu.js";
import * as graph from "../graph.js";
import { InputFmt } from "../input_fmt.js";
import { Workspace } from "../workspace.js";

export class Noise extends graph.Node {
	constructor(dims) {
		super();
		this.pre_init();

		this.dims = dims;
		/**
		 * @type {gpu.Tensor}
		 */
		this.tensor = null;
		this.randomize();

		this.post_init();
	}

	randomize() {
		let elem_cnt = 1;
		for (const x of this.dims) elem_cnt *= x;

		const buf = new Float32Array(elem_cnt);
		for (let i = 0; i < elem_cnt; i++) buf[i] = Math.random();
		this.tensor = gpu.Tensor.from_dims_and_data(4, this.dims, buf);
	}

	draw_content() {
		while (this.content_div.firstChild) this.content_div.firstChild.remove();

		const fmt = new InputFmt();
		fmt.push_text("noise([");
		for (let i = 0; i < this.dims.length; i++) {
			fmt.push_input("input_" + i, this.dims[i], (value) => this.on_input(i, value));
			fmt.push_text(", ");
		}
		fmt.push_text("])");

		const button = document.createElement("button");
		button.textContent = "+";
		button.addEventListener("click", async () => {
			await graph.Context.wait_for_not_in_eval();

			const i = this.dims.length;
			this.dims.push(1);

			fmt.pop();
			fmt.push_input("input_" + i, this.dims[i], (value) => this.on_input(i, value));
			fmt.push_text(", ");
			fmt.push_text("])");

			graph.Context.schedule_eval(this);
			await graph.Context.do_eval();
		});

		this.content_div.append(fmt.div, button);
	}

	async on_input(i, value) {
		await graph.Context.wait_for_not_in_eval();

		this.dims[i] = +value;
		this.randomize();

		graph.Context.schedule_eval(this);
		await graph.Context.do_eval();
	}

	output_names() {
		return ["o"];
	}

	async eval() {
		const pinout = new graph.Pinout();
		pinout.set("o", this.tensor);
		return pinout;
	}

	static async create(dims) {
		await graph.Context.wait_for_not_in_eval();
		return new Noise(dims);
	}

	static async register_factory() {
		Workspace.register_tool("Noise", async (x, y) => {
			const node = await Noise.create([1]);
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.width * 0.5, y - rect.height * 0.5);
		});
		graph.Context.register_deserializer("noise", Noise.deserialize);
	}

	static async deserialize(obj) {
		const node = await Noise.create(obj.dims);
		return node;
	}

	serialize() {
		return {
			kind: "noise",
			dims: this.dims,
		};
	}
}

