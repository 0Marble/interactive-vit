import * as gpu from "../gpu.js";
import * as graph from "../graph.js";
import { InputFmt } from "../input_fmt.js";
import { Workspace } from "../workspace.js";

export class ConstNode extends graph.Node {
	/**
	 * @param {number} value 
	 */
	constructor(value) {
		super();
		this.pre_init();

		this.value = value;
		/**
		 * @type {number[]}
		 */
		this.dims = [1];
		/**
		 * @type{gpu.Tensor}
		 */
		this.tensor = null;
		this.rebuild_tensor();

		this.size_div = null;

		this.post_init();
	}

	draw_content() {
		if (this.size_div === null) {
			this.size_div = document.createElement("div");
			this.size_div.className = "const_size_div";
			this.content_div.appendChild(this.size_div);
		}
		while (this.size_div.firstChild) this.size_div.firstChild.remove();

		const fmt = new InputFmt();
		fmt.push_text("const(");
		fmt.push_input("x", this.value, async (value) => {
			await graph.Context.wait_for_not_in_eval();
			this.value = +value;
			this.rebuild_tensor();

			graph.Context.schedule_eval(this);
			await graph.Context.do_eval();
		});

		fmt.push_text(", [");
		for (let i = 0; i < this.dims.length; i++) {
			fmt.push_input("input_" + i, this.dims[i], (value) => this.on_dim_input(i, value));
			fmt.push_text(",");
		}
		fmt.push_text("])");

		const button = document.createElement("button");
		button.textContent = "+";
		button.addEventListener("click", async () => {
			await graph.Context.wait_for_not_in_eval();

			const i = this.dims.length;
			this.dims.push(1);
			fmt.pop();
			fmt.push_input("input_" + i, this.dims[i], (value) => this.on_dim_input(i, value));
			fmt.push_text(",");
			fmt.push_text("])");

			graph.Context.schedule_eval(this);
			await graph.Context.do_eval();
		});

		this.size_div.append(fmt.div);
		this.size_div.append(button);
	}

	async on_dim_input(i, value) {
		await graph.Context.wait_for_not_in_eval();
		this.dims[i] = +value;
		this.rebuild_tensor();
		graph.Context.schedule_eval(this);
		await graph.Context.do_eval();
	}

	rebuild_tensor() {
		let elem_cnt = 1;
		for (const x of this.dims) elem_cnt *= x;

		const arr = new Float32Array(elem_cnt);
		arr.fill(this.value);

		this.tensor = gpu.Tensor.from_dims_and_data(4, this.dims, arr.buffer);
	}

	output_names() {
		return ["o"];
	}

	async eval() {
		const pinout = new graph.Pinout();
		pinout.set("o", this.tensor);
		return pinout;
	}

	/**
	 * @param {number} value 
	 */
	static async create(value) {
		await graph.Context.wait_for_not_in_eval();
		return new ConstNode(value);
	}

	static async register_factory() {
		Workspace.register_tool("Const", async (x, y) => {
			const node = await ConstNode.create(0);
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.left, y - rect.top);
		});
		graph.Context.register_deserializer("const", ConstNode.deserialize);
	}

	static async deserialize(obj) {
		const node = await ConstNode.create(+obj.value);
		return node;
	}

	serialize() {
		return {
			kind: "const",
			value: this.value,
		};
	}
}

