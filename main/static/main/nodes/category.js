import * as gpu from "../gpu.js";
import * as graph from "../graph.js";

export class Category extends graph.Node {
	/**
	 * @param {string[]} cats 
	 */
	constructor(cats) {
		super();
		this.pre_init();

		this.cats = cats;
		this.scores = new Float32Array(this.cats.length);

		this.post_init();
	}

	input_names() {
		return ["o"];
	}

	draw_content() {
		while (this.content_div.firstChild) this.content_div.firstChild.remove();

		const sorted_ids = [];
		const n = this.cats.length;
		for (let i = 0; i < n; i++) {
			sorted_ids.push(i);
		}
		sorted_ids.sort((i, j) => this.scores[i] - this.scores[j]);
		const max = this.scores[n - 1];

		const scroll_div = document.createElement("div");
		scroll_div.className = "category_div";
		for (let i = 0; i < n; i++) {
			const j = sorted_ids[n - i - 1];
			const div = document.createElement("div");
			const w = (this.scores[j] / max) * 100.0;
			div.style = `background: linear-gradient(to-right, #888800 ${w}%, transparent ${w}%);`;
			div.append(`${this.cats[j]}: ${this.scores[j]}`);
			scroll_div.append(div);
		}

		this.content_div.append(scroll_div);
	}

	async eval() {
		const edge = this.single_input("o");
		if (!edge) throw new Error("input not connected");
		/**
		 * @type {gpu.Tensor}
		 */
		const t = await edge.read_packet();
		if (!t) throw new Error("could not compute input");

		if (!t.is_Nd(1) || t.dims[0] !== this.cats.length) throw new Error(
			`expected a 1D input of size ${this.cats.length}, got [${t.dims}]`
		);

		const n = this.cats.length;
		const arr = new Float32Array(await t.to_cpu());
		for (let i = 0; i < n; i++) {
			const j = i * t.strides[0] + t.offset;
			this.scores[i] = arr[j];
		}
		this.draw_content();

		return null;
	}

	static async register_factory() {
		graph.Context.register_deserializer("category", Category.deserialize);
	}

	static async create(cats) {
		await graph.Context.wait_for_not_in_eval();
		return new Category(cats);
	}

	serialize() {
		return {
			kind: "category",
			cats: this.cats,
		};
	}

	static async deserialize(obj) {
		return await Category.create(obj.cats);
	}
}

