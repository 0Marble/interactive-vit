import * as graph from "../graph.js";
import * as gpu from "../gpu.js";
import { Workspace } from "../workspace.js";

const WRK_SIZE = 4;

const shader = `
${gpu.shader_tesnor_def(0, 0, "read", "input", "f32", 3)}
${gpu.shader_tesnor_def(1, 0, "read_write", "output", "u32", 3)}

override WRK_SIZE = ${WRK_SIZE};

@compute @workgroup_size(WRK_SIZE, WRK_SIZE, WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	if (!input_in_bounds(array(id.z, id.y, id.x))) {
		return;
	}

	var a = u32(input[input_idx(array(id.z, id.y, id.x))] * 255.0);
	output[output_idx(array(id.z, id.y, id.x))] = (a << u32(8 * (id.z % 3))) | 0xFF000000;
}
`;

export class MultiView extends graph.Node {

	constructor() {
		super();
		this.pre_init();
		this.kernel = new gpu.Kernel(shader);

		this.images_div = document.createElement("div");
		this.images_div.className = "multi_view_div";
		/**
		 * @type {HTMLCanvasElement[][]}
		 */
		this.canvases = []

		this.post_init();
	}

	input_names() {
		return ["o"];
	}

	draw_content() {
		this.content_div.appendChild(this.images_div);
	}

	/**
	 * @returns {Promise<graph.Pinout|null>}
	 */
	async eval() {
		const edge = this.single_input("o");
		if (edge === null) return null;
		/**
		 * @type {gpu.Tensor}
		 */
		const input = await edge.read_packet();
		if (input === null) return null;

		this.kernel.set_tensor(0, 0, input);

		const [c, h, w] = input.dims;
		const output = gpu.Tensor.from_dims_and_data(4, [c, h, w]);
		console.assert(output.strides[0] == h * w);
		this.kernel.set_tensor(1, 0, output);
		const work = [
			Math.ceil(w / WRK_SIZE),
			Math.ceil(h / WRK_SIZE),
			Math.ceil(c / WRK_SIZE),
		];
		this.kernel.run(work);

		const data = await output.to_cpu();
		console.debug(data);

		this.clear_images();
		const y = Math.ceil(Math.sqrt(c));
		const x = Math.ceil(c / y);
		this.set_image_grid_size(y, x);

		for (let i = 0; i < c; i++) {
			const img = new Uint8Array(data, i * h * w * 4, h * w * 4);
			this.set_image(Math.floor(i / x), i % x, img, w, h);
		}

		this.finish_drawing(c, h, w);
		this.on_visual_update();

		return null;
	}

	clear_images() {
		this.canvases = [];
	}

	/**
	 * @param {number} y 
	 * @param {number} x
	 */
	set_image_grid_size(y, x) {
		this.canvases = [];
		for (let j = 0; j < y; j++) {
			const row = [];
			for (let i = 0; i < x; i++) {
				row.push(null);
			}
			this.canvases.push(row);
		}
	}

	/**
	 * @param {number} y 
	 * @param {number} x 
	 * @param {Uint8Array} data 
	 * @param {number} w 
	 * @param {number} h 
	 */
	set_image(y, x, data, w, h) {
		const canvas = document.createElement("canvas");
		canvas.className = "multi_view_canvas";
		canvas.width = w;
		canvas.height = h;
		this.canvases[y][x] = canvas;

		const ctx = canvas.getContext("2d");
		const img = ctx.createImageData(w, h);
		img.data.set(data);
		ctx.putImageData(img, 0, 0);
	}

	finish_drawing(c, h, w) {
		while (this.images_div.firstChild) this.images_div.firstChild.remove();

		const dim_text_div = document.createElement("div");
		const dim_text = document.createElement("p");
		dim_text.textContent = `${c} channels, ${w} x ${h}`;
		dim_text_div.className = "multi_view_dim_text";
		dim_text_div.appendChild(dim_text);
		this.images_div.appendChild(dim_text_div);

		const table = document.createElement("table");

		const hover_div = document.createElement("div");
		hover_div.className = "multi_view_coords_hover";
		hover_div.style = "visibility: hidden;";
		this.images_div.appendChild(hover_div);

		for (let i = 0; i < this.canvases.length; i++) {
			const tr = document.createElement("tr");
			const row = this.canvases[i];
			for (let j = 0; j < row.length; j++) {
				const td = document.createElement("td");
				tr.appendChild(td);
				const img = row[j];
				if (img === null) continue;

				const div = document.createElement("div");
				div.className = "multi_view_canvas_div";

				img.addEventListener("mouseenter", () => {
					const rect = img.getBoundingClientRect();
					hover_div.style = `visibility: visible; left: ${rect.left}px; top: ${rect.bottom}px;`;
					hover_div.innerHTML = `<p>${i * row.length + j} Ch.<p>`;
					img.style = "border: 2px solid yellow";
				});
				img.addEventListener("mouseleave", () => {
					hover_div.style = "visibility: hidden;";
					img.style = "";
				});

				div.appendChild(img);
				td.appendChild(div);
			}
			table.appendChild(tr);
		}
		this.images_div.appendChild(table);
	}

	static async create() {
		await graph.Context.wait_for_not_in_eval();
		return new MultiView();
	}

	static async register_factory() {
		Workspace.register_tool("MultiView", async (x, y) => {
			const node = await MultiView.create();
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.width / 2, y - rect.height / 2);
		});
		graph.Context.register_deserializer("multi-view", MultiView.deserialize);
	}

	static async deserialize(_obj) {
		const node = await MultiView.create();
		return node;
	}

	serialize() {
		return {
			kind: "multi-view",
		};
	}
}

