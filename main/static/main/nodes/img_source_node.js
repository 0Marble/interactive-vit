import * as graph from "../graph.js";
import * as gpu from "../gpu.js";
import { Workspace } from "../workspace.js";

const WRK_SIZE = 16;
const to_rgb_src = `
${gpu.shader_tesnor_def(0, 0, "read", "input", "u32", 2)}
${gpu.shader_tesnor_def(1, 0, "read_write", "output", "f32", 3)}

override WRK_SIZE = ${WRK_SIZE};
@compute @workgroup_size(WRK_SIZE, WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	if (!input_in_bounds(array(id.y, id.x))) {
		return;
	}

	var idx = input_idx(array(id.y, id.x));
	var r = f32((input[idx] & 0x000000FF) >> 0) / 255;
	var g = f32((input[idx] & 0x0000FF00) >> 8) / 255;
	var b = f32((input[idx] & 0x00FF0000) >> 16) / 255;

	output[output_idx(array(0, id.y, id.x))] = r;
	output[output_idx(array(1, id.y, id.x))] = g;
	output[output_idx(array(2, id.y, id.x))] = b;
}
`;

export class ImgSourceNode extends graph.Node {

	constructor() {
		super();

		this.pre_init();

		this.has_img = false;

		this.img_div = document.createElement("div");
		const input = document.createElement("input");
		input.type = "file";
		input.accept = "image/*";

		this.canvas = document.createElement("canvas");
		this.canvas.className = "image_view_canvas";
		this.ctx = this.canvas.getContext("2d");

		const input_div = document.createElement("div");
		const canvas_div = document.createElement("div");
		input_div.appendChild(input);
		canvas_div.appendChild(this.canvas);
		this.img_div.appendChild(input_div);
		this.img_div.appendChild(canvas_div);

		input.addEventListener("change", () => {
			const file = input.files[0];
			const img = document.createElement("img");
			img.src = URL.createObjectURL(file);

			img.addEventListener("load", async () => {
				await graph.Context.wait_for_not_in_eval();

				this.canvas.width = img.width;
				this.canvas.height = img.height;
				this.ctx.drawImage(img, 0, 0, img.width, img.height);
				this.has_img = true;
				this.on_visual_update();

				graph.Context.schedule_eval(this);

				await graph.Context.do_eval();
			});
		});

		this.kernel = new gpu.Kernel(to_rgb_src);

		this.post_init();
	}

	/**
	 * @override 
	 * @returns {Promise<graph.Pinout | null>}
	 */
	async eval() {
		if (!this.has_img) return null;

		const img_data = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);

		const input = gpu.Tensor.from_dims_and_data(4, [this.canvas.height, this.canvas.width], img_data.data);
		const output = gpu.Tensor.from_dims_and_data(4, [3, this.canvas.height, this.canvas.width]);

		this.kernel.set_tensor(0, 0, input);
		this.kernel.set_tensor(1, 0, output);
		this.kernel.run([
			Math.ceil(this.canvas.width / WRK_SIZE),
			Math.ceil(this.canvas.height / WRK_SIZE),
		]);

		const pinout = new graph.Pinout();
		pinout.set("o", output);

		return pinout;
	}

	/**
	 * @override
	 * @returns {Iterable<string>}
	 */
	output_names() {
		return ["o"];
	}

	draw_content() {
		this.content_div.appendChild(this.img_div);
	}

	static async register_factory() {
		graph.Context.register_deserializer("img_src", ImgSourceNode.deserialize);
		Workspace.register_tool("ImgSrc", async (x, y) => {
			const node = await ImgSourceNode.create();
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.width * 0.5, y - rect.height * 0.5);
		});
	}

	static async create() {
		await graph.Context.wait_for_not_in_eval();
		const node = new ImgSourceNode();
		return node;
	}

	serialize() {
		return {
			kind: "img_src",
		};
	}

	static async deserialize(_obj) {
		return await ImgSourceNode.create();
	}
}
