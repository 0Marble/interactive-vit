import * as graph from "../graph.js";
import * as gpu from "../gpu.js";

const split_kernel_src = `
@group(0) @binding(0)
var<storage, read> image: array<u32>;
@group(0) @binding(1)
var<storage, read_write> r_channel: array<f32>;
@group(0) @binding(2)
var<storage, read_write> g_channel: array<f32>;
@group(0) @binding(3)
var<storage, read_write> b_channel: array<f32>;

override WRK_SIZE = 64;
@compute @workgroup_size(WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	r_channel[id.x] = f32((image[id.x] & 0x000000FF) >> 0) / 255;
	g_channel[id.x] = f32((image[id.x] & 0x0000FF00) >> 8) / 255;
	b_channel[id.x] = f32((image[id.x] & 0x00FF0000) >> 16) / 255;
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

		this.kernel = new gpu.Kernel(split_kernel_src);

		this.post_init();
	}

	/**
	 * @override 
	 * @returns {Promise<graph.Pinout | null>}
	 */
	async eval() {
		if (!this.has_img) return null;

		const img_data = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);

		const bufs = [
			new gpu.Tensor([this.canvas.height, this.canvas.width], 4, img_data.data),
			new gpu.Tensor([this.canvas.height, this.canvas.width], 4),
			new gpu.Tensor([this.canvas.height, this.canvas.width], 4),
			new gpu.Tensor([this.canvas.height, this.canvas.width], 4),
		];
		this.kernel.run(
			bufs.map((tensor, binding) => { return { binding, tensor }; }),
			Math.ceil(this.canvas.width * this.canvas.height / 64),
		);

		const pinout = new graph.Pinout();
		pinout.set("R", bufs[1]);
		pinout.set("G", bufs[2]);
		pinout.set("B", bufs[3]);

		return pinout;
	}

	/**
	 * @override
	 * @returns {Iterable<string>}
	 */
	output_names() {
		return ["R", "G", "B"];
	}

	draw_content() {
		this.content_div.appendChild(this.img_div);
	}

	static async register_factory() {

		const node_button = document.createElement("button");
		node_button.textContent = "New ImgSrc Node";
		node_button.addEventListener("click", async () => { await ImgSourceNode.create(); });

		graph.Context.register_deserializer("img_src", ImgSourceNode.deserialize);

		return node_button;
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
