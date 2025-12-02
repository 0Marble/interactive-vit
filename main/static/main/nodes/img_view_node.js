import * as gpu from "../gpu.js";
import * as graph from "../graph.js";

const WRK_SIZE = 16;
const merge_kernel_src = `
${gpu.shader_tesnor_def(0, 0, "read_write", "image", "u32", 2)}
${gpu.shader_tesnor_def(1, 0, "read", "data", "f32", 2)}

struct Config {
	bit_offset: u32,
	bit_mask: u32,
	padding: vec2u,
}
@group(2) @binding(0)
var<uniform> cfg: Config;

override WRK_SIZE = ${WRK_SIZE};
@compute @workgroup_size(WRK_SIZE, WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	if (!image_in_bounds(array(id.y, id.x))){
		return;
	}

	var idx_1 = image_idx(array(id.y, id.x));
	var idx_2 = data_idx(array(id.y, id.x));

	var prev = f32((image[idx_1] & cfg.bit_mask) >> cfg.bit_offset) / 255.0;
	var color = u32(clamp(prev + data[idx_2], 0.0, 1.0) * 255.0);
	image[idx_1] &= ~cfg.bit_mask;
	image[idx_1] |= 0xFF000000 | (color << cfg.bit_offset);
}
`;

export class ImgViewNode extends graph.Node {
	constructor() {
		super();
		this.pre_init();

		this.img_div = document.createElement("div");

		this.canvas = document.createElement("canvas");
		this.canvas.className = "image_view_canvas";
		this.ctx = this.canvas.getContext("2d");
		this.img_div.appendChild(this.canvas);

		this.merge = new gpu.Kernel(merge_kernel_src, [new gpu.UniformBinding("cfg", 2, 0, 4 * 4)]);

		this.post_init();
	}

	/**
	 * @override 
	 */
	async eval() {
		this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
		let size = undefined;

		for (const edge of this.inputs()) {
			/**
			 * @type {gpu.Tensor | null}
			 */
			const packet = await edge.read_packet();
			if (!packet) return null;

			if (packet.dims.length !== 2) {
				console.error(`Invalid input on ImgViewNode ${this}. Expected a 2d buffer, got ${packet.dims}`);
				return null;
			}
			const w = packet.dims[1];
			const h = packet.dims[0];
			if (!size) {
				size = { w, h };
			} else if (size.w != w || size.h != h) {
				console.error(`Inconsistent input sizes on ImgViewNode ${this}`);
				return null;
			}
		}

		if (!size) {
			return null;
		}

		this.canvas.width = size.w;
		this.canvas.height = size.h;

		const output = gpu.Tensor.from_dims_and_data(4, [size.h, size.w]);
		this.merge.set_tensor(0, 0, output);
		for (const edge of this.inputs()) {
			/**
			 * @type {gpu.Tensor | null}
			 */
			const input = await edge.read_packet();

			let cfg = null;
			switch (edge.out_port.channel) {
				case "R": cfg = new Uint32Array([0, 0x000000FF, 0, 0]); break;
				case "G": cfg = new Uint32Array([8, 0x0000FF00, 0, 0]); break;
				case "B": cfg = new Uint32Array([16, 0x00FF0000, 0, 0]); break;
			}

			this.merge.set_tensor(1, 0, input);
			this.merge.set_uniform("cfg", cfg.buffer);

			this.merge.run([
				Math.ceil(size.w / WRK_SIZE),
				Math.ceil(size.h / WRK_SIZE),
			]);
		}

		const rgba = new Uint8Array(await output.to_cpu());

		const img = this.ctx.createImageData(size.w, size.h);
		for (let i = 0; i < size.w * size.h * 4; i++) {
			img.data[i] = rgba[i];
		}
		this.ctx.putImageData(img, 0, 0);
		this.on_visual_update();

		return null;
	}

	/**
	 * @override
	 * @returns {Iterable<string>}
	 */
	input_names() {
		return ["R", "G", "B"];
	}

	draw_content() {
		this.content_div.appendChild(this.img_div);
	}

	static async register_factory() {
		const node_button = document.createElement("button");
		node_button.textContent = "New ImgView Node";
		node_button.addEventListener("click", async () => { await ImgViewNode.create() });

		graph.Context.register_deserializer("img_view", ImgViewNode.deserialize);

		return node_button;
	}

	static async create() {
		await graph.Context.wait_for_not_in_eval();
		return new ImgViewNode();
	}

	serialize() {
		return {
			kind: "img_view",
		};
	}

	static async deserialize(_obj) {
		return await ImgViewNode.create();
	}
}
