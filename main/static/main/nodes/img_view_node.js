import * as gpu from "../gpu.js";
import * as graph from "../graph.js";
import { Workspace } from "../workspace.js";

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

			let err = false;
			let h = 0;
			let w = 0;
			if (packet.dims.length === 2) {
				[h, w] = packet.dims;
				if (!size) {
					size = { w, h };
				} else if (size.w != w || size.h != h) {
					throw new Error(`inconsistent input sizes on`);
				}
			} else if (packet.dims.length === 3 && edge.out_port.channel === "o") {
				const c = packet.dims[0];
				h = packet.dims[1];
				w = packet.dims[2];
				if (c !== 3) err = true;
			} else {
				err = true;
			}

			if (!size) {
				size = { w, h };
			} else if (size.w != w || size.h != h) {
				throw new Error(`inconsistent input sizes`);
			}

			if (err) {
				throw new Error(`Invalid input. Expected a 2d buffer, got ${packet.dims}`);
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
				case "o": {
					for (let i = 0; i < 3; i++) {
						const ch = gpu.Tensor.from_dims_and_data(4, [size.h, size.w]);
						ch.data_buffer = input.get_data_buffer();
						ch.strides = [input.strides[1], input.strides[2]];
						ch.offset = input.offset + input.strides[0] * i;
						const cfg = new Uint32Array([i * 8, 0xFF << (i * 8), 0, 0]);

						this.merge.set_tensor(1, 0, ch);
						this.merge.set_uniform("cfg", cfg.buffer);

						this.merge.run([
							Math.ceil(size.w / WRK_SIZE),
							Math.ceil(size.h / WRK_SIZE),
						]);
					}
					continue;
				}
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
		return ["R", "G", "B", "o"];
	}

	draw_content() {
		this.content_div.appendChild(this.img_div);
	}

	static async register_factory() {
		graph.Context.register_deserializer("img_view", ImgViewNode.deserialize);
		Workspace.register_tool("ImgView", async (x, y) => {
			const node = await ImgViewNode.create();
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.width * 0.5, y - rect.height * 0.5);
		});
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
