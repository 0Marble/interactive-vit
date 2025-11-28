import * as gpu from "./gpu.js";
import * as graph from "./graph.js";

const clear_kernel_src = `
@group(0) @binding(0)
var<storage, read_write> image: array<u32>;
override WRK_SIZE = 64;
@compute @workgroup_size(WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	image[id.x] = 0xFF000000;
}`;

const merge_kernel_src = `
@group(0) @binding(0)
var<storage, read_write> image: array<u32>;
@group(0) @binding(1)
var<storage, read> data: array<f32>;

struct Config {
	bit_offset: u32,
	bit_mask: u32,
	padding: vec2u,
}
@group(0) @binding(2)
var<uniform> cfg: Config;

override WRK_SIZE = 64;
@compute @workgroup_size(WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	var prev = f32((image[id.x] & cfg.bit_mask) >> cfg.bit_offset) / 255.0;
	var color = u32(clamp(prev + data[id.x], 0.0, 1.0) * 255.0);
	image[id.x] &= ~cfg.bit_mask;
	image[id.x] |= 0xFF000000 | (color << cfg.bit_offset);
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

		this.merge = new gpu.Kernel(merge_kernel_src, [new gpu.UniformInfo("cfg", 4 * 4, 2)]);
		this.clear = new gpu.Kernel(clear_kernel_src);

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
				console.error(`Invalid input on ImgViewNode ${this}. Expected a 2d buffer`);
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

		const output = new gpu.Tensor([size.h, size.w], 4);
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

			this.merge.set_uniform("cfg", cfg.buffer);
			this.merge.run(
				[
					{ binding: 0, tensor: output },
					{ binding: 1, tensor: input },
				],
				Math.ceil(output.elem_cnt / 64),
			);
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
		node_button.addEventListener("click", () => {
			new ImgViewNode();
		});

		graph.Context.register_deserializer("img_view", ImgViewNode.deserialize);

		return node_button;
	}

	serialize() {
		return {
			kind: "img_view",
		};
	}

	static async deserialize(_obj) {
		return new ImgViewNode();
	}
}
