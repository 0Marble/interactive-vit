import * as dataflow from "./dataflow.js";
import * as gpu from "./gpu.js";

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

export class ImgViewNode extends dataflow.NodeFunction {
	constructor() {
		super();

		this.div = document.createElement("div");

		this.canvas = document.createElement("canvas");
		this.canvas.className = "image_view_canvas";
		this.ctx = this.canvas.getContext("2d");
		this.div.appendChild(this.canvas);

		this.merge = new gpu.Kernel(merge_kernel_src, [new gpu.UniformInfo("cfg", 4 * 4, 2)]);
		this.clear = new gpu.Kernel(clear_kernel_src);

		this.buf = undefined;
	}

	/**
	 * @param {dataflow.Node} df_node 
	 */
	post_init(df_node, parent_div) {
		this.df_node = df_node;
		parent_div.appendChild(this.div);
	}

	/**
	 * For `impl` in `dataflow.Node`
	 * @override 
	 */
	eval() {
		console.debug(`ImgViewNode.eval(${this.df_node.index})`);
		if (this.buf) return true;

		let size = undefined;
		for (const edge of this.df_node.inputs()) {
			const prev = edge.in_port.node;
			if (!prev.impl.eval()) {
				return false;
			}
			/**
			 * @type {undefined | gpu.Tensor}
			 */
			const packet = prev.impl.read_packet(edge.in_port.channel);
			if (!packet) return false;
			if (packet.dims.length !== 2) {
				console.error(`Invalid input on ImgViewNode (node ${this.df_node.index}). Expected a 2d buffer`);
				return false;
			}
			const w = packet.dims[1];
			const h = packet.dims[0];
			if (!size) {
				size = { w, h };
			} else if (size.w != w || size.h != h) {
				console.error(`Inconsistent input sizes on ImgViewNode (node ${this.df_node.index})`);
				return false;
			}
		}

		if (!size) {
			return true;
		}

		this.canvas.width = size.w;
		this.canvas.height = size.h;

		this.buf = new gpu.Tensor([size.h, size.w], 4);
		for (const edge of this.df_node.inputs()) {
			const prev = edge.in_port.node;
			const packet = prev.impl.read_packet(edge.in_port.channel);
			let cfg = null;
			switch (edge.out_port.channel) {
				case "R": cfg = new Uint32Array([0, 0x000000FF, 0, 0]); break;
				case "G": cfg = new Uint32Array([8, 0x0000FF00, 0, 0]); break;
				case "B": cfg = new Uint32Array([16, 0x00FF0000, 0, 0]); break;
			}

			this.merge.set_uniform("cfg", cfg.buffer);
			this.merge.run(
				[
					{ binding: 0, tensor: this.buf },
					{ binding: 1, tensor: packet },
				],
				Math.ceil(this.buf.elem_cnt / 64),
			);
			this.move_buffer_to_canvas();
		}
		return true;
	}

	move_buffer_to_canvas() {
		const size = this.buf.as_2d_size();

		dataflow.Context.acquire_edit_lock();

		this.buf.to_cpu().then((buf) => {
			const rgba = new Uint8Array(buf);
			const img = this.ctx.createImageData(size.w, size.h);
			for (let i = 0; i < size.w * size.h * 4; i++) {
				img.data[i] = rgba[i];
			}
			this.ctx.putImageData(img, 0, 0);
			dataflow.Context.release_edit_lock();
		});
	}

	/**
	 * @override
	 */
	on_upstream_change() {
		this.buf = undefined;
		this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
	}

	/**
	 * @override
	 * @returns {Iterable<string>}
	 */
	in_channel_names() {
		return ["R", "G", "B"];
	}
}
