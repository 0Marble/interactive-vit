import * as dataflow from "./dataflow.js";
import * as gpu from "./gpu.js";

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

export class ImgSourceNode extends dataflow.NodeFunction {
	static counter = 0;
	constructor() {
		ImgSourceNode.counter += 1;
		super();

		this.div = document.createElement("div");
		const input = document.createElement("input");
		input.type = "file";
		input.accept = "image/*";

		this.canvas = document.createElement("canvas");
		this.canvas.className = "image_view_canvas";
		this.ctx = this.canvas.getContext("2d");
		this.div.appendChild(input);
		this.div.appendChild(this.canvas);

		input.addEventListener("change", () => {
			const file = input.files[0];
			const img = document.createElement("img");
			img.src = URL.createObjectURL(file);

			img.addEventListener("load", () => {
				this.canvas.width = img.width;
				this.canvas.height = img.height;
				this.ctx.drawImage(img, 0, 0, img.width, img.height);
				this.on_image_loaded();
			});
		});

		this.kernel = new gpu.Kernel(split_kernel_src);
		this.bufs = undefined;
	}

	/**
	 * @param {dataflow.Node} df_node 
	 */
	post_init(df_node, parent_div) {
		this.df_node = df_node;
		parent_div.appendChild(this.div);
	}

	/**
	 * When an image is loaded, it is drawn onto the `this.canvas`,
	 * then, this is called.
	 * 1. Runs the required kernels 
	 * 2. Does a `dataflow.Node.on_upstream_change()`.
	 */
	on_image_loaded() {
		const img_data = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);

		this.bufs = [
			new gpu.Tensor([this.canvas.height, this.canvas.width], 4, img_data.data),
			new gpu.Tensor([this.canvas.height, this.canvas.width], 4),
			new gpu.Tensor([this.canvas.height, this.canvas.width], 4),
			new gpu.Tensor([this.canvas.height, this.canvas.width], 4),
		];
		this.kernel.run(
			this.bufs.map((tensor, binding) => { return { binding, tensor }; }),
			Math.ceil(this.canvas.width * this.canvas.height / 64),
		);
		this.df_node.on_this_changed();
	}

	/**
	 * For `impl` in `dataflow.Node`
	 * @override 
	 */
	eval() {
		console.debug(`ImgSourceNode.eval(${this.df_node.index})`);
		if (!this.bufs) return false;
		return true;
	}

	/**
	 * @param {string} channel 
	 */
	read_packet(channel) {
		if (!this.bufs) return undefined;

		switch (channel) {
			case "R": return this.bufs[1];
			case "G": return this.bufs[2];
			case "B": return this.bufs[3];
			default: return undefined;
		}
	}

	/**
	 * @override
	 */
	on_upstream_change() {
		this.bufs = undefined;
		this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
	}

	/**
	 * @override
	 * @returns {Iterable<string>}
	 */
	out_channel_names() {
		return ["R", "G", "B"];
	}
}
