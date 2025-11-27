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

export class ImgSourceNode extends dataflow.Node {
	constructor() {
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
		this.has_img = false;

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

	post_init(parent_div) {
		parent_div.appendChild(this.div);
	}

	/**
	 * When an image is loaded, it is drawn onto the `this.canvas`,
	 * then, this is called.
	 */
	on_image_loaded() {
		this.invalidate_with_descendants();
		this.has_img = true;
		dataflow.Context.acquire_edit_lock();
		this.eval_with_descendants().then(() => dataflow.Context.release_edit_lock());
	}

	/**
	 * @override 
	 */
	async eval_impl() {
		if (!this.has_img) return false;
		if (this.bufs) {
			this.emit_result("R", this.bufs[1]);
			this.emit_result("G", this.bufs[2]);
			this.emit_result("B", this.bufs[3]);
			return true;
		}

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

		this.emit_result("R", this.bufs[1]);
		this.emit_result("G", this.bufs[2]);
		this.emit_result("B", this.bufs[3]);

		return true;
	}

	/**
	 * @override
	 */
	invalidate_impl() {
		this.bufs = null;
	}

	/**
	 * @override
	 * @returns {Iterable<string>}
	 */
	out_channel_names() {
		return ["R", "G", "B"];
	}

	/**
	 * @override
	 */
	kind() {
		return "img_src";
	}
}
