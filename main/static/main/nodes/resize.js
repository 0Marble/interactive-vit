import * as gpu from "../gpu.js";
import * as graph from "../graph.js";
import { Workspace } from "../workspace.js";
import { InputFmt } from "../input_fmt.js";

const WRK_SIZE = 16;

const from_f32_src = `
${gpu.shader_tesnor_def(0, 0, "read", "input", "f32", 3)}
${gpu.shader_tesnor_def(1, 0, "read_write", "output", "u32", 2)}

override WRK_SIZE = ${WRK_SIZE};
@compute @workgroup_size(WRK_SIZE, WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	if (!output_in_bounds(array(id.y, id.x))){
		return;
	}

	var r = u32(input[input_idx(array(0, id.y, id.x))] * 255.0);
	var g = u32(input[input_idx(array(1, id.y, id.x))] * 255.0);
	var b = u32(input[input_idx(array(2, id.y, id.x))] * 255.0);
	output[output_idx(array(id.y, id.x))] = (0xFF << 24) | (r << 0) | (g << 8) | (b << 16);
}
`;

const to_f32_src = `
${gpu.shader_tesnor_def(0, 0, "read", "input", "u32", 2)}
${gpu.shader_tesnor_def(1, 0, "read_write", "output", "f32", 3)}

override WRK_SIZE = ${WRK_SIZE};
@compute @workgroup_size(WRK_SIZE, WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	if (!input_in_bounds(array(id.y, id.x))){
		return;
	}

	var abgr = input[input_idx(array(id.y, id.x))];
	var r = f32((abgr >> 0) & 0xFF) / 255.0;
	var g = f32((abgr >> 8) & 0xFF) / 255.0;
	var b = f32((abgr >> 16) & 0xFF) / 255.0;

	output[output_idx(array(0, id.y, id.x))] = r;
	output[output_idx(array(1, id.y, id.x))] = g;
	output[output_idx(array(2, id.y, id.x))] = b;
}
`;


export class Resize extends graph.Node {
	/**
	 * @param {number} w 
	 * @param {number} h 
	 */
	constructor(w, h) {
		super();
		this.pre_init();

		this.from_f32 = new gpu.Kernel(from_f32_src);
		this.to_f32 = new gpu.Kernel(to_f32_src);
		this.w = w;
		this.h = h;
		this.canvas = document.createElement("canvas");
		this.canvas.width = this.w;
		this.canvas.height = this.h;


		this.post_init();
	}

	input_names() {
		return ["o"];
	}
	output_names() {
		return ["o"];
	}

	draw_content() {
		const div = document.createElement("div");

		const fmt = new InputFmt();
		const resize = async (w, h) => {
			await graph.Context.wait_for_not_in_eval();
			this.w = +w;
			this.h = +h;
			this.canvas.width = this.w;
			this.canvas.height = this.h;
			graph.Context.schedule_eval(this);
			await graph.Context.do_eval();
		};

		fmt.push_text("size:");
		fmt.push_input("w", this.w, async (w) => { await resize(w, this.h) });
		fmt.push_text("x");
		fmt.push_input("h", this.h, async (h) => { await resize(this.w, h) });

		div.append(this.canvas, fmt.div);
		this.content_div.append(div);
	}

	async eval() {
		const edge = this.single_input("o");
		if (!edge) return null;
		/**
		 * @type {gpu.Tensor}
		 */
		const input = await edge.read_packet();
		if (!input) return null;
		if (!input.is_Nd(3)) {
			console.warn(`${this}: expected 3d rgb input, got ${input.dims}`);
			return null;
		}
		const [c, h, w] = input.dims;
		if (c != 3) {
			console.warn(`${this}: expected 3d rgb input, got ${input.dims}`);
			return null;
		}

		const rgba_input = gpu.Tensor.from_dims_and_data(4, [h, w]);
		this.from_f32.set_tensor(0, 0, input);
		this.from_f32.set_tensor(1, 0, rgba_input);
		this.from_f32.run([Math.ceil(w / WRK_SIZE), Math.ceil(h / WRK_SIZE)]);

		const src_canvas = document.createElement("canvas");
		src_canvas.width = w;
		src_canvas.height = h;
		const src_ctx = src_canvas.getContext("2d");
		const in_img = src_ctx.createImageData(w, h);
		in_img.data.set(new Uint8Array(await rgba_input.to_cpu()));
		src_ctx.putImageData(in_img, 0, 0);

		const dst_ctx = this.canvas.getContext("2d");
		dst_ctx.drawImage(src_canvas, 0, 0, w, h, 0, 0, this.w, this.h);

		const out_img = dst_ctx.getImageData(0, 0, this.w, this.h);
		const rgba_output = gpu.Tensor.from_dims_and_data(4, [this.h, this.w], out_img.data.buffer);
		const output = gpu.Tensor.from_dims_and_data(4, [3, this.h, this.w]);

		this.to_f32.set_tensor(0, 0, rgba_output);
		this.to_f32.set_tensor(1, 0, output);
		this.to_f32.run([Math.ceil(this.w / WRK_SIZE), Math.ceil(this.h / WRK_SIZE)]);

		const pinout = new graph.Pinout();
		pinout.set("o", output);
		return pinout;
	}

	static async register_factory() {
		graph.Context.register_deserializer("resize", Resize.deserialize);
		Workspace.register_tool("Resize", async (x, y) => {
			const node = await Resize.create(200, 200);
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.width * 0.5, y - rect.height * 0.5);
		});
	}

	static async create(w, h) {
		await graph.Context.wait_for_not_in_eval();
		return new Resize(w, h);
	}

	serialize() {
		return {
			kind: "resize",
			size: [this.w, this.h]
		};
	}

	static async deserialize(obj) {
		return await Resize.create(...obj.size);
	}
}

