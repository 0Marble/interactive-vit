import * as gpu from "../gpu.js";
import * as graph from "../graph.js";
import { InputFmt } from "../input_fmt.js";
import { Workspace } from "../workspace.js";

/**
 * @param {any[]} a 
 * @param {any[]} b 
 */
function arr_eql(a, b) {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) {
		if (a[i] !== b[i]) return false;
	}
	return true;
}

const WRK_SIZE = 64;

function make_src(op) {
	const src = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> a_strides: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> b_strides: array<u32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;
@group(0) @binding(5) var<storage, read> c_strides: array<u32>;

@group(0) @binding(6) var<storage, read> dims: array<u32>;
@group(0) @binding(7) var<uniform> cfg: Config;

struct Config {
	dim_cnt: u32,
	a_offset: u32,
	b_offset: u32,
	elem_cnt: u32,
}

override WRK_SIZE = ${WRK_SIZE};
@compute @workgroup_size(WRK_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
	if (id.x > cfg.elem_cnt) {
		return;
	}

	var a_idx = cfg.a_offset;
	var b_idx = cfg.b_offset;
	var c_idx: u32 = 0;
	var k = id.x;

	for (var i: u32 = 0; i < cfg.dim_cnt; i++) {
		var j = cfg.dim_cnt - 1 - i;
		var x = k % dims[j];
		a_idx += x * a_strides[j];
		b_idx += x * b_strides[j];
		c_idx += x * c_strides[j];
		k /= dims[j];
	}

	c[c_idx] = a[a_idx] ${op} b[b_idx];
}
`;

	return src;
}

export class BinOp extends graph.Node {
	constructor(op) {
		super();
		this.pre_init();
		this.op = op;

		/**
		 * @type {Map<string, gpu.Kernel>}
		 */
		this.kernels = new Map();

		const uniforms = [new gpu.UniformBinding("cfg", 0, 7, 4 * 4)];
		for (const op of ["+", "-", "*", "/"]) {
			this.kernels.set(op, new gpu.Kernel(make_src(op), uniforms));
		}

		this.post_init();
	}

	input_names() {
		return ["a", "b"];
	}

	output_names() {
		return ["c"];
	}

	draw_content() {
		while (this.content_div.firstChild) this.content_div.firstChild.remove();

		const fmt = new InputFmt();
		fmt.push_text("c = a ");
		fmt.push_input("op", this.op, async (op) => {
			await graph.Context.wait_for_not_in_eval();
			this.op = op;
			graph.Context.schedule_eval(this);
			await graph.Context.do_eval();
		});
		fmt.push_text("b");
		this.content_div.append(fmt.div);
	}

	async eval() {
		const edge_a = this.single_input("a");
		if (!edge_a) throw new Error("input 'a' not connected");
		const edge_b = this.single_input("b");
		if (!edge_b) throw new Error("input 'b' not connected");

		/**
		 * @type {gpu.Tensor}
		 */
		const a = await edge_a.read_packet();
		if (!a) throw new Error("could not compute 'a'");
		const b = await edge_b.read_packet();
		if (!b) throw new Error("could not compute 'b'");

		if (!arr_eql(a.dims, b.dims)) {
			throw new Error(`error: binop dimension mismatch: a: ${a.dims}, b: ${b.dims}`);
		}

		let elem_cnt = 1;
		for (const x of a.dims) elem_cnt *= x;

		const c = gpu.Tensor.from_dims_and_data(4, a.dims);
		const a_strides = gpu.Tensor.from_dims_and_data(4, [a.dims.length], new Uint32Array(a.strides));
		const b_strides = gpu.Tensor.from_dims_and_data(4, [b.dims.length], new Uint32Array(b.strides));
		const c_strides = gpu.Tensor.from_dims_and_data(4, [c.dims.length], new Uint32Array(c.strides));
		const dims = gpu.Tensor.from_dims_and_data(4, [a.dims.length], new Uint32Array(a.dims));

		const kernel = this.kernels.get(this.op);
		kernel.set_binding(0, 0, a.get_data_buffer());
		kernel.set_binding(0, 1, a_strides.get_data_buffer());
		kernel.set_binding(0, 2, b.get_data_buffer());
		kernel.set_binding(0, 3, b_strides.get_data_buffer());
		kernel.set_binding(0, 4, c.get_data_buffer());
		kernel.set_binding(0, 5, c_strides.get_data_buffer());

		kernel.set_binding(0, 6, dims.get_data_buffer());
		kernel.set_uniform("cfg", new Uint32Array([a.dims.length, a.offset, b.offset, elem_cnt]).buffer);

		kernel.run([Math.ceil(elem_cnt / WRK_SIZE)]);

		const pinout = new graph.Pinout();
		pinout.set("c", c);
		return pinout;
	}

	static async create(op) {
		await graph.Context.wait_for_not_in_eval();
		return new BinOp(op);
	}

	static async register_factory() {
		Workspace.register_tool("BinOp", async (x, y) => {
			const node = await BinOp.create("+");
			const rect = node.div.getBoundingClientRect();
			node.move_to(x - rect.width * 0.5, y - rect.height * 0.5);
		});
		graph.Context.register_deserializer("binop", BinOp.deserialize);
	}

	static async deserialize(obj) {
		const node = await BinOp.create(obj.op);
		return node;
	}

	serialize() {
		return {
			kind: "binop",
			op: this.op,
		};
	}
}
