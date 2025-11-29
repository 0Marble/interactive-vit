class Runtime {
	static device = undefined;
}

export class UniformBinding {
	/**
	 * @param {string} name 
	 * @param {number} byte_size
	 * @param {number} binding
	 * @param {number} group 
	 */
	constructor(name, group, binding, byte_size) {
		this.name = name;
		this.byte_size = byte_size;
		this.binding = binding;
		this.group = group;

		this.buf = Runtime.device.createBuffer({
			size: this.byte_size,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}
}

class Group {
	/**
	 * @param {number} n 
	 */
	constructor(n, pipeline) {
		this.n = n;
		this.handle = null;
		this.pipeline = pipeline;

		/**
		 * @type {Map<number, any>}
		 */
		this.buffers = new Map();
	}

	set(binding, buffer) {
		this.handle = null;
		this.buffers.set(binding, buffer);
	}

	get_handle() {
		if (this.handle) return this.handle;

		const entries = [];
		for (const [binding, buffer] of this.buffers) {
			entries.push({ binding, resource: { buffer } });
		}
		this.handle = Runtime.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(this.n),
			entries,
		});
		return this.handle;
	}
}

export class Kernel {
	/**
	 * @param {string} source 
	 * @param {UniformBinding[]|undefined} uniforms 
	 */
	constructor(source, uniforms) {
		this.kernel = Runtime.device.createShaderModule({
			code: source,
		});

		this.pipeline = Runtime.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.kernel,
				entryPoint: "main",
			},
		});

		/**
		 * @type {Map<number, Group>}
		 */
		this.groups = new Map();
		/**
		 * @type {Map<string, UniformBinding>}
		 */
		this.uniforms = new Map();

		if (uniforms) {
			for (const uni of uniforms) {
				this.uniforms.set(uni.name, uni);
				this.set_binding(uni.group, uni.binding, uni.buf);
			}
		}
	}

	/**
	 * @param {number} group 
	 * @param {number} binding
	 * @private
	 */
	set_binding(group, binding, buffer) {
		if (!this.groups.has(group)) {
			this.groups.set(group, new Group(group, this.pipeline));
		}
		const g = this.groups.get(group);
		g.set(binding, buffer);
	}

	/**
	 * @param {number} group 
	 * @param {number} binding_offset 
	 * @param {Tensor} tensor 
	 */
	set_tensor(group, binding_offset, tensor) {
		this.set_binding(group, binding_offset + 0, tensor.data_buffer);
		this.set_binding(group, binding_offset + 1, tensor.cfg_buffer);
	}

	/**
	 * @param {string} name
	 * @param {ArrayBuffer} value
	 */
	set_uniform(name, value) {
		const uni = this.uniforms.get(name);
		Runtime.device.queue.writeBuffer(uni.buf, 0, value);
	}

	/**
	 * @param {number[]} workgroups_count
	 */
	run(workgroups_count) {
		const cmd = Runtime.device.createCommandEncoder();
		const pass = cmd.beginComputePass();
		pass.setPipeline(this.pipeline);

		for (const group of this.groups.values()) {
			pass.setBindGroup(group.n, group.get_handle());
		}

		pass.dispatchWorkgroups(...workgroups_count);
		pass.end();
		Runtime.device.queue.submit([cmd.finish()]);
	}
}

export class Tensor {
	/**
	 * @param {number[]} dims 
	 * @param {number} elem_size
	 * @param {ArrayBuffer | undefined} value
	 */
	constructor(dims, elem_size, value) {
		this.dims = dims;
		this.strides = [];
		this.offset = 0;

		this.elem_size = elem_size;
		this.byte_size = elem_size;
		this.elem_cnt = 1;
		for (const m of dims) {
			this.byte_size *= m;
			this.elem_cnt *= m;
			this.strides.push(1);
		}
		for (let i = 1; i < this.dims.length; i++) {
			const j = this.dims.length - i - 1;
			this.strides[j] = this.strides[j + 1] * this.dims[j + 1];
		}

		this.data_buffer = Runtime.device.createBuffer({
			size: this.byte_size,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			mappedAtCreation: (value ? true : false),
		});

		this.cfg_buffer = Runtime.device.createBuffer({
			size: this.dims.length * 16 * 2 + 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true,
		});

		if (value) {
			new Uint8Array(this.data_buffer.getMappedRange()).set(value);
			this.data_buffer.unmap();
		}

		const cfg = new Uint32Array(this.cfg_buffer.getMappedRange());
		for (let i = 0; i < this.dims.length; i++) {
			const d = this.dims[i];
			const o = this.strides[i];
			cfg[4 * i + 0] = d;
			cfg[4 * i + 4 * this.dims.length] = o;
		}
		cfg[4 * 4 * 2 * this.dims.length] = this.offset;

		this.cfg_buffer.unmap();
	}

	/**
	 * @param {number} n 
	 * @returns {boolean}
	 */
	is_Nd(n) {
		return this.dims.length == n;
	}

	/**
	 *
	 * @returns {undefined | {w: number, h: number}}
	 */
	as_2d_size() {
		if (!this.is_Nd(2)) return undefined;
		return { w: this.dims[1], h: this.dims[0] };
	}

	/**
	 * @returns {Promise<ArrayBuffer>}
	 */
	async to_cpu() {
		const staging = Runtime.device.createBuffer({
			size: this.byte_size,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});
		const cmd = Runtime.device.createCommandEncoder();
		cmd.copyBufferToBuffer(this.data_buffer, staging, this.byte_size);
		Runtime.device.queue.submit([cmd.finish()]);
		await staging.mapAsync(GPUMapMode.READ, 0, this.byte_size);
		const res = staging.getMappedRange(0, this.byte_size).slice(0)
		staging.unmap();
		return res;
	}

	/**
	 * @param {ArrayBuffer} data 
	 */
	from_cpu(data) {
		Runtime.device.queue.writeBuffer(this.data_buffer, 0, data);
	}
}

export async function init() {
	if (!navigator.gpu) {
		alert("WebGPU not supported or disabled in your browser!");
		throw new Error("unsupported");
	}
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		alert("WebGPU not supported or disabled in your browser!");
		throw new Error("unsupported");
	}

	const device = await adapter.requestDevice();
	if (!device) {
		alert("WebGPU not supported or disabled in your browser!");
		throw new Error("unsupported");
	}

	console.debug("Initialized WebGPU");

	Runtime.device = device;
}


/**
 * @param {number} group 
 * @param {number} binding_offset 
 * @param {string} name 
 * @param {string} type 
 * @param {number} dim 
 * @param {string} access 
 */
export function shader_tesnor_def(group, binding_offset, access, name, type, dim) {
	return `
@group(${group}) @binding(${binding_offset})
var<storage, ${access}> ${name}: array<${type}>;

struct ${name}_config_t {
	size: array<vec4u, ${dim}>,
	stride: array<vec4u, ${dim}>,
	offset: u32,
	padding0: u32,
	padding1: u32,
	padding2: u32,
}
@group(${group}) @binding(${binding_offset + 1}) 
var<uniform> ${name}_cfg: ${name}_config_t;

fn ${name}_idx(idx: array<u32, ${dim}>) -> u32 {
	var res = ${name}_cfg.offset;
	for (var i = 0; i < ${dim}; i++) {
		res += idx[i] * ${name}_cfg.stride[i].x;
	}
	return res;
}

fn ${name}_in_bounds(idx: array<u32, ${dim}>) -> bool {
	for (var i = 0; i < ${dim}; i++) {
		if (idx[i] >= ${name}_cfg.size[i].x) {
			return false;
		}
	}
	return true;
}
`;
}
