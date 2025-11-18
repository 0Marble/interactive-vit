class Runtime {
	static device = undefined;
}

export class UniformInfo {
	/**
	 * @param {string} name
	 * @param {number} byte_size
	 * @param {number} binding
	 */
	constructor(name, byte_size, binding) {
		this.name = name;
		this.byte_size = byte_size;
		this.binding = binding;
		this.buffer = null;
	}
}

export class Kernel {
	/**
	 * @param {string} source 
	 * @param {UniformInfo[] | undefined} uniforms
	 */
	constructor(source, uniforms) {
		this.kernel = Runtime.device.createShaderModule({
			code: source,
		});
		/**
		 * @type {Map<string, UniformInfo>}
		 */
		this.uniforms = new Map();

		if (uniforms) {
			for (const uni of uniforms) {
				const buf = Runtime.device.createBuffer({
					size: uni.byte_size,
					usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
				});
				uni.buffer = buf;
				this.uniforms.set(uni.name, uni);
			}
		}
	}

	/**
	 * @param {{binding: number, tensor: Tensor}[]} args
	 */
	run(args, workgroups_count) {
		const pipeline = Runtime.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.kernel,
				entryPoint: "main",
			},
		});

		const entries = [];
		for (const { binding, buffer } of this.uniforms.values()) {
			entries.push({ binding, resource: { buffer } });
		}
		for (const { binding, tensor } of args) {
			entries.push({ binding, resource: { buffer: tensor.handle } });
		}

		const group = Runtime.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries,
		});

		const cmd = Runtime.device.createCommandEncoder();
		const pass = cmd.beginComputePass();
		pass.setPipeline(pipeline);
		pass.setBindGroup(0, group);
		pass.dispatchWorkgroups(workgroups_count);
		pass.end();
		Runtime.device.queue.submit([cmd.finish()]);
	}

	/**
	 * @param {string} name
	 * @param {ArrayBuffer} value
	 */
	set_uniform(name, value) {
		const uni = this.uniforms.get(name);
		Runtime.device.queue.writeBuffer(uni.buffer, 0, value);
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
		this.elem_size = elem_size;
		this.byte_size = elem_size;
		this.elem_cnt = 1;
		for (const m of dims) {
			this.byte_size *= m;
			this.elem_cnt *= m;
		}

		this.handle = Runtime.device.createBuffer({
			size: this.byte_size,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			mappedAtCreation: (value ? true : false),
		});

		if (value) {
			new Uint8Array(this.handle.getMappedRange()).set(value);
			this.handle.unmap();
		}
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
		cmd.copyBufferToBuffer(this.handle, staging, this.byte_size);
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
		Runtime.device.queue.writeBuffer(this.handle, 0, data);
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

