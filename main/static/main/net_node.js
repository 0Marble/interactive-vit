
import * as dataflow from "./dataflow.js";
import * as gpu from "./gpu.js";

/*
 * This node will do opaque calculations on the server
 *
 * APIs:
 * /{endpoint}/description - get IO description
 * /{endpoint}/contents    - get displayed html
 * /{endpoint}/compute     - run eval
 *
 */

class IOPort {
	/**
	 * @param {string} channel 
	 * @param {"in"|"out"} kind 
	 * @param {"1"|"1+"|"*"} access 
	 */
	constructor(kind, channel, access) {
		this.kind = kind;
		this.channel = channel;
		this.access = access;
	}

	static parse(raw_obj) {
		if (typeof raw_obj !== "object") throw new TypeError("IOPort should be an object");

		if (!raw_obj.kind) throw new TypeError("IOPort should have a field 'kind'")
		const kind = raw_obj.kind;
		if (kind !== "in" && kind !== "out") throw new TypeError("IOPort.kind should be \"in\" or \"out\"");

		if (!raw_obj.channel) throw new TypeError("IOPort should have a field 'channel'");
		const channel = raw_obj.channel;
		if (typeof channel !== "string") throw new TypeError("IOPort.channel should be a string");

		if (!raw_obj.access) throw new TypeError("IOPort should have a field 'access'")
		const access = raw_obj.access;
		if (access !== "1" && access !== "1+" && access != "*") throw new TypeError("IOPort.access should be \"1\" or \"1+\" or \"*\"");

		return new IOPort(kind, channel, access);
	}
}

class IODescription {
	/**
	 *
	 * @param {IOPort[]} ports 
	 */
	constructor(ports) {
		this.ports = ports;
	}

	in_names() {
		const res = [];
		for (const port of this.ports) {
			if (port.kind === "in") res.push(port.channel);
		}
		return res;
	}

	out_names() {
		const res = [];
		for (const port of this.ports) {
			if (port.kind === "out") res.push(port.channel);
		}
		return res;
	}

	/**
	 * @param {"in"|"out"} kind 
	 * @param {string} name 
	 * @param {number} count 
	 */
	channel_access_valid(kind, name, count) {
		for (const port of this.ports) {
			if (port.kind === kind && port.channel === name) {
				switch (port.access) {
					case "1": return count === 1;
					case "1+": return count >= 1;
					case "*": return true;
				}
			}
		}
		return false;
	}

	static parse(raw_obj) {
		if (typeof raw_obj !== "object") throw new TypeError("IODescription should be an IOPort[]");
		const ports = [];
		for (const raw_port of raw_obj) ports.push(IOPort.parse(raw_port));
		return new IODescription(ports);
	}

}

export class NetworkNode extends dataflow.NodeFunction {

	/**
	 * @param {string} endpoint 
	 */
	constructor(endpoint, on_io_description_acquired) {
		super();

		this.endpoint = endpoint;
		this.div = document.createElement("div");
		this.io = null;

		fetch(`${endpoint}/description`, { method: "GET" })
			.then(resp => resp.json())
			.then(io => {
				this.io = IODescription.parse(io);
			})
			.then(on_io_description_acquired)
			.catch(err => {
				console.warn("Invalid IO description response:", err);
			});

		this.fetch_node();
	}

	fetch_node() {
		while (this.div.firstChild) this.div.firstChild.remove();

		this.div.innerHTML = "<p>Loading...</p>"
		fetch(`${this.endpoint}/contents`, { method: "GET" })
			.then(resp => resp.text())
			.then(html => {
				this.div.innerHTML = html;
			})
			.catch(err => {
				console.warn("Invalid IO description response:", err);
				this.init_retry();
			});
	}

	init_retry() {
		while (this.div.firstChild) this.div.firstChild.remove();
		const button = document.createElement("button");
		button.textContent = "Retry"
		button.addEventListener("click", () => this.fetch_node());
		this.div.appendChild(button);
	}

	/**
	 *
	 * @param {dataflow.Node} df_node 
	 */
	post_init(df_node, parent_div) {
		console.assert(this.io);
		this.df_node = df_node;
		parent_div.appendChild(this.div);
	}

	/**
	 * @virtual
	 * @returns {Iterable<string>}
	 */
	in_channel_names() {
		return this.io.in_names();
	}

	/**
	 * @virtual
	 * @returns {Iterable<string>}
	 */
	out_channel_names() {
		return this.io.out_names();
	}

	/**
	 * @virtual
	 */
	on_upstream_change() { }

	/**
	 * @virtual
	 * @returns {boolean}
	 */
	eval() {
		return false;
	}

	/**
	 * @virtual
	 * @returns {boolean}
	 */
	verify_io() {
		for (const ch of this.in_channel_names()) {
			let cnt = 0;
			for (const _ of this.df_node.inputs(ch)) cnt++;
			if (!this.io.channel_access_valid("in", ch, cnt) && cnt !== 0) {
				return false;
			}
		}
		for (const ch of this.out_channel_names()) {
			let cnt = 0;
			for (const _ of this.df_node.outputs(ch)) cnt++;
			if (!this.io.channel_access_valid("out", ch, cnt) && cnt !== 0) {
				return false;
			}
		}

		return true;
	}

	/**
	 * Returns `eval`'ed packet, `undefined` if couldn't `eval`
	 * @param {string} channel 
	 * @returns {undefined | any }
	 */
	read_packet(_channel) {
		return undefined;
	}

}
