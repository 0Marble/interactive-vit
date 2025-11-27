import { ImgSourceNode } from "./img_source_node.js";
import { Conv2dNode } from "./conv2d_node.js";
import { ImgViewNode } from "./img_view_node.js";
import { NetworkNode } from "./net_node.js";
import * as ui from "./graph_ui.js";

/**
 * @typedef {Node[]} GraphDescription
 *
 * Node.tensor is available only on Node.kind=conv2d 
 * Node.endpoint is available only on Node.kind=net_node
 * @typedef {{
 *		kind: "img_src" | "img_view" | "conv2d" | "net_node", 
 *		edges: Edge[],
 *		tensor: Tensor | undefined,
 *		endpoint: string | undefined,
 *	}} Node
 *
 *	@typedef {{channel: string, dir: "in"|"out", node: number }} Port
 *	@typedef {{in_port: Port, out_port: Port}} Edge
 *
 *	Tensor.data is a f32[] encoded in base64
 *	@typedef {{shape: number[], data: string}} Tensor
 */

/**
 * @param {string} base64 
 * @returns {Float32Array}
 */
function decode_base64_f32_array(base64) {
	if (Uint8Array.fromBase64) {
		/**
		 * @type {Uint8Array}
		 */
		const bytes = Uint8Array.fromBase64(base64, { alphabet: "base64url" });
		return new Float32Array(bytes.buffer);
	} else {
		throw new Error("old browser doesnt support Uint8Array.fromBase64");
	}
}

/**
 * Loads a graph from json string
 * @param {string} json_str 
 */
export async function from_string(json_str) {
	/**
	 * @type {GraphDescription}
	 */
	const obj = JSON.parse(json_str);
	/**
	 * @type {ui.Node[]}
	 */
	const nodes = [];

	for (const node of obj) {
		/**
		 * @type {ui.Node}
		 */
		let ui_node = null;

		switch (node.kind) {
			case "img_src": {
				const n = new ImgSourceNode();
				ui_node = new ui.Node(n);
				n.post_init(ui_node.content_div);
			} break;
			case "img_view": {
				const n = new ImgViewNode();
				ui_node = new ui.Node(n);
				n.post_init(ui_node.content_div);
			} break;
			case "conv2d": {
				const [h, w] = node.tensor.shape;
				const data = decode_base64_f32_array(node.tensor.data);

				const n = new Conv2dNode(w, h, data);
				ui_node = new ui.Node(n);
				n.post_init(ui_node.content_div);
			} break;
			case "net_node": {
				const n = await NetworkNode.load(node.endpoint)
				ui_node = new ui.Node(n);
				n.post_init(ui_node.content_div);
			} break;
		}

		nodes.push(n);
	}

	for (let i = 0; i < nodes.length; i++) {
		for (const e of obj[i].edges) {
			const a = nodes[e.in_port.node];
			const b = nodes[e.out_port.node];
			await ui.Node.connect(
				new ui.Port(a, e.in_port.dir, e.in_port.channel),
				new ui.Port(b, e.out_port.dir, e.out_port.channel),
			);
		}
	}
}

