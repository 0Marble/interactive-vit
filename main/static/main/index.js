import * as gpu from "./gpu.js";

import { ImgSourceNode } from "./img_source_node.js";
import { ImgViewNode } from "./img_view_node.js";
import { Conv2dNode } from "./conv2d_node.js";
import { NetworkNode } from "./net_node.js";

await gpu.init();

async function init_toolbar() {
	await ImgSourceNode.register_factory();
	await ImgViewNode.register_factory();
	await Conv2dNode.register_factory();
	await NetworkNode.register_factory("dummy");
}

await init_toolbar();

// await graph.test();
