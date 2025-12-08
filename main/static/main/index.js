import * as gpu from "./gpu.js";
import * as graph from "./graph.js";
import { init_loader, init_saver } from "./load.js";

import { ImgSourceNode } from "./nodes/img_source_node.js";
import { ImgViewNode } from "./nodes/img_view_node.js";
import { Conv2dNode } from "./nodes/conv2d_node.js";
import { SliceNode, ShuffleNode } from "./nodes/index.js";
import { NetworkNode } from "./nodes/net_node.js";
import { init_workspace } from "./workspace.js";

await gpu.init();

async function init_toolbar() {
	const toolbar = document.getElementById("toolbar");

	await init_workspace();

	toolbar.appendChild(await ImgSourceNode.register_factory());
	toolbar.appendChild(await ImgViewNode.register_factory());
	toolbar.appendChild(await Conv2dNode.register_factory());
	toolbar.appendChild(await SliceNode.register_factory());
	toolbar.appendChild(await ShuffleNode.register_factory());
	await NetworkNode.register_factory();

	toolbar.appendChild(await init_loader());
	toolbar.appendChild(await init_saver());
}

await init_toolbar();

// await graph.test();
