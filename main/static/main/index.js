import * as gpu from "./gpu.js";
import { init_loader, init_saver } from "./load.js";

import { ImgSourceNode } from "./nodes/img_source_node.js";
import { ImgViewNode } from "./nodes/img_view_node.js";
import { Conv2dNode } from "./nodes/conv2d_node.js";
import { SliceNode, ShuffleNode } from "./nodes/index.js";
import { NetworkNode } from "./nodes/net_node.js";
import { Workspace } from "./workspace.js";
import { MultiView } from "./nodes/multi_view.js";
import { ConstNode } from "./nodes/const.js";
import { BinOp } from "./nodes/binop.js";
import { Noise } from "./nodes/noise.js";

await gpu.init();

async function init_toolbar() {
	const toolbar = document.getElementById("toolbar");

	toolbar.appendChild(await init_loader());
	toolbar.appendChild(await init_saver());

	let toolbar_open = false;
	document.getElementById("open_toolbar").addEventListener("click", (e) => {
		e.stopPropagation();
		if (!toolbar_open) {
			toolbar.style = "visibility: visible;";
			toolbar_open = true;
		} else {
			toolbar.style = "";
			toolbar_open = false;
		}
	});
}

await Workspace.init();
await ImgSourceNode.register_factory();
await ImgViewNode.register_factory();
await Conv2dNode.register_factory();
await SliceNode.register_factory();
await ShuffleNode.register_factory();
await MultiView.register_factory();
await ConstNode.register_factory();
await BinOp.register_factory();
await Noise.register_factory();

await NetworkNode.register_factory();

await init_toolbar();

// await graph.test();
