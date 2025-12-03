import * as gpu from "./gpu.js";
import * as graph from "./graph.js";
import { init_loader } from "./load.js";

import { ImgSourceNode } from "./nodes/img_source_node.js";
import { ImgViewNode } from "./nodes/img_view_node.js";
import { Conv2dNode } from "./nodes/conv2d_node.js";
import { SliceNode, ShuffleNode } from "./nodes/index.js";
import { NetworkNode } from "./nodes/net_node.js";

await gpu.init();

async function init_toolbar() {
	const toolbar = document.getElementById("toolbar");

	toolbar.appendChild(await ImgSourceNode.register_factory());
	toolbar.appendChild(await ImgViewNode.register_factory());
	toolbar.appendChild(await Conv2dNode.register_factory());
	toolbar.appendChild(await SliceNode.register_factory());
	toolbar.appendChild(await ShuffleNode.register_factory());
	await NetworkNode.register_factory();

	toolbar.appendChild(await init_loader());

	const save = document.createElement("button");
	save.textContent = "Save";
	save.addEventListener("click", async () => {
		await graph.Context.wait_for_not_in_eval();
		let obj = graph.Context.serialize();
		let src = JSON.stringify(obj);

		let data = new Blob([src], { type: "text/plain" });
		let url = URL.createObjectURL(data);

		let a = document.createElement("a");
		a.href = url;
		a.download = "graph.json";
		a.click();
		URL.revokeObjectURL(url);
	});
	toolbar.appendChild(save);
}

await init_toolbar();

// await graph.test();
