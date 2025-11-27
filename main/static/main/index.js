import * as gpu from "./gpu.js";
import * as ui from "./graph_ui.js";

import { ImgSourceNode } from "./img_source_node.js";
import { Conv2dNode } from "./conv2d_node.js";
import { ImgViewNode } from "./img_view_node.js";

import * as graph from "./graph.js";

await gpu.init();

function init_toolbar() {
	const toolbar = document.getElementById("toolbar");

	const img_src_tool = document.createElement("button");
	img_src_tool.textContent = "ImageSrc";
	toolbar.appendChild(img_src_tool);
	img_src_tool.addEventListener("click", () => {
		const img_src = new ImgSourceNode();
		const ui_node = new ui.Node(img_src);
		img_src.post_init(ui_node.content_div);
	});

	const img_view_tool = document.createElement("button");
	img_view_tool.textContent = "ImageView";
	toolbar.appendChild(img_view_tool);
	img_view_tool.addEventListener("click", () => {
		const img_view = new ImgViewNode();
		const ui_node = new ui.Node(img_view);
		img_view.post_init(ui_node.content_div);
	});

	const conv2d_tool = document.createElement("button");
	conv2d_tool.textContent = "Conv2D";
	toolbar.appendChild(conv2d_tool);
	conv2d_tool.addEventListener("click", () => {
		const conv2d = new Conv2dNode();
		const ui_node = new ui.Node(conv2d);
		conv2d.post_init(ui_node.content_div);
	});
}

init_toolbar();

await graph.test();
