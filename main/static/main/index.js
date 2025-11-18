import { Conv2dNode } from "./conv2d_node.js";
import * as dataflow from "./dataflow.js";
import * as gpu from "./gpu.js";
import { ImgSourceNode } from "./img_source_node.js";
import { ImgViewNode } from "./img_view_node.js";
import * as ui from "./graph_ui.js";
import { NetworkNode } from "./net_node.js";

await gpu.init();

// const csrf_token = document.querySelector('[name=csrfmiddlewaretoken]')?.value;

function init_toolbar() {
	const toolbar = document.getElementById("toolbar");

	const img_src_tool = document.createElement("button");
	img_src_tool.textContent = "ImageSrc";
	toolbar.appendChild(img_src_tool);
	img_src_tool.addEventListener("click", () => {
		const img_src = new ImgSourceNode();
		const df = new dataflow.Node(img_src);
		const ui_node = new ui.Node(df);
		img_src.post_init(df, ui_node.content_div);
	});

	const img_view_tool = document.createElement("button");
	img_view_tool.textContent = "ImageView";
	toolbar.appendChild(img_view_tool);
	img_view_tool.addEventListener("click", () => {
		const img_view = new ImgViewNode();
		const df = new dataflow.Node(img_view);
		const ui_node = new ui.Node(df);
		img_view.post_init(df, ui_node.content_div);
	});

	const conv2d_tool = document.createElement("button");
	conv2d_tool.textContent = "Conv2D";
	toolbar.appendChild(conv2d_tool);
	conv2d_tool.addEventListener("click", () => {
		const conv2d = new Conv2dNode();
		const df = new dataflow.Node(conv2d);
		const ui_node = new ui.Node(df);
		conv2d.post_init(df, ui_node.content_div);
	});

	const net_tool = document.createElement("button");
	net_tool.textContent = "DummyNet"
	toolbar.appendChild(net_tool);
	net_tool.addEventListener("click", () => {
		const net = new NetworkNode("dummy", () => {
			const df = new dataflow.Node(net);
			const ui_node = new ui.Node(df);
			net.post_init(df, ui_node.content_div);
		});
	});
}

init_toolbar();

