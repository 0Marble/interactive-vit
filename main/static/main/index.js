import { Conv2dNode } from "./conv2d_node.js";
import * as dataflow from "./dataflow.js"
import * as gpu from "./gpu.js"
import { ImgSourceNode } from "./img_source_node.js";
import { ImgViewNode } from "./img_view_node.js";
import * as ui from "./graph_ui.js";

await gpu.init();

// const csrf_token = document.querySelector('[name=csrfmiddlewaretoken]')?.value;

function gaussian_conv(r, sigma) {
	const n = r * 2 + 1;
	let mat = new Float32Array(n * n);
	const mul = 1.0 / (2.0 * Math.PI * sigma * sigma);
	for (let j = 0; j < n; j++) {
		for (let i = 0; i < n; i++) {
			let x = i - r;
			let y = j - r;
			const g = mul * Math.exp(-(x * x + y * y) / (2.0 * sigma * sigma));
			mat[j * n + i] = g;
		}
	}
	const conv = new Conv2dNode(n, n, mat);
	const df = new dataflow.Node(conv);
	conv.post_init(df);
	return df;
}

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
}

init_toolbar();


async function run_test() {

	const img_view = new ImgViewNode();
	const df2 = new dataflow.Node(img_view);
	img_view.post_init(df2, document.getElementById("img2"));

	const df3 = gaussian_conv(7, 3.0);

	const edges = []
	for (const dim of ["R", "G", "B"]) {
		await new Promise(r => setTimeout(r, 5 * 1000));
		const e = dataflow.Node.connect(new dataflow.Port(df1, dim, "out"), new dataflow.Port(df2, dim, "in"));
		edges.push(e);
	}

	for (const e of edges) {
		await new Promise(r => setTimeout(r, 1 * 1000));
		e.disconnect();
	}

	await new Promise(r => setTimeout(r, 1 * 1000));
	dataflow.Node.connect(new dataflow.Port(df1, "R", "out"), new dataflow.Port(df3, "o", "in"));
	for (const dim of ["R", "G", "B"]) {
		await new Promise(r => setTimeout(r, 1 * 1000));
		dataflow.Node.connect(new dataflow.Port(df3, "o", "out"), new dataflow.Port(df2, dim, "in"));
	}
}

