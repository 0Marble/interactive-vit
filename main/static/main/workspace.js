import { Picker } from "./picker.js";

const graph_div = document.getElementById("graph_div");
const picker = new Picker({
	open_speed: 0.1,
	expand_amt: 1.2,
});

const drag_start = { x: 0, y: 0 };

export function init_workspace() {
	graph_div.addEventListener("contextmenu", (e) => {
		e.preventDefault();
		picker.open(e.clientX + graph_div.scrollLeft, e.clientY + graph_div.scrollTop);
	});
	// graph_div.addEventListener("dragstart", (e) => {
	// 	graph_div.style = "cursor: move;";
	// 	drag_start.x = e.x;
	// 	drag_start.y = e.y;
	// });
	// graph_div.addEventListener("dragend", (e) => {
	// 	graph_div.style = "cursor: auto;";
	// });
	// graph_div.draggable = true;
}

export async function register_tool(name, callback) {
	picker.add_option(name, callback);
}
