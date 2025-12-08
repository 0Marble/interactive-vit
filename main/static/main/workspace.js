import { Picker } from "./picker.js";

const graph_div = document.getElementById("graph_div");
const picker = new Picker({
	open_speed: 0.1,
	expand_amt: 1.2,
});

export function init_workspace() {
	graph_div.addEventListener("contextmenu", (e) => {
		e.preventDefault();
		picker.open(e.clientX + graph_div.scrollLeft, e.clientY + graph_div.scrollTop);
	});
}

export async function register_tool(name, callback) {
	picker.add_option(name, callback);
}
