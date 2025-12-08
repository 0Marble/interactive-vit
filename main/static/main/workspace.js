import { Picker } from "./picker.js";

const graph_div = document.getElementById("graph_div");
const picker = new Picker({
	open_speed: 0.1,
	expand_amt: 1.2,
});

export function init_workspace() {
	picker.add_option("foo", () => { console.log("picked foo"); });
	picker.add_option("bar", () => { console.log("picked bar"); });
	picker.add_option("baz", () => { console.log("picked baz"); });

	graph_div.addEventListener("contextmenu", (e) => {
		e.preventDefault();
		picker.open(e.clientX + graph_div.scrollLeft, e.clientY + graph_div.scrollTop);
	});
}
