import * as dataflow from "./dataflow.js";

const graph_div = document.getElementById("graph_div");

export class Node {
	/**
	 * @param {dataflow.Node} n 
	 */
	constructor(n) {
		this.n = n;

		this.div = document.createElement("div");
		this.div.className = "graph_node";
		this.x = 0;
		this.y = 0;

		this.content_div = document.createElement("div");
		this.div.appendChild(this.content_div);
		this.content_div.className = "node_content";

		this.init_header();
		this.init_footer();
		this.init_drag();

		graph_div.appendChild(this.div);
	}

	init_drag() {
		this.div.style = `top: ${this.y}; left: ${this.x};`;
		let delta = { x: 0, y: 0 };
		this.div.addEventListener("dragstart", (event) => {
			delta.x = this.x - event.x;
			delta.y = this.y - event.y;
			event.dataTransfer.effectAllowed = "move";
		});
		this.div.draggable = true;
		this.div.addEventListener("dragend", (event) => {
			this.move_to(event.x + delta.x, event.y + delta.y);
		});
	}

	init_header() {
	}

	init_footer() {
	}

	move_to(x, y) {
		this.x = x;
		this.y = y;

		this.render();
	}

	render() {
		this.div.style = `top: ${this.y}px; left: ${this.x}px;`;
	}
}
