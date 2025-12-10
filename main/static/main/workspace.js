import { Picker } from "./picker.js";

const workspace = document.getElementById("workspace");
const graph_div = document.getElementById("graph_div");
const grid_svg = document.getElementById("grid_svg");

export class Workspace {
	/**
	 * @private
	 */
	constructor() {
		this.picker = new Picker({
			open_speed: 0.1,
			expand_amt: 1.2,
		});

		this.offset = { x: 0, y: 0 };
		workspace.addEventListener("contextmenu", (e) => {
			e.preventDefault();
			this.picker.open(e.clientX, e.clientY);
		});

		const drag_start = { x: 0, y: 0 };
		let in_drag = false;
		workspace.addEventListener("mousedown", (e) => {
			if (!in_drag && e.button == 1) {
				workspace.style = "cursor: move;";
				drag_start.x = e.x;
				drag_start.y = e.y;
				in_drag = true;
			}
		});
		workspace.addEventListener("mousemove", (e) => {
			if (in_drag) {
				const x = this.offset.x + e.x - drag_start.x;
				const y = this.offset.y + e.y - drag_start.y;
				graph_div.style = `left: ${x}px; top: ${y}px;`;
				this.draw_grid(x, y);
			}
		});
		workspace.addEventListener("mouseup", (e) => {
			if (in_drag) {
				workspace.style = "cursor: auto;";
				in_drag = false;
				this.offset.x += (e.x - drag_start.x);
				this.offset.y += (e.y - drag_start.y);
			}
		});
		this.draw_grid(0, 0);
	}

	draw_grid(x, y) {
		while (grid_svg.firstChild) grid_svg.firstChild.remove();

		const spacing = 20;
		const big = 5;
		const small_color = "#666644";
		const small_width = "0.5px";
		const big_color = "#888800";
		const big_width = "2px";

		const x_offset = x - Math.floor(x / spacing) * spacing;
		const y_offset = y - Math.floor(y / spacing) * spacing;
		const X_offset = x - Math.floor(x / (spacing * big)) * (spacing * big);
		const Y_offset = y - Math.floor(y / (spacing * big)) * (spacing * big);

		this.draw_grid_impl(x_offset, y_offset, spacing, small_color, small_width);
		this.draw_grid_impl(X_offset, Y_offset, spacing * big, big_color, big_width);
	}

	draw_grid_impl(left, top, spacing, color, width) {
		const rect = workspace.getBoundingClientRect()
		for (let x = left; x < rect.right; x += spacing) {
			const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
			line.setAttribute("x1", x);
			line.setAttribute("y1", 0);
			line.setAttribute("x2", x);
			line.setAttribute("y2", rect.height);
			line.setAttribute("stroke", color);
			line.setAttribute("stroke-width", width);

			grid_svg.appendChild(line);
		}

		for (let y = top; y < rect.bottom; y += spacing) {
			const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
			line.setAttribute("x1", 0);
			line.setAttribute("y1", y);
			line.setAttribute("x2", rect.width);
			line.setAttribute("y2", y);
			line.setAttribute("stroke", color);
			line.setAttribute("stroke-width", width);

			grid_svg.appendChild(line);
		}
	}

	static instance = null;
	static async init() {
		Workspace.instance = new Workspace();
	}

	static register_tool(name, callback) {
		Workspace.instance.picker.add_option(name, (x, y) => {
			callback(x - Workspace.instance.offset.x, y - Workspace.instance.offset.y);
		});
	}
}
