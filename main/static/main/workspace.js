import { Picker } from "./picker.js";

const workspace = document.getElementById("workspace");
const graph_div = document.getElementById("graph_div");
/**
 * @type{HTMLCanvasElement}
 */
const grid_canvas = document.getElementById("grid_canvas");

export class Workspace {
	/**
	 * @private
	 */
	constructor() {
		this.picker = new Picker({
			open_speed: 0.1,
			expand_amt: 1.2,
		});
		const rect = workspace.getBoundingClientRect()
		grid_canvas.width = rect.width;
		grid_canvas.height = rect.height;

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
		grid_canvas.getContext("2d").clearRect(0, 0, grid_canvas.width, grid_canvas.height);

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
		const ctx = grid_canvas.getContext("2d");
		ctx.strokeStyle = color;
		ctx.lineWidth = width;
		ctx.beginPath();

		for (let x = left; x < rect.right; x += spacing) {
			ctx.moveTo(x, rect.top);
			ctx.lineTo(x, rect.bottom);
		}

		for (let y = top; y < rect.bottom; y += spacing) {
			ctx.moveTo(rect.left, y);
			ctx.lineTo(rect.right, y);
		}

		ctx.stroke();
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
