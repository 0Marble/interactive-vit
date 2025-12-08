const body = document.getElementById("graph_div");

export class Picker {
	static counter = 0;

	/**
	 * @param {{
	 *	radius: number | undefined
	 *	open_speed: number | undefined
	 *	expand_amt: number | undefined
	 * }} props
	 */
	constructor(props) {
		Picker.counter++;

		this.index = Picker.counter;

		this.div = document.createElement("div");
		this.div.style = `visibility: hidden;`;
		this.div.addEventListener("contextmenu", (e) => {
			e.stopPropagation();
			e.preventDefault();
			this.close();
		});

		this.div.className = "picker";
		body.appendChild(this.div);

		this.is_open = false;

		/**
		 * @type {{title: string, on_click: ()=>Promise<void>}[]}
		 */
		this.options = [];
		this.x = 0;
		this.y = 0;

		this.radius = props.radius || 100
		this.open_speed = props.open_speed || 1
		this.expand_amt = props.expand_amt || 1.5
	}

	/**
	 * @param {string} title 
	 * @param {() => Promise<void>} on_click 
	 */
	add_option(title, on_click) {
		this.options.push({ title, on_click });
	}

	/**
	 * @param {number} x 
	 * @param {number} y 
	 */
	open(x, y) {
		this.div.style = "visibility: visible;";
		this.x = x;
		this.y = y;
		this.is_open = true;
		this.redraw();
	}

	close() {
		this.div.style = `visibility: hidden;`;
		this.is_open = false;
	}

	/**
	 * @private
	 */
	redraw() {
		while (this.div.firstChild) this.div.firstChild.remove();

		const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
		svg.setAttribute("width", "100%");
		svg.setAttribute("height", "100%");
		svg.setAttribute("overflow", "visible");

		const angle = Math.PI * 2.0 / this.options.length;
		const poly_radius = this.expand_amt * this.radius / Math.cos(angle / 2);

		for (let i = 0; i < this.options.length; i++) {
			const rad_x = Math.sin(angle * i) * this.radius + this.x;
			const rad_y = Math.cos(angle * i) * this.radius + this.y;

			const text_x = Math.sin(angle * (i + 0.5)) * this.radius * 0.5 + this.x;
			const text_y = Math.cos(angle * (i + 0.5)) * this.radius * 0.5 + this.y;

			const x1 = Math.sin(angle * i) * poly_radius + this.x;
			const y1 = Math.cos(angle * i) * poly_radius + this.y;
			const x2 = Math.sin(angle * (i + 1)) * poly_radius + this.x;
			const y2 = Math.cos(angle * (i + 1)) * poly_radius + this.y;

			const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
			poly.setAttribute("points", `${this.x},${this.y} ${x1},${y1} ${x2},${y2}`);
			poly.setAttribute("stroke", "black");
			poly.setAttribute("fill", "#cccccc");

			const clip = document.createElementNS("http://www.w3.org/2000/svg", "clipPath");
			clip.appendChild(poly);
			clip.id = `picker_clip_${this.index}_${i}`;

			const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
			circle.setAttribute("cx", this.x);
			circle.setAttribute("cy", this.y);
			circle.setAttribute("r", this.radius);
			circle.setAttribute("fill", "#aaaaaa");
			circle.setAttribute("stroke", "black");
			circle.setAttribute("stroke-width", 2);
			circle.setAttribute("clip-path", `url(#${clip.id})`);
			add_linear(circle, "r", 0, this.radius, this.open_speed);

			svg.appendChild(clip);
			svg.appendChild(circle);

			const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
			line.setAttribute("x1", this.x);
			line.setAttribute("y1", this.y);
			line.setAttribute("x2", rad_x);
			line.setAttribute("y2", rad_y);
			line.setAttribute("stroke", "black");
			line.setAttribute("stroke-width", 2);
			add_linear(line, "x2", this.x, rad_x, this.open_speed);
			add_linear(line, "y2", this.y, rad_y, this.open_speed);

			svg.appendChild(line);

			const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
			text.textContent = this.options[i].title;
			text.setAttribute("x", text_x);
			text.setAttribute("y", text_y);
			text.setAttribute("text-anchor", "middle");
			add_linear(text, "x", this.x, text_x, this.open_speed);
			add_linear(text, "y", this.y, text_y, this.open_speed);

			svg.appendChild(text);

			circle.addEventListener("click", async () => {
				await this.options[i].on_click();
				this.close();
			});

			let circle_expand_anim = null;
			circle.addEventListener("mouseenter", () => {
				const rad = circle.r.animVal.value;
				if (circle_expand_anim) circle_expand_anim.remove();

				circle.setAttribute("r", this.radius * this.expand_amt);
				circle_expand_anim = add_linear(
					circle,
					"r",
					rad,
					this.radius * this.expand_amt,
					this.open_speed,
				);
				circle_expand_anim.beginElement();
			});

			circle.addEventListener("mouseleave", () => {
				const rad = circle.r.animVal.value;
				circle.setAttribute("r", this.radius);
				if (circle_expand_anim) circle_expand_anim.remove();

				circle_expand_anim = add_linear(
					circle,
					"r",
					rad,
					this.radius,
					this.open_speed,
				);
				circle_expand_anim.beginElement();
			});
		}

		this.div.appendChild(svg);
	}
}

function add_linear(node, attrib, a, b, dur) {
	const anim = document.createElementNS("http://www.w3.org/2000/svg", "animate");
	anim.setAttribute("attributeName", attrib);
	anim.setAttribute("from", a);
	anim.setAttribute("to", b);
	anim.setAttribute("dur", `${dur}s`);
	node.appendChild(anim);

	return anim;
}
