const body = document.getElementById("workspace");

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
		});
		this.div.addEventListener("click", (e) => {
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
		const poly_radius = (this.expand_amt + 1.0) * this.radius / Math.cos(angle / 2);

		for (let i = 0; i < this.options.length; i++) {
			const rad1_x1 = Math.sin(angle * i) * this.radius + this.x;
			const rad1_y1 = Math.cos(angle * i) * this.radius + this.y;
			const rad1_x2 = Math.sin(angle * i) * this.radius * this.expand_amt + this.x;
			const rad1_y2 = Math.cos(angle * i) * this.radius * this.expand_amt + this.y;
			const rad2_x1 = Math.sin(angle * (i + 1)) * this.radius + this.x;
			const rad2_y1 = Math.cos(angle * (i + 1)) * this.radius + this.y;
			const rad2_x2 = Math.sin(angle * (i + 1)) * this.radius * this.expand_amt + this.x;
			const rad2_y2 = Math.cos(angle * (i + 1)) * this.radius * this.expand_amt + this.y;

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
			const circle_anim = LinearAnimation.add_and_run(circle, "r", 0, this.radius, this.open_speed);

			svg.appendChild(clip);
			svg.appendChild(circle);

			const line1 = document.createElementNS("http://www.w3.org/2000/svg", "line");
			line1.setAttribute("x1", this.x);
			line1.setAttribute("y1", this.y);
			line1.setAttribute("x2", rad1_x1);
			line1.setAttribute("y2", rad1_y1);
			line1.setAttribute("stroke", "black");
			line1.setAttribute("stroke-width", 2);
			const line1_x = LinearAnimation.add_and_run(line1, "x2", this.x, rad1_x1, this.open_speed);
			const line1_y = LinearAnimation.add_and_run(line1, "y2", this.y, rad1_y1, this.open_speed);
			svg.appendChild(line1);

			const line2 = document.createElementNS("http://www.w3.org/2000/svg", "line");
			line2.setAttribute("x1", this.x);
			line2.setAttribute("y1", this.y);
			line2.setAttribute("x2", rad2_x1);
			line2.setAttribute("y2", rad2_y1);
			line2.setAttribute("stroke", "black");
			line2.setAttribute("stroke-width", 2);
			const line2_x = LinearAnimation.add_and_run(line2, "x2", this.x, rad2_x1, this.open_speed);
			const line2_y = LinearAnimation.add_and_run(line2, "y2", this.y, rad2_y1, this.open_speed);
			svg.appendChild(line2);

			const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
			text.textContent = this.options[i].title;
			text.setAttribute("x", text_x);
			text.setAttribute("y", text_y);
			text.setAttribute("text-anchor", "middle");
			text.style = "pointer-events: none;";
			LinearAnimation.add_and_run(text, "x", this.x, text_x, this.open_speed);
			LinearAnimation.add_and_run(text, "y", this.y, text_y, this.open_speed);

			svg.appendChild(text);

			circle.addEventListener("click", async () => {
				await this.options[i].on_click(this.x, this.y);
				this.close();
			});

			circle.addEventListener("mouseenter", () => {
				circle_anim.run(circle_anim.current(), this.radius * this.expand_amt, this.open_speed);
				line1_x.run(line1_x.current(), rad1_x2, this.open_speed);
				line1_y.run(line1_y.current(), rad1_y2, this.open_speed);
				line2_x.run(line2_x.current(), rad2_x2, this.open_speed);
				line2_y.run(line2_y.current(), rad2_y2, this.open_speed);
			});

			circle.addEventListener("mouseleave", () => {
				circle_anim.run(circle_anim.current(), this.radius, this.open_speed);
				line1_x.run(line1_x.current(), rad1_x1, this.open_speed);
				line1_y.run(line1_y.current(), rad1_y1, this.open_speed);
				line2_x.run(line2_x.current(), rad2_x1, this.open_speed);
				line2_y.run(line2_y.current(), rad2_y1, this.open_speed);
			});
		}

		this.div.appendChild(svg);
	}
}

class LinearAnimation {
	static add_and_run(node, attrib, a, b, dur) {
		const anim = new LinearAnimation(node, attrib);
		anim.run(a, b, dur);
		return anim;
	}

	constructor(node, attrib) {
		this.anim = document.createElementNS("http://www.w3.org/2000/svg", "animate");
		this.anim.setAttribute("attributeName", attrib);
		node.appendChild(this.anim);
		this.node = node;
		this.attrib = attrib;
	}

	current() {
		return this.node[this.attrib].animVal.value;
	}

	run(a, b, dur) {
		this.node.setAttribute(this.attrib, b);
		this.anim.setAttribute("from", a);
		this.anim.setAttribute("to", b);
		this.anim.setAttribute("dur", `${dur}s`);
		this.anim.beginElement();
	}
}
