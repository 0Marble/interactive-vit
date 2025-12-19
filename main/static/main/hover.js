
const body = document.getElementById("body");

export class Hover {
	constructor() {
		this.div = document.createElement("div");
		this.div.className = "hover_div";
		this.div.style = "visibility: hidden;";
		body.append(this.div);
	}

	/**
	 * @param {HTMLElement | string} content
	 */
	set_content(content) {
		while (this.div.firstChild) this.div.firstChild.remove();
		this.div.append(content);
	}

	/**
	 * @param {HTMLElement} node 
	 */
	attatch(node) {
		node.addEventListener("mouseenter", () => {
			const rect = node.getBoundingClientRect();
			this.div.style = `visibility: visible; left: ${rect.left}px; top: ${rect.bottom}px;`;
		});
		node.addEventListener("mouseleave", () => {
			this.div.style = "visibility: hidden;";
		});
	}
}
