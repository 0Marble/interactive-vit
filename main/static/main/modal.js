const body_elem = document.getElementById("body");

export class Modal {
	constructor() {
		this.div = document.createElement("div");
		body_elem.appendChild(this.div);
		this.div.className = "modal_bg";
		this.body = document.createElement("div");
		this.body.className = "modal_body";
		this.div.appendChild(this.body);

		const button = document.createElement("button");
		button.textContent = "x";
		button.addEventListener("click", () => {
			this.close();
		});
		this.body.appendChild(button);

		this.contents = document.createElement("div");
		this.body.appendChild(this.contents);

		this.close();
	}

	add_contents(contents) {
		this.contents.appendChild(contents);
	}

	clear() {
		while (this.contents.firstChild) this.contents.firstChild.remove();
	}

	open() {
		this.div.style = "";
	}

	close() {
		this.div.style = "display: none;";
	}
}

