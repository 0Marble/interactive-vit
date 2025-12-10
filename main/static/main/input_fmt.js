
export class InputFmt {
	constructor() {
		this.div = document.createElement("div");
		this.div.className = "input_fmt";

		/**
		 * @type {Map<string, any>}
		 */
		this.values = new Map();
	}

	/**
	 * @param {string} text
	 */
	push_text(text) {
		const span = document.createElement("span");
		span.textContent = text;
		this.div.append(span);
	}

	/**
	 * @param {string} name
	 * @param {any} value 
	 * @param {(value: any, name: string) => void} callback
	 */
	push_input(name, value, callback) {
		this.values.set(name, value);

		const input = document.createElement("input");
		input.value = value;
		input.addEventListener("change", () => { callback(input.value, name); });
		this.div.append(input);
	}

	pop() {
		this.div.lastChild.remove();
	}
}
