import * as graph from "./graph.js";
import { Modal } from "./modal.js";

function init_load_from_local_file() {
	const load_button = document.createElement("input");
	load_button.type = "file";
	load_button.accept = "application/json";
	load_button.addEventListener("change", () => {
		let reader = new FileReader();
		reader.readAsText(load_button.files[0]);
		reader.addEventListener("load", async (e) => {
			await graph.Context.wait_for_not_in_eval();
			let src = e.target.result;
			let obj = JSON.parse(src);
			await graph.Context.deserialize(obj);
			await graph.Context.do_eval();
		});
	});
	return load_button;
}

/**
 * @param {Modal} modal 
 * @param {string} name 
 */
function init_load_from_buitlin(modal, name) {
	const button = document.createElement("button");
	button.textContent = name;
	button.addEventListener("click", async () => {
		const params = new URLSearchParams({ name });
		const url = "load_model?" + params.toString();

		const resp = await fetch(url);
		const json = await resp.json();
		await graph.Context.wait_for_not_in_eval();
		await graph.Context.deserialize(json);
		await graph.Context.do_eval();
		modal.close();
	});

	return button;
}

/**
 * @param {Modal} modal 
 */
async function fetch_model_list(modal, list_div) {
	while (list_div.firstChild) list_div.firstChild.remove();

	try {
		const resp = await fetch("list_models");
		if (!resp.ok) throw new Error("something went wrong");
		const json = await resp.json();
		for (const name of json) {
			list_div.appendChild(init_load_from_buitlin(modal, name));
		}
	} catch (err) {
		console.error(err);
		const retry = document.createElement("button");
		retry.textContent = "Retry";
		retry.addEventListener("click", () => {
			fetch_model_list(list_div);
		});
		list_div.appendChild(retry);
	}
}

export async function init_loader() {
	const button = document.createElement("button");
	button.textContent = "Load";
	const modal = new Modal();
	button.addEventListener("click", async () => {
		modal.clear();
		const bottom = document.createElement("div");
		bottom.appendChild(init_load_from_local_file());

		const list = document.createElement("div");
		modal.add_contents(list);
		modal.add_contents(bottom);
		modal.open();

		await fetch_model_list(modal, list);
	});

	return button;
}


