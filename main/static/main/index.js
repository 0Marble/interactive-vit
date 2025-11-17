let csrf_token = "";

function init() {
	csrf_token = document.querySelector('[name=csrfmiddlewaretoken]')?.value;

	const div = document.getElementById("graph_div");
	const input = new InputImage(div);
}

class InputImage {
	constructor(parent_div) {
		this.div = document.createElement("div");
		const input = document.createElement("input");
		input.type = "file";
		input.accept = "image/*";

		this.canvas = document.createElement("canvas");
		this.canvas.className = "image_view_canvas";
		this.ctx = this.canvas.getContext("2d");
		this.div.appendChild(input);
		this.div.appendChild(this.canvas);

		input.addEventListener("change", () => {
			const file = input.files[0];
			const img = document.createElement("img");
			img.src = URL.createObjectURL(file);

			img.addEventListener("load", () => {
				this.canvas.width = img.width;
				this.canvas.height = img.height;
				this.ctx.drawImage(img, 0, 0, img.width, img.height);
				this.on_image_loaded(file);
			});
		});

		parent_div.appendChild(this.div)
	}

	on_image_loaded(img_file) {
		const form = new FormData();
		form.append("image_input", img_file);
		fetch("/input", {
			method: "POST", 
			body: form,
			headers: {
				'X-CSRFToken': csrf_token,
			},
		})
			.then(data => console.log("Upload: ", data))
			.catch(err => console.log("Failed: ", err));

	}
}
