import * as dataflow from "./dataflow";

export class Node {
	/**
	 * @param {dataflow.Node} n 
	 */
	constructor(n) {
		this.n = n;
		this.div = document.createElement("div");
		const header = this.init_header();
		const footer = this.init_footer();
	}

	init_header() {
	}
}
