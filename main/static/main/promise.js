
/**
 *
 * Gets resolved when the `resolve` method is called
 */
export class CallbackPromise {
	constructor() {
		this.callbacks = [];
		this.error_handler = null;
	}

	then(then_callback) {
		this.callbacks.push(then_callback);
		return this;
	}
	catch(catch_callback) {
		this.error_handler = catch_callback;
	}

	trigger() {
		let value = undefined;
		try {
			for (const callback of this.callbacks) {
				value = callback(value);
			}
		} catch (err) {
			if (this.error_handler) {
				this.error_handler(err);
			} else {
				throw err;
			}
		}
	}
}
