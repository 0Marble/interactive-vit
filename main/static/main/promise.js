export class CallbackPromise {
	constructor() {
		this.res = null;
		this.rej = null;
		this.promise = new Promise((res, rej) => {
			this.res = res;
			this.rej = rej;
		});
	}

	resolve(val) {
		this.res(val);
		return this;
	}

	trigger(val) {
		return this.resolve(val);
	}

	reject(val) {
		this.rej(val);
		return this;
	}

	then(on_fulfilled, on_rejected) {
		return this.promise.then(on_fulfilled, on_rejected);
	}
	catch(on_rejected) {
		return this.promise.catch(on_rejected);
	}
	finally(on_finally) {
		return this.promise.finally(on_finally);
	}
}

export class AsyncLock {
	constructor(open) {
		this.promise = new CallbackPromise();
		if (open) this.promise.trigger();
	}

	async acquire() {
		await this.promise;
		this.promise = new CallbackPromise();
	}

	release() {
		this.promise.trigger();
	}
}
