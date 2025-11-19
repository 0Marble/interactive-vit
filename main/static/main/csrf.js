const csrf_token = document.querySelector('[name=csrfmiddlewaretoken]')?.value;

export function get_csrf_token() {
	return csrf_token;
}
