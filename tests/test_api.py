import json

import requests

base_url = 'http://127.0.0.1:8000'


def submit():
	end_point = 'submit'

	sources_files = [
		'./images/elon-1.jpg',
		'./images/elon-2.jpg',
		'./images/elon-3.jpg',
	]
	sources = []
	for item in sources_files:
		with open(item, 'rb') as source_f:
			source_data = source_f.read()
		sources.append(('sources', (item, source_data)))

	target_file = './images/mark.jpg'
	with open(target_file, 'rb') as target_f:
		target_data = target_f.read()
	target = ('target', (target_file, target_data))

	files = [
		*sources,
		target,
	]
	options = {

	}

	resp = requests.post(f'{base_url}/{end_point}', files=files, data={'options': json.dumps(options)})
	print(resp.status_code, resp.text)


def state():
	end_point = 'state'
	resp = requests.get(f'{base_url}/{end_point}')
	print(resp.status_code, resp.text)


def download(output_path):
	end_point = 'download'
	resp = requests.get(f'{base_url}/{end_point}')
	if resp.status_code != 200:
		print(resp.status_code, resp.text)
		return

	with open(output_path, 'wb') as f:
		f.write(resp.content)
	print(f'Successfully downloaded file to {output_path}')


# submit()
# state()
# download('./images/result.jpg')
