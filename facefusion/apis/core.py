#!-*- coding:utf-8 -*-
# FileName:

import mimetypes
import os
import uuid

import requests
from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import RedirectResponse
import uvicorn
from starlette.responses import FileResponse, JSONResponse

from facefusion import state_manager, core, logger

app = FastAPI()
router = APIRouter()

INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/input'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/output'))
# SOURCE_DIR = os.path.join(INPUT_DIR, 'source')
# TARGET_DIR = os.path.join(INPUT_DIR, 'target')
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@router.get("/")
async def root():
	return RedirectResponse(url='/docs')


@router.get("/healthy")
async def healthy():
	return {'code': 200}


@app.exception_handler(RuntimeError)
async def handle_cuda_error(_, exc):
	raise exc


@app.exception_handler(AssertionError)
async def assert_handler(_, exc: AssertionError):
	print(f'捕获assert: {str(exc)}')
	return JSONResponse(
		status_code=400,
		content={'code': 400, 'msg': str(exc)}
	)


@app.exception_handler(Exception)
async def exception_handler(_, exc: Exception):
	print(f'捕获异常：{str(exc)}')
	if isinstance(exc, SystemExit):
		raise exc
	return JSONResponse(
		status_code=500,
		content={'code': 500, 'msg': '服务器异常', 'content': str(exc)}
	)


@router.post("/deep_swapper")
async def deep_swapper(
	args: Request,
):
	args = await args.json()

	uid = str(uuid.uuid4())
	input_file = download(args.pop('file'), f'{uid}.{args.get("type", "mp4")}')
	output_file = os.path.join(OUTPUT_DIR, f'{uid}.{args.get("type", "mp4")}')

	state_manager.set_item('target_path', input_file)
	state_manager.set_item('output_path', output_file)
	state_manager.set_item('processors', ['deep_swapper'])
	state_manager.set_item('deep_swapper_model', f'custom/{args.get("dfm_id")}')
	apply_manager(args)

	logger.info(f'Start conditional_process: {uid}', 'api.deep_swapper')
	core.conditional_process()
	logger.info(f'End conditional_process: {uid}', 'api.deep_swapper')

	return FileResponse(
		path=output_file,
		media_type=get_media_type(output_file),
		filename=f'{uid}.mp4',
		headers={"Content-Disposition": "inline"}
	)


def download(url: str, file_name: str) -> str:
	assert url.startswith('http') or url.startswith('https'), 'error file, need one url'
	logger.info(f'Start download file[{file_name}]: {url}', 'api.download')
	response = requests.get(url, stream=True)
	response.raise_for_status()  # 检查请求是否成功

	path = os.path.join(INPUT_DIR, file_name)
	with open(path, 'wb') as file:
		for chunk in response.iter_content(chunk_size=8192):
			file.write(chunk)
	logger.info(f'File[{file_name}] downloaded.', 'api.download')
	return path


def get_media_type(file_path):
	# 猜测文件的MIME类型
	content_type, _ = mimetypes.guess_type(file_path)

	if not content_type:
		content_type = 'application/octet-stream'  # 二进制流，默认未知类型

	return content_type


def apply_manager(items: dict):
	for k, n in items.items():
		if k in default_args:
			state_manager.set_item(k, n)


default_args = {
	'face_detector_model': os.getenv('face_detector_model', 'yoloface'),
	'face_detector_angles': [int(item) for item in os.getenv('face_detector_angles', '0').split(',')],
	'face_detector_size': os.getenv('face_detector_size', '640x640'),
	'face_detector_score': float(os.getenv('face_detector_score', 0.5)),
	'face_landmarker_model': os.getenv('face_landmarker_model', '2dfan4'),
	'face_landmarker_score': float(os.getenv('face_landmarker_score', 0.5)),
	'face_selector_mode': os.getenv('face_selector_mode', 'reference'),
	'face_selector_order': os.getenv('face_selector_order', 'large-small'),
	'face_selector_gender': os.getenv('face_selector_gender'),
	'face_selector_race': os.getenv('face_selector_race'),
	'face_selector_age_start': os.getenv('face_selector_age_start'),
	'face_selector_age_end': os.getenv('face_selector_age_end'),
	'reference_face_position': int(os.getenv('reference_face_position', 0)),
	'reference_face_distance': float(os.getenv('reference_face_position', 0.6)),
	'reference_frame_number': int(os.getenv('reference_frame_number', 0)),
	'face_occluder_model': os.getenv('face_occluder_model', 'xseg_1'),
	'face_parser_model': os.getenv('face_occluder_model', 'bisenet_resnet_34'),
	'face_mask_types': os.getenv('face_mask_types', 'box').split(','),
	'face_mask_blur': float(os.getenv('face_mask_blur', 0.3)),
	'face_mask_padding': (0, 0, 0, 0),
	'face_mask_regions': ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses', 'nose', 'mouth',
						  'upper-lip', 'lower-lip'],
	'trim_frame_start': os.getenv('trim_frame_start'),
	'trim_frame_end': os.getenv('trim_frame_end'),
	'temp_frame_format': os.getenv('temp_frame_format', 'png'),
	'keep_temp': os.getenv('keep_temp'),
	'output_image_quality': int(os.getenv('output_image_quality', 80)),
	'output_image_resolution': os.getenv('output_image_resolution'),
	'output_audio_encoder': os.getenv('output_audio_encoder', 'aac'),
	'output_video_encoder': os.getenv('output_video_encoder', 'libx264'),
	'output_video_preset': os.getenv('output_video_preset', 'veryfast'),
	'output_video_quality': int(os.getenv('output_video_quality', 80)),
	'output_video_resolution': os.getenv('output_video_resolution', '1080x1920'),
	'output_video_fps': int(os.getenv('output_video_fps', 30)),
	'skip_audio': os.getenv('skip_audio'),
	'processors': os.getenv('processors', 'deep_swapper').split(','),
	'deep_swapper_model': os.getenv('deep_swapper_model', 'iperov/elon_musk_224'),
	'deep_swapper_morph': int(os.getenv('deep_swapper_morph', 80)),
	'age_modifier_model': os.getenv('age_modifier_model', 'styleganex_age'),
	'age_modifier_direction': int(os.getenv('age_modifier_direction', 0)),
	'expression_restorer_model': os.getenv('expression_restorer_model', 'live_portrait'),
	'expression_restorer_factor': int(os.getenv('expression_restorer_factor', 80)),
	'face_debugger_items': os.getenv('face_debugger_items', 'face-landmark-5/68,face-mask').split(','),
	'face_editor_model': os.getenv('face_editor_model', 'live_portrait'),
	'face_editor_eyebrow_direction': int(os.getenv('face_editor_eyebrow_direction', 0)),
	'face_editor_eye_gaze_horizontal': int(os.getenv('face_editor_eye_gaze_horizontal', 0)),
	'face_editor_eye_gaze_vertical': int(os.getenv('face_editor_eye_gaze_vertical', 0)),
	'face_editor_eye_open_ratio': int(os.getenv('face_editor_eye_open_ratio', 0)),
	'face_editor_lip_open_ratio': int(os.getenv('face_editor_lip_open_ratio', 0)),
	'face_editor_mouth_grim': int(os.getenv('face_editor_mouth_grim', 0)),
	'face_editor_mouth_pout': int(os.getenv('face_editor_mouth_pout', 0)),
	'face_editor_mouth_purse': int(os.getenv('face_editor_mouth_purse', 0)),
	'face_editor_mouth_smile': int(os.getenv('face_editor_mouth_smile', 0)),
	'face_editor_mouth_position_horizontal': int(os.getenv('face_editor_mouth_position_horizontal', 0)),
	'face_editor_mouth_position_vertical': int(os.getenv('face_editor_mouth_position_vertical', 0)),
	'face_editor_head_pitch': int(os.getenv('face_editor_head_pitch', 0)),
	'face_editor_head_yaw': int(os.getenv('face_editor_head_yaw', 0)),
	'face_editor_head_roll': int(os.getenv('face_editor_head_roll', 0)),
	'face_enhancer_model': os.getenv('face_enhancer_model', 'gfpgan_1.4'),
	'face_enhancer_blend': int(os.getenv('face_enhancer_blend', 80)),
	'face_enhancer_weight': float(os.getenv('face_enhancer_weight', 1.0)),
	'face_swapper_model': os.getenv('face_swapper_model', 'inswapper_128_fp16'),
	'face_swapper_pixel_boost': os.getenv('face_swapper_pixel_boost', '128x128'),
	'frame_colorizer_model': os.getenv('frame_colorizer_model', 'ddcolor'),
	'frame_colorizer_blend': int(os.getenv('frame_colorizer_blend', 100)),
	'frame_colorizer_size': os.getenv('frame_colorizer_size', '256x256'),
	'frame_enhancer_model': os.getenv('frame_enhancer_model', 'span_kendata_x4'),
	'frame_enhancer_blend': int(os.getenv('frame_enhancer_blend', 80)),
	'lip_syncer_model': os.getenv('lip_syncer_model', 'wav2lip_gan_96'),
	'download_providers': os.getenv('download_providers', 'github,huggingface').split(','),
	'execution_providers': os.getenv('execution_providers', 'cpu'),
	'temp_path': os.getenv('temp_path', '/temp'),
	'execution_thread_count': int(os.getenv('execution_thread_count', 4)),
	'execution_queue_count': int(os.getenv('execution_queue_count', 1)),
	'log_level': os.getenv('log_level', 'info'),
}
apply_manager(default_args)


def launch():
	app.include_router(router)
	uvicorn.run(app, host="0.0.0.0", port=int(state_manager.get_item('port')))
