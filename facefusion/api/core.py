import json
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, APIRouter, BackgroundTasks, UploadFile, File, Form, status
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from facefusion.core import conditional_process
from facefusion import process_manager, globals

app = FastAPI()
router = APIRouter()

INPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/output')
SOURCE_DIR = os.path.join(INPUT_DIR, 'source')
TARGET_DIR = os.path.join(INPUT_DIR, 'target')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def update_global_variables(params):
	for var_name, value in params.items():
		if value is not None:
			if hasattr(globals, var_name):
				setattr(globals, var_name, value)


def save_file(file: UploadFile, file_path: str):
	if not os.path.exists(Path(file_path).parent):
		os.makedirs(Path(file_path).parent)
	with open(file_path, "wb") as f:
		f.write(file.file.read())


def apply_args():
	from facefusion.vision import is_image, is_video, detect_image_resolution, detect_video_resolution, \
		detect_video_fps, create_image_resolutions, create_video_resolutions, pack_resolution
	from facefusion.normalizer import normalize_fps
	if is_image(globals.target_path):
		output_image_resolution = detect_image_resolution(globals.target_path)
		output_image_resolutions = create_image_resolutions(output_image_resolution)
		if globals.output_image_resolution in output_image_resolutions:
			globals.output_image_resolution = globals.output_image_resolution
		else:
			globals.output_image_resolution = pack_resolution(output_image_resolution)
	if is_video(globals.target_path):
		output_video_resolution = detect_video_resolution(globals.target_path)
		output_video_resolutions = create_video_resolutions(output_video_resolution)
		if globals.output_video_resolution in output_video_resolutions:
			globals.output_video_resolution = globals.output_video_resolution
		else:
			globals.output_video_resolution = pack_resolution(output_video_resolution)
	if globals.output_video_fps or is_video(globals.target_path):
		globals.output_video_fps = normalize_fps(globals.output_video_fps) or detect_video_fps(globals.target_path)


@router.get("/")
async def index():
	return 'Hello World!'


@router.post("/submit")
async def process_frames(
	background_tasks: BackgroundTasks,
	options: str = Form(None),
	sources: List[UploadFile] = File(...),
	target: UploadFile = File(...),
):
	if not (process_manager.is_pending() or process_manager.is_stopping()):
		return JSONResponse(content={"message": f'当前存在进程，状态为：{process_manager.get_process_state()}'},
							status_code=status.HTTP_201_CREATED)

	try:
		options = options or '{}'
		options = json.loads(options)
	except Exception as e:
		return JSONResponse(content={"message": f'无法解析options为JSON：{str(e)}'},
							status_code=status.BAD_REQUEST)

	if not isinstance(options, dict):
		return JSONResponse(content={"message": f'无法解析options参数为dict，当前格式为：{type(options).__name__}'},
							status_code=status.BAD_REQUEST)
	update_global_variables(options)
	# TODO：切换处理器

	source_paths = []
	for file in sources:
		file_path = os.path.join(SOURCE_DIR, Path(file.filename).name)
		save_file(file, file_path)
		source_paths.append(file_path)

	target_path = os.path.join(TARGET_DIR, Path(target.filename).name)
	save_file(target, target_path)

	globals.source_paths = source_paths
	globals.target_path = target_path
	globals.output_path = os.path.join(OUTPUT_DIR, Path(target.filename).name)
	apply_args()

	background_tasks.add_task(conditional_process)
	return {'code': 200}


@router.get("/state")
async def process_state() -> dict:
	return {'data': process_manager.get_process_state()}


@router.get("/download")
async def process_download():
	if process_manager.is_processing() or process_manager.is_analysing():
		return JSONResponse(content={"message": f'当前进程未结束，状态为：{process_manager.get_process_state()}'},
							status_code=status.HTTP_201_CREATED)
	if process_manager.is_pending() and not globals.output_path:
		return JSONResponse(content={"message": f'当前无进程，状态为：{process_manager.get_process_state()}'},
							status_code=status.HTTP_201_CREATED)

	return FileResponse(
		path=globals.output_path,
		media_type="application/octet-stream",  # 可以根据文件类型进行调整
		headers={"Content-Disposition": f"attachment; filename={Path(globals.output_path).name}"}
	)


def launch():
	app.include_router(router)
	uvicorn.run(app, host="0.0.0.0", port=globals.port)
