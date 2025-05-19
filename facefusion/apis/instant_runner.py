from time import sleep

from facefusion import process_manager, state_manager
from facefusion.args import collect_step_args
from facefusion.core import process_step
from facefusion.filesystem import is_directory
from facefusion.jobs import job_helper, job_manager, job_runner, job_store
from facefusion.temp_helper import clear_temp_directory
from facefusion.typing import Args
from facefusion.uis.ui_helper import suggest_output_path


def run():
	step_args = collect_step_args()
	output_path = step_args.get('output_path')

	if is_directory(step_args.get('output_path')):
		step_args['output_path'] = suggest_output_path(step_args.get('output_path'), state_manager.get_item('target_path'))
	if job_manager.init_jobs(state_manager.get_item('jobs_path')):
		create_and_run_job(step_args)
		state_manager.set_item('output_path', output_path)


def create_and_run_job(step_args : Args) -> bool:
	job_id = job_helper.suggest_job_id('api')

	for key in job_store.get_job_keys():
		state_manager.sync_item(key) #type:ignore

	return job_manager.create_job(job_id) and job_manager.add_step(job_id, step_args) and job_manager.submit_job(job_id) and job_runner.run_job(job_id, process_step)


def stop():
	process_manager.stop()


def clear():
	while process_manager.is_processing():
		sleep(0.5)
	if state_manager.get_item('target_path'):
		clear_temp_directory(state_manager.get_item('target_path'))
