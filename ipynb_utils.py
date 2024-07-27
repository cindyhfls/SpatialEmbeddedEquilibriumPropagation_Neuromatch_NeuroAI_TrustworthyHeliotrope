import logging

from main import load_default_config, parse_shell_args
from lib import config




def ipynb_main(_argv, exp):
	# Parse shell arguments as input configuration
	user_config = parse_shell_args(_argv[1:])

	# Load default parameter configuration from file for the specified energy-based model
	cfg = load_default_config(user_config["energy"])

	# Overwrite default parameters with user configuration where applicable
	cfg.update(user_config)

	# Setup global logger and logging directory
	config.setup_logging(cfg["energy"] if cfg["energy"] else "bp" + "_" + cfg["c_energy"] + "_" + cfg["dataset"],
						dir=cfg['log_dir'])
	logging.info(f"Cmd: python {' '.join(_argv)}")
	logging.info(f"Device:\n{config.device}")
	# Run the script using the created parameter configuration
	_result = exp(cfg)
	# Close logging
	# logging.info('log file is open')
	_log_file_name = config.get_log_name()
	# https://stackoverflow.com/a/61457520/8612123
	logger = logging.getLogger()
	while logger.hasHandlers():
		logger.removeHandler(logger.handlers[0])
	logging.shutdown()

	# logging.info('log file should be closed')
	# Return name for log to re-use in plotting function
	return {'log':_log_file_name, 'result':_result}
