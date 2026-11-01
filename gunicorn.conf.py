# Gunicorn automatically loads this file, so these settings apply even when
# Render's dashboard start command is just "gunicorn app:app".
workers = 1
threads = 4
timeout = 120
graceful_timeout = 30
worker_tmp_dir = "/dev/shm"
accesslog = "-"
