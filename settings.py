APP_NAME = 'hitsignal'

## Path
import os

dirname = os.path.dirname(__file__)
base_path = dirname + ""

# tmp
TEMP_FOLDER = f"{base_path}/tmp"
# Log
LOG_FOLDER = f"{base_path}/log"

# models
MODEL_FOLDER = f"{base_path}/models"

# data
DATA_FOLDER = f"{base_path}/data"

# Faust
FAUST_APP_NAME = f"{APP_NAME}_stream_app"
FAUST_BROKER = "kafka://localhost:9092"

# Sanic Web App
WEB_APP_STATIC_FOLDER = f'{base_path}/ui/static'

# Socket io Settings
SOCKET_IO_MESSAGE_BROKER = "redis://"
