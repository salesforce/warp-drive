import atexit
from warp_drive.utils.device_context import make_current_context

# Initialize torch and CUDA context

context = make_current_context()
device = context.get_device()
atexit.register(context.pop)