import logging


class CUDAEnvironmentContext:
    """
    Environment Context class to manage APIs for the communication
    between EnvWrapper class and the Environment class
    """

    def __init__(self):
        self.cuda_data_manager = None
        self.cuda_function_manager = None
        self.cuda_step = None
        self.cuda_step_function_feed = None

    def initialize_step_function_context(
        self,
        cuda_data_manager,
        cuda_function_manager,
        cuda_step_function_feed,
        step_function_name,
    ):
        try:
            self.cuda_data_manager = cuda_data_manager
            self.cuda_function_manager = cuda_function_manager
            self.cuda_function_manager.initialize_functions([step_function_name])
            self.cuda_step = self.cuda_function_manager.get_function(step_function_name)
            self.cuda_step_function_feed = cuda_step_function_feed
            return True
        except Exception as err:
            logging.error(err)
            return False
