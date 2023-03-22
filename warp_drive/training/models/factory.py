import importlib

# warpdrive reserved models
default_models = {
    "fully_connected": "warp_drive.training.models.fully_connected:FullyConnected",
}


def dynamic_import(model_name: str, model_pool: dict):
    """
    Dynamically import a member from the specified module.
    :param model_name: the name of the model, e.g., fully_connected
    :param model_pool: the dictionary of all available models
    :return: imported class
    """

    if model_name not in model_pool:
        raise ValueError(
            f"model_name {model_name} should be registered in the model factory in the form of,"
            f"e.g. {'fully_connected': 'warp_drive.training.models.fully_connected:FullyConnected' } "
        )
    if ":" not in model_pool[model_name]:
        raise ValueError(
            f"Invalid model path format, expect ':' to separate the path and the object name"
            f"e.g. 'warp_drive.training.models.fully_connected:FullyConnected' "
        )

    module_name, objname = model_pool[model_name].split(":")
    m = importlib.import_module(module_name)
    return getattr(m, objname)


class ModelFactory:

    model_pool = {}

    @classmethod
    def add(cls, model_name: str, model_path: str, object_name: str):
        """

        :param model_name: e.g., "fully_connected"
        :param model_path: e.g., "warp_drive.training.models.fully_connected"
        :param object_name: e.g., "FullyConnected"
        :return:
        :rtype:
        """
        assert model_name not in default_models and model_name not in cls.model_pool, \
            f"{model_name} has already been used by the model collection"

        cls.model_pool.update({model_name: f"{model_path}:{object_name}"})

    @classmethod
    def create(cls, model_name):
        cls.model_pool.update(default_models)
        return dynamic_import(model_name, model_pool=cls.model_pool)
