class EnvironmentRegistrar:
    """
    Environment Registry Class
    """

    _cpu_envs = {}
    _cuda_envs = {}
    _customized_cuda_env_src_paths = {}

    def add(self, device="cpu", cuda_env_src_path=None):
        if not isinstance(device, list):
            devices = [device.lower()]
        else:
            devices = [d.lower() for d in device]

        def add_wrapper(cls):
            cls_name = cls.name.lower()

            for device in devices:
                if device == "cpu":
                    if cls_name not in self._cpu_envs:
                        self._cpu_envs[cls_name] = cls
                    else:
                        raise Exception(f"CPU environment {cls_name} already registered, you may need to go to your env class to "
                                        f"define a different class name ")
                elif device == "cuda" or device == "gpu":
                    if cls_name not in self._cuda_envs:
                        self._cuda_envs[cls_name] = cls
                    else:
                        raise Exception(f"CUDA environment {cls_name} already registered, you may need to go to your env class to "
                                        f"define a different class name ")
                    if cuda_env_src_path is not None:
                        self.add_cuda_env_src_path(cls_name, cuda_env_src_path)
                else:
                    raise Exception("Invalid device: only support CPU and CUDA/GPU")
            return cls
        return add_wrapper

    def get(self, name, use_cuda=False):
        name = name.lower()
        if use_cuda is False:
            if name not in self._cpu_envs:
                raise Exception(f"CPU environment {name} not found ")
            print(f"return CPU environment {name} ")
            return self._cpu_envs[name]
        else:
            if name not in self._cuda_envs:
                raise Exception(f"CUDA environment {name} not found ")
            print(f"return CUDA environment {name} ")
            return self._cuda_envs[name]

    def add_cuda_env_src_path(self, name, cuda_env_src_path):
        """
        Register the customized environment for developers.
        The FunctionManager will then be able to include the
        environment source code in the compilation.
        :param name: name of your customized environment
        :param cuda_env_src_path: ABSOLUTE path to the customized
            environment source code in CUDA
        """
        name = name.lower()
        if name in self._customized_cuda_env_src_paths:
            print(
                f"[WARNING]: EnvironmentRegistrar has already registered an "
                f"environment path called {name} but we will re-register it "
                f"by overwriting the previous source code path"
            )
        assert (
            cuda_env_src_path.rsplit(".", 1)[1] == "cu"
        ), "the customized environment is expected to be a CUDA source code (*.cu)"
        assert (
            name in self._cuda_envs
        ), "the customized environment is not yet added in the registrar"
        self._customized_cuda_env_src_paths[name] = cuda_env_src_path

    def get_cuda_env_src_path(self, name):
        name = name.lower()
        return self._customized_cuda_env_src_paths.get(name, None)

    def has_env(self, name, device="cpu"):
        name = name.lower()
        if device == "cpu":
            return name in self._cpu_envs
        elif device == "cuda" or device == "gpu":
            return name in self._cuda_envs
        else:
            raise Exception("Invalid device: only support CPU and CUDA/GPU")


env_registry = EnvironmentRegistrar()