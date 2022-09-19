import logging


class EnvironmentRegistrar:
    """
    Environment Registrar Class
    """

    _cpu_envs = {}
    _cuda_envs = {}
    _numba_envs = {}
    _customized_cuda_env_src_paths = {
        "pycuda": {},
        "numba": {},
    }

    def add(self, env_backend="cpu", cuda_env_src_path=None):
        if not isinstance(env_backend, list):
            env_backends = [env_backend.lower()]
        else:
            env_backends = [d.lower() for d in env_backend]

        def add_wrapper(cls):
            cls_name = cls.name.lower()

            for backend in env_backends:
                if backend == "cpu":
                    if cls_name not in self._cpu_envs:
                        self._cpu_envs[cls_name] = cls
                    else:
                        raise Exception(
                            f"CPU environment {cls_name} already registered, "
                            f"you may need to go to your env class to "
                            f"define a different class name "
                        )
                elif backend == "pycuda":
                    if cls_name not in self._cuda_envs:
                        self._cuda_envs[cls_name] = cls
                    else:
                        raise Exception(
                            f"PyCUDA environment {cls_name} already registered, "
                            f"you may need to go to your env class to "
                            f"define a different class name "
                        )
                    if cuda_env_src_path is not None:
                        self.add_cuda_env_src_path(
                            cls_name, cuda_env_src_path, env_backend
                        )
                elif backend == "numba":
                    if cls_name not in self._numba_envs:
                        self._numba_envs[cls_name] = cls
                    else:
                        raise Exception(
                            f"Numba environment {cls_name} already registered, "
                            f"you may need to go to your env class to "
                            f"define a different class name "
                        )
                    if cuda_env_src_path is not None:
                        self.add_cuda_env_src_path(
                            cls_name, cuda_env_src_path, env_backend
                        )
                else:
                    raise Exception("Invalid device: only support CPU and CUDA/GPU")
            return cls

        return add_wrapper

    def get(self, name, env_backend="cpu"):
        name = name.lower()
        if env_backend == "cpu":
            if name not in self._cpu_envs:
                raise Exception(f"CPU environment {name} not found ")
            logging.info(f"returning CPU environment {name} ")
            return self._cpu_envs[name]
        elif env_backend == "pycuda":
            if name not in self._cuda_envs:
                raise Exception(f"PyCUDA environment {name} not found ")
            logging.info(f"returning CUDA environment {name} ")
            return self._cuda_envs[name]
        elif env_backend == "numba":
            if name not in self._numba_envs:
                raise Exception(f"Numba environment {name} not found ")
            logging.info(f"returning Numba environment {name} ")
            return self._numba_envs[name]
        else:
            raise Exception("Invalid backend: only support CPU, PyCUDA/CUDA and Numba")

    def add_cuda_env_src_path(self, name, cuda_env_src_path, env_backend="pycuda"):
        """
        Register the customized environment for developers.
        The FunctionManager will then be able to include the
        environment source code in the compilation.
        :param name: name of your customized environment
        :param cuda_env_src_path: ABSOLUTE path to the customized
            environment source code in CUDA
        """
        name = name.lower()
        if name in self._customized_cuda_env_src_paths[env_backend]:
            logging.warning(
                f"EnvironmentRegistrar has already registered an "
                f"environment path called {name} but we will re-register it "
                f"by overwriting the previous source code path"
            )
        if env_backend == "pycuda":
            assert (
                cuda_env_src_path.rsplit(".", 1)[1] == "cu"
            ), "the customized environment is expected to be a CUDA source code (*.cu)"
        elif env_backend == "numba":
            assert (
                "/" not in cuda_env_src_path
            ), "the customized environment is expected to be a valid PYTHONPATH"
        else:
            raise Exception(f"unknown env_backend: {env_backend}")
        self._customized_cuda_env_src_paths[env_backend][name] = cuda_env_src_path

    def get_cuda_env_src_path(self, name, env_backend="pycuda"):
        name = name.lower()
        assert env_backend in ("pycuda", "numba"), f"unknown env_backend: {env_backend}"
        return self._customized_cuda_env_src_paths[env_backend].get(name, None)

    def has_env(self, name, env_backend="cpu"):
        name = name.lower()
        if env_backend == "cpu":
            return name in self._cpu_envs
        if env_backend in ("pycuda", "cuda", "gpu"):
            return name in self._cuda_envs
        if env_backend == "numba":
            return name in self._numba_envs
        raise Exception("Invalid device: only support CPU and CUDA/GPU")


env_registrar = EnvironmentRegistrar()
