class CustomizedEnvironmentRegistrar:

    _customized_envs = {}

    def register_environment(self, env_name, cuda_env_src_path):
        """
        Register the customized environment for developers.
        The FunctionManager will then be able to include the
        environment source code in the compilation.
        :param env_name: name of your customized environment
        :param cuda_env_src_path: ABSOLUTE path to the customized
            environment source code in CUDA
        """

        if env_name in self._customized_envs:
            print(
                f"[WARNING]: EnvironmentRegistrar has already registered an "
                f"environment called {env_name} but we will re-register it "
                f"by overwriting the previous source code path"
            )
        assert (
            "cu" == cuda_env_src_path.rsplit(".", 1)[1]
        ), "the customized environment is expected to be a CUDA source code (*.cu)"
        self._customized_envs[env_name] = cuda_env_src_path

    def get_env_directory(self, env_name):
        return self._customized_envs.get(env_name, None)

    @property
    def get_all_customized_envs(self):
        return self._customized_envs
