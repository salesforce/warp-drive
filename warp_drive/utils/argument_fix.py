import functools
import warnings


class arg_fix:
    """Decorator ensuring backward compatibility when an argument name is
    modified in a function definition.
    """

    def __init__(self, arg_mapping):
        """
        Args:
            arg_mapping (dict): mapping between the function's old argument
                names and the new ones.
        """
        self.arg_mapping = arg_mapping
        self.warning_msg = ("'%s' is a deprecated argument name " +
                            "for the function '%s', use '%s' instead.")

    def __call__(self, f):
        @functools.wraps(f)
        def fixed_f(*args, **kwargs):
            for old_arg, new_arg in self.arg_mapping.items():
                if old_arg in kwargs:
                    #  warn that the function interface has changed:
                    warnings.warn(self.warning_msg %
                        (old_arg, f.__name__, new_arg), DeprecationWarning)
                    # Substitute new_arg to old_arg
                    kwargs[new_arg] = kwargs.pop(old_arg)

            # Call the function with the fixed arguments
            return f(*args, **kwargs)
        return fixed_f
