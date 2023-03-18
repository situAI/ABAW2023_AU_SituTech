class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = name + '_' + suffix

        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj


    def get(self, name, suffix='soulwalker'):
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix)
            print(f'Name {name} is not found, use name: {name}_{suffix}!')
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def register(self, obj=None, suffix=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                if name not in self.keys():
                    self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        if name not in self.keys():
            self._do_register(name, obj, suffix)

    def build(self, info, suffix='soulwalker'):

        name = info['name'] if isinstance(info, dict) else info
        
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix)
            print(f'Name {name} is not found, use name: {name}_{suffix}!')
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        
        if isinstance(info, dict) and 'args' in info.keys():
            args = info['args'].copy()
            return ret(**args)
        else:
            return ret()

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()
    
    def clear(self):
        self._obj_map = {}
        

MODEL_REGISTRY = Registry('model')
DATASET_REGISTRY =  Registry('data')

BUILD_REGISTRY = Registry('build')
SOLVER_REGISTRY = Registry('solver')

LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
OPTIMIZER_REGISTRY = Registry('optimizer')
LR_SCHEDULER_REGISTRY = Registry('lr_scheduler')