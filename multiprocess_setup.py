import concurrent.futures
import copy_reg
import types

def _pickle_method(method):
    """Pickle methods of classes for multi-processing.
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    """Unpickle methods of classes for multi-processing.
    """
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def model_wrapper(group):
    """Wrapper function for Model.
    """
    repeat, _ = group
    repeat.init()
    return group
