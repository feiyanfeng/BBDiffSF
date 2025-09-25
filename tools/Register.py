import logging
# import importlib
# import os


class Register:
    def __init__(self, registry_name):
        self.dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable")
        if key is None:
            key = value.__name__
        if key in self.dict:
            logging.warning("Key %s already in registry %s." % (key, self.__name__))
        self.dict[key] = value

    def register_with_name(self, name):
        def register(target):
            def add(key, value):
                self[key] = value
                return value

            if callable(target):
                return add(name, target)
            return lambda x: add(target, x)
        return register

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, key):
        return key in self.dict

    def keys(self):
        return self.dict.keys()



class Registers:
    def __init__(self):
        '''Registers 类本身没有实现具体的功能，它更像是一个用于创建和管理 Register 对象的工厂或容器。而 Register 对象的具体功能则取决于 Register 类的实现。'''
        raise RuntimeError("Registries is not intended to be instantiated")
        '''这个类在初始化时会抛出一个 RuntimeError, 这意味着它并不打算被实例化。'''

    datasets = Register('datasets')
    runners = Register('runners')



# def get_model(args):  # runner_name, config
#     # runner = Registers.runners[runner_name](config)
#     return Registers.runners[args.diffusion.name](args.model, args.diffusion)
#     # return runner


'''
注册器 Registers
调用: runner = Registers.runners[runner_name](config)
比如runner_name='BBDMRunner', 只要在一个类前面加上一行: @Registers.runners.register_with_name('BBDMRunner'),  (方法也适用，并且可以起多个名字)
该类就可以被实例化并返回, config是该类需传入的参数
在不同文件下时需要导入: from Register import Registers
用eval好像也可以实现, 但这个显得更高端
'''