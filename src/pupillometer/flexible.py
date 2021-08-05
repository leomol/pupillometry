# 2020-02-18. Leonardo Molina.
# 2021-01-23. Last modified.

class Flexible:
    def __init__(self, *args, **kwpairs):
        # Flexible(a=1, b=2, ...)
        # Flexible('a', 1, 'b', 2, ...)
        self.__keys = []
        self.set(*args, **kwpairs)
    
    def keys(self):
        return (key for key in self.__keys)
    
    def items(self):
        return ((key, getattr(self, key)) for key in self.keys())
    
    def set(self, *args, **kwpairs):
        # set(a=1, b=2, ...)
        # set('a', 1, 'b', 2, ...)
        # set(['a', 'b', ...], 1)
        # set(['a', 'b', ...])
        # set('a')
        target = {}
        if len(kwpairs) > 0:
            target = kwpairs
        elif len(args) > 0:
            if len(args) == 1 or isinstance(args[0], (list, tuple)):
                target = dict.fromkeys(args[0], None if len(args) == 1 else args[1])
            elif len(args) > 1:
                target = {args[2 * i]:args[2 * i + 1] for i in range(0, int(len(args) / 2))}
        self.__set(target)
    
    def __set(self, target):
        for key, value in target.items():
            setattr(self, key, value)
            if key not in self.__keys:
                self.__keys.append(key)

    def __setattr__(self, key, value): # !! potentially slow.
        if hasattr(self, "_%s__keys" % type(self).__name__):
            if key not in self.__keys:
                self.__keys.append(key)
            super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)
        
if __name__ == "__main__":
    flex = Flexible()
    flex.set(a=1)
    print(flex.a)
    flex.set(a=1, b=2)
    print(flex.a)
    print(flex.b)
    flex.set('c', 3, 'd', 4)
    print(flex.c)
    print(flex.d)
    flex.set(['e', 'f'], 5)
    print(flex.e)
    print(flex.f)
    flex.g = 6
    print(flex.g)
    flex.set(['h', 'i'])
    print(flex.h)
    print(flex.i)
    flex.set('j')
    print(flex.j)
    
    flex.z = 7
    print(dict(flex.items()))
    flex2 = Flexible()
    flex2.set(**dict(flex.items()))
    print(dict(flex2.items()))