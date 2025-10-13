import numpy as np
# import tensorflow as tf

def basic_stuff():
    # x = 1
    # x = 1.0
    # x = "1.0"
    y = 'Hello, "Nils"!'

    # print(type(x))
    # print(type(y), y)

    # for(i=0;i<10;i++) -> C++/C/Java
    for i in range(len(y)):
        print(i, y[i])

def myfunction(x, y=0, a=0, b=0, c=0):
    print('Hello world!', x, y)
    return 99, 'Hi', [a, b, c]

# v1, _, _ = myfunction(y=2, x='Nils', c=3)
# v1, v2, v3 = myfunction(y=2, x='Nils', c=3)
# ret = myfunction(y=2, x='Nils', c=3)

# for r in ret:
#     print(r)

# basic_stuff()

# d = {'Nils': ['a', 'b', 1, 2.0], 'ABC': 30, 3: 'Hello'}
# d['Martin'] = 32
# for k in d:
#     print(k, d[k])

x = [1.2, 3.4, 5.6, 7.8, 1.234, 5.678, 20.4]
a = np.array(x)
print(x, type(x))
print(a, type(a))

print(a.mean())
print(a.std())
print(a.min())
print(a.max())
print(np.quantile(a, 0.75))

