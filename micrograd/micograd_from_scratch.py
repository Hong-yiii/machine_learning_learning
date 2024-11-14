import math
import numpy as np
import matplotlib.pyplot as plt


class Value:
    def _init_(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.prev = set(_children)
        self.op = _op
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ## if other is not of a class Value, wrap the int in a Value LOL
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        ## this propagates the gradient backwards call this function to basically "copy" the grad of the output of the addition function, as the grad "flows through" addition functions

        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            other.grad += self.data * out.grad
            self.grad += other.grad * out.grad
        out.backward = _backward

        return out
    
    def __rmul__(self,other): #other * self
        return self * other
    # defending against int * value error

    def __truediv__(self, other): #self/other
        return self * other **-1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int,float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data **(other - 1) * out.grad
        out.backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, self, 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
            out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        ## this is a TOPO sort, ordering it from the "head" of the tree to the "roots", where we

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        ## we call .backwards from the head downwards, elucidating in the correct order 




