from libc.math cimport sin, cos, log, exp, tan, pow
from libc.math cimport M_PI

cdef class Dual:
    """
    A class representing dual numbers for automatic differentiation.
    """
    cdef public double real
    cdef public double dual

    def __init__(self, double real, double dual=1.0):
        """
        Initialize a Dual instance.
        """
        self.real = real
        self.dual = dual
    
    def __repr__(self):
        return f"Dual(real={self.real}, dual={self.dual})"
    
    def __str__(self):
        return f"Dual(real={self.real}, dual={self.dual})"

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, (float, int)):
            return Dual(self.real + other, self.dual)
        raise TypeError("Operand must be a Dual instance or a float/int")
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, (float, int)):
            return Dual(self.real - other, self.dual)
        raise TypeError("Operand must be a Dual instance or a float/int")
    
    def __rsub__(self, other):
        if isinstance(other, (float, int)):
            return Dual(other - self.real, -self.dual)
        raise TypeError("Operand must be a Dual instance, float or int")
    
    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real * other.real, 
                        self.real * other.dual + other.real * self.dual)
        elif isinstance(other, (float, int)):
            return Dual(self.real * other, self.dual * other)
        raise TypeError("Operand must be a Dual instance or a float/int")
    
    def __rmul__(self, other):
        return self.__mul__(other)  # Reuse the logic in __mul__

    def __truediv__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real / other.real,
                        self.dual / other.real - self.real * other.dual / (other.real * other.real))
        elif isinstance(other, (float, int)):
            return Dual(self.real / other, self.dual / other)
        raise TypeError("Operand must be a Dual instance or a float/int")
    
    def __pow__(self, other):
        return Dual(pow(self.real, other), 
                    other * pow(self.real, other - 1) * self.dual)
    
    cpdef Dual sin(self):
        return Dual(sin(self.real), self.dual * cos(self.real))
    
    cpdef Dual cos(self):
        return Dual(cos(self.real), -self.dual * sin(self.real))
    
    cpdef Dual log(self):
        if self.real <= 0:
            raise ValueError("Cannot compute logarithm of non-positive number")
        return Dual(log(self.real), self.dual / self.real)
    
    cpdef Dual exp(self):
        cdef double exp_real = exp(self.real)
        return Dual(exp_real, self.dual * exp_real)
    
    cpdef Dual tan(self):
        cdef double cos_x = cos(self.real)
        if abs(cos_x) < 1e-10:
            raise ValueError("Tangent undefined at this point")
        return Dual(tan(self.real), self.dual / (cos_x * cos_x))
