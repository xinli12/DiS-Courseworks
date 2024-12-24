import math


class Dual:
    """
    A class representing dual numbers for automatic differentiation.

    Attributes:
        real (float): Real part of the dual number.
        dual (float): Dual part of the dual number. 

    Examples:
        >>> x = Dual(2.0, 1.0)  # Represents 2.0 + 1.0ε
        >>> y = x * x
        >>> print(y.real) # Value of x^2 at x = 2
        4.0
        >>> print(y.dual) # Derivative of x^2 at x = 2
        4.0

    """
    def __init__(self, real: float, dual: float = 1.0):
        """
        Initialise a Dual instance.

        Parameters:
        - real (float): Real part of the dual number.
        - dual (float): Dual part of the dual number. Defaults to 1.0.

        Examples:
            >>> x = Dual(2.0, 1.0)
            >>> x
            Dual(real=2.0, dual=1.0)
            >>> y = Dual(3.0)
            >>> y
            Dual(real=3.0, dual=1.0)
        
        """
        self.real = real
        self.dual = dual
    
    def __repr__(self):
        """Return a string representation of the dual number.

        Returns:
            str: A string in the format "Dual(real=x, dual=y)"

        Examples:
            >>> x = Dual(2.0, 1.0)
            >>> x
            Dual(real=2.0, dual=1.0)
        """
        return f"Dual(real={self.real}, dual={self.dual})"
    
    def __str__(self):
        """Return a string representation of the dual number.

        Returns:
            str: A string in the format "Dual(real=x, dual=y)"

        Examples:
            >>> x = Dual(2.0, 1.0)
            >>> print(x)
            Dual(real=2.0, dual=1.0)
        """
        return f"Dual(real={self.real}, dual={self.dual})"
    
    def __add__(self, other):
        """
        Overload the + operator to add two dual numbers or a dual number and a float/int.

        Examples:
            >>> x1 = Dual(2.0, 1.0)
            >>> y1 = x1 + 1.0
            >>> y1
            Dual(real=3.0, dual=1.0)
            >>> x2 = Dual(1.5, 0.5)
            >>> y2 = x1 + x2
            >>> y2
            Dual(real=3.5, dual=1.5)
        """
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, (float, int)):
            return Dual(self.real + other, self.dual)
        raise TypeError("Operand must be a Dual instance or a float/int")
    
    def __radd__(self, other):
        """
        Overload the + operator for right addition: float/int + Dual.

        Examples:
            >>> x1 = Dual(1.0, 0.5)
            >>> y1 = 2.0 + x1
            >>> y1
            Dual(real=3.0, dual=0.5)
        """
        return self.__add__(other)


    def __sub__(self, other):
        """
        Overload the - operator to subtract two dual numbers or a dual number and a float/int.

        Examples:
            >>> x1 = Dual(2.0, 1.0)
            >>> y1 = x1 - 1.0
            >>> y1
            Dual(real=1.0, dual=1.0)
            >>> x2 = Dual(1.5, 0.5)
            >>> y2 = x1 - x2
            >>> y2
            Dual(real=0.5, dual=0.5)
        """
        if isinstance(other, Dual): 
            return Dual(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, (float, int)):
            return Dual(self.real - other, self.dual)
        raise TypeError("Operand must be a Dual instance or a float/int")
    
    def __rsub__(self, other):
        """
        Overload the reverse subtraction operator for float/int minus Dual.

        Args:
            other (float or int): The minuend.

        Returns:
            Dual: The result of the subtraction.

        Raises:
            TypeError: If other is not a float or int.

        Examples:
            >>> x = Dual(2.0, 1.0)
            >>> y = 3.0 - x
            >>> y
            Dual(real=1.0, dual=-1.0)
        """
        if isinstance(other, (float, int)):
            return Dual(other - self.real, -self.dual)
        raise TypeError("Operand must be a Dual instance, float or int")
    
    def __neg__(self):
        """
        Overload the unary - operator to negate a dual number.
        
        Returns:
            Dual: A new Dual number with negated real and dual parts.
        
        Examples:
            >>> x = Dual(2.0, 1.0)
            >>> -x
            Dual(real=-2.0, dual=-1.0)
        """
        return Dual(-self.real, -self.dual)
    
    def __mul__(self, other):
        """
        Overload the * operator to multiply two dual numbers or a dual number and a float/int.

        Examples:
            >>> x1 = Dual(2.0, 1.0)
            >>> y1 = x1 * 2.0
            >>> y1
            Dual(real=4.0, dual=2.0)
            >>> x2 = Dual(1.5, 0.5)
            >>> y2 = x1 * x2
            >>> y2
            Dual(real=3.0, dual=2.5)
        """
        if isinstance(other, Dual): 
            return Dual(self.real * other.real, 
                        self.real * other.dual + other.real * self.dual)
        elif isinstance(other, (float, int)):
            return Dual(self.real * other, self.dual * other)
        raise TypeError("Operand must be a Dual instance or a float/int")
    
    def __rmul__(self, other):
        """
        Overload the * operator for right multiplication: float/int * Dual.

        Examples:
            >>> x1 = Dual(2.0, 1.0)
            >>> y1 = 2.0 * x1
            >>> y1
            Dual(real=4.0, dual=2.0)
        """
        return self.__mul__(other)  # Reuse the logic in __mul__

    def __truediv__(self, other):
        """
        Overload the / operator to divide two dual numbers or a dual number and a float/int.

        Examples:
            >>> x1 = Dual(2.0, 1.0)
            >>> y1 = x1 / 2.0
            >>> y1
            Dual(real=1.0, dual=0.5)
            >>> x2 = Dual(1.5, 0.5)
            >>> y2 = x1 / x2
            >>> y2
            Dual(real=1.3333333333333333, dual=0.2222222222222222)
        """
        if isinstance(other, Dual):
            return Dual(self.real / other.real,
                        self.dual / other.real - self.real * other.dual / other.real**2)
        elif isinstance(other, (float, int)):
            return Dual(self.real / other, self.dual / other)
        raise TypeError("Operand must be a Dual instance or a float/int")
    
    
    def __pow__(self, other):
        """
        Overload the ** operator to raise a dual number to a power.

        Examples:
            >>> x1 = Dual(2.0, 1.0)
            >>> y1 = x1 ** 2
            >>> y1
            Dual(real=4.0, dual=4.0)
            >>> x2 = Dual(1.5, 0.5)
        """
        if isinstance(other, (float, int)):
            return Dual(self.real**other, 
                        other * self.real**(other - 1) * self.dual)
        raise TypeError("Operand must be a float or int")
    
    def sin(self):
        """
        Compute the sine of a dual number.
        """
        return Dual(math.sin(self.real), 
                    self.dual * math.cos(self.real))
    
    def cos(self):
        """
        Compute the cosine of a dual number.
        """
        return Dual(math.cos(self.real),
                    - self.dual * math.sin(self.real))
    
    def log(self):
        """
        Compute the natural logarithm of a dual number.

        Returns:
            Dual: The natural logarithm of the dual number.

        Raises:
            ValueError: If attempting to compute log of a non-positive real part.

        Examples:
            >>> x = Dual(1.0, 1.0)
            >>> y = x.log()
            >>> print(f"{y.real:.6f}, {y.dual:.6f}")
            0.000000, 1.000000
        """
        if self.real <= 0:
            raise ValueError("Cannot compute logarithm of non-positive number")
        return Dual(math.log(self.real), self.dual / self.real)
    
    def exp(self):
        """
        Compute the exponential of a dual number.

        Returns:
            Dual: The exponential of the dual number.

        Examples:
            >>> x = Dual(0.0, 1.0)
            >>> y = x.exp()
            >>> print(f"{y.real:.6f}, {y.dual:.6f}")
            1.000000, 1.000000
        """
        exp_real = math.exp(self.real)
        return Dual(exp_real, self.dual * exp_real)
    
    def tan(self):
        """
        Compute the tangent of a dual number.

        Returns:
            Dual: The tangent of the dual number.

        Raises:
            ValueError: If cosine is zero (tangent undefined).

        Examples:
            >>> x = Dual(math.pi/4, 1.0)
            >>> y = x.tan()
            >>> print(f"{y.real:.6f}, {y.dual:.6f}")
            1.000000, 2.000000
        """
        cos_x = math.cos(self.real)
        if abs(cos_x) < 1e-10:
            raise ValueError("Tangent undefined at this point")
        return Dual(math.tan(self.real), 
                   self.dual / (cos_x * cos_x))
    
