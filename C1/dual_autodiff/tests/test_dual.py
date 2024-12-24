from dual_autodiff import Dual
import pytest
import math

def test_init():
    dual = Dual(1.1, 1.6)
    assert dual.real == 1.1
    assert dual.dual == 1.6

# use pytest.mark.parametrize decorator to test basic operations
@pytest.mark.parametrize("a, b, operation, expected_real, expected_dual", [
    (Dual(1, 1.4), Dual(2.5, 5), 
     lambda x, y: x + y, 3.5, 6.4),
    (Dual(1, 1.4), Dual(2.5, 5), 
     lambda x, y: x - y, -1.5, -3.6),
    (Dual(1, 1.4), Dual(2.5, 5), 
     lambda x, y: x * y, 2.5, 1*5+1.4*2.5),
    (Dual(1, 1.4), Dual(2.5, 5), 
     lambda x, y: x / y, 1/2.5, 1.4/2.5-1*5/(2.5**2)),
    (Dual(1, 1.4), 2, 
     lambda x, y: x ** y, 1, 1.4*2*1**(2-1)),
])
def test_basic_operations(a, b, operation, expected_real, expected_dual):
    result = operation(a, b)
    assert result.real == pytest.approx(expected_real, rel=1e-5) # relative tolerance
    assert result.dual == pytest.approx(expected_dual, rel=1e-5)

# test the functions
@pytest.mark.parametrize("func, x, expected_real, expected_dual", [
    (Dual.sin, Dual(1.5, 1), math.sin(1.5), math.cos(1.5)),
    (Dual.cos, Dual(1.5, 1), math.cos(1.5), -math.sin(1.5)),
    (Dual.tan, Dual(1.5, 1), math.tan(1.5), 1/math.cos(1.5)**2),
    (Dual.log, Dual(1.5, 1), math.log(1.5), 1/1.5),
    (Dual.exp, Dual(1.5, 1), math.exp(1.5), math.exp(1.5)),
])
def test_functions(func, x, expected_real, expected_dual):
    result = func(x)
    assert result.real == pytest.approx(expected_real, rel=1e-5)
    assert result.dual == pytest.approx(expected_dual, rel=1e-5)


def test_string_representations():
    dual = Dual(1.1, 1.6)
    assert str(dual) == "Dual(real=1.1, dual=1.6)"
    assert repr(dual) == "Dual(real=1.1, dual=1.6)"

def test_arithmetic_error_handling():
    dual = Dual(1.0, 1.0)
    # Test addition with invalid type
    with pytest.raises(TypeError, match="Operand must be a Dual instance or a float/int"):
        dual + "invalid"
    
    # Test subtraction with invalid type
    with pytest.raises(TypeError, match="Operand must be a Dual instance or a float/int"):
        dual - "invalid"
    
    # Test multiplication with invalid type
    with pytest.raises(TypeError, match="Operand must be a Dual instance or a float/int"):
        dual * "invalid"
    
    # Test division with invalid type
    with pytest.raises(TypeError, match="Operand must be a Dual instance or a float/int"):
        dual / "invalid"
    
    # Test power with invalid type
    with pytest.raises(TypeError, match="Operand must be a float or int"):
        dual ** Dual(2.0, 1.0)

def test_reverse_operations():
    dual = Dual(2.0, 1.0)
    
    # Test right addition
    result1 = 3.0 + dual
    assert result1.real == 5.0
    assert result1.dual == 1.0

    # Test right subtraction
    result2 = 3.0 - dual
    assert result2.real == 1.0
    assert result2.dual == -1.0
    
    # Test negation
    result3 = -dual
    assert result3.real == -2.0
    assert result3.dual == -1.0
    
    # Test right multiplication
    result4 = 3.0 * dual
    assert result4.real == 6.0
    assert result4.dual == 3.0

def test_chain_rule():
    x = Dual(1.0)
    f = (x * x).sin()  # sin(x²)
    assert f.real == pytest.approx(math.sin(1.0), rel=1e-5)
    assert f.dual == pytest.approx(2*math.cos(1.0), rel=1e-5)

@pytest.mark.parametrize("x", [0.1, 1.0, 2.0, 5.0])
def test_exp_derivative(x):
    d = Dual(x)
    result = d.exp()
    expected = math.exp(x)
    assert result.real == pytest.approx(expected, rel=1e-5)
    assert result.dual == pytest.approx(expected, rel=1e-5)
