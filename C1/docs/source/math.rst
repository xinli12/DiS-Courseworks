.. _mathematical_background:

Mathematical Background
=======================

This section provides a comprehensive explanation of the mathematical concepts behind automatic differentiation using dual numbers, including theoretical foundations, proofs, and practical applications.

Dual Numbers: A Rigorous Foundation
-----------------------------------

Definition and Algebraic Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A dual number is defined as an ordered pair :math:`(a, b)`, typically written as :math:`a + bε`, where:

* :math:`a, b \in \mathbb{R}` (the real numbers)
* :math:`ε` is the dual unit with the property :math:`ε^2 = 0`


Properties and Operations
^^^^^^^^^^^^^^^^^^^^^^^^^

Basic arithmetic operations follow from the algebraic structure:

Addition and Subtraction
""""""""""""""""""""""""

.. math::

   (a + bε) \pm (c + dε) = (a \pm c) + (b \pm d)ε

This operation is componentwise and preserves the dual number structure.

Multiplication
""""""""""""""

.. math::

   (a + bε)(c + dε) &= ac + (bc + ad)ε + bdε^2 \\
                    &= ac + (bc + ad)ε \quad \text{(since } ε^2 = 0\text{)}

The multiplication rule is crucial for understanding how derivatives emerge naturally.

Division
""""""""

For :math:`c \neq 0`:

.. math::

   \frac{a + bε}{c + dε} = \frac{a}{c} + \frac{bc - ad}{c^2}ε

This is derived using the multiplication rule and the fact that:

.. math::

   (c + dε)^{-1} = \frac{1}{c} - \frac{d}{c^2}ε

Power Rules
"""""""""""

For :math:`n \in \mathbb{N}`:

.. math::

   (a + bε)^n = a^n + nba^{n-1}ε

For :math:`r \in \mathbb{R}` and :math:`a > 0`:

.. math::

   (a + bε)^r = a^r + rba^{r-1}ε

Automatic Differentiation Theory
--------------------------------

Forward Mode Fundamentals
^^^^^^^^^^^^^^^^^^^^^^^^^

The key insight of forward-mode automatic differentiation is that dual numbers naturally encode derivative information:

.. math::

   f(a + bε) = f(a) + f'(a)bε

This is not just a convenient formula but follows from the Taylor series expansion and the property of ε:

.. math::

   f(a + bε) &= f(a) + f'(a)(bε) + \frac{f''(a)}{2!}(bε)^2 + \cdots \\
             &= f(a) + f'(a)bε \quad \text{(since } ε^2 = 0\text{)}

Chain Rule and Composition
^^^^^^^^^^^^^^^^^^^^^^^^^^

The chain rule emerges naturally from dual number arithmetic. For :math:`h(x) = f(g(x))`:

.. math::

   h(a + bε) &= f(g(a + bε)) \\
             &= f(g(a) + g'(a)bε) \\
             &= f(g(a)) + f'(g(a))g'(a)bε

This demonstrates why dual numbers are so powerful for automatic differentiation: they handle the chain rule automatically through their algebraic structure.

Chain Rule Example
""""""""""""""""""

To illustrate this more concretely, let's examine a specific example. Consider:

.. math::

   g(x) &= x^2 \\
   f(x) &= \sin(x) \\
   h(x) &= f(g(x)) = \sin(x^2)

Using dual numbers to compute the derivative:

.. math::

   g(a + bε) &= (a + bε)^2 = a^2 + 2abε \\
   f(g(a + bε)) &= f(a^2 + 2abε) \\
                &= \sin(a^2) + 2ab\cos(a^2)ε

Therefore:

.. math::

   h'(a) = 2a\cos(a^2)

This matches what we would get using the traditional chain rule: :math:`h'(x) = \frac{d}{dx}\sin(x^2) = \cos(x^2) \cdot \frac{d}{dx}(x^2) = 2x\cos(x^2)`.

Multiple Composition
""""""""""""""""""""

The power of dual numbers extends naturally to multiple function compositions. For :math:`k(x) = f(g(h(x)))`:

.. math::

   k(a + bε) &= f(g(h(a + bε))) \\
             &= f(g(h(a) + h'(a)bε)) \\
             &= f(g(h(a)) + g'(h(a))h'(a)bε) \\
             &= f(g(h(a))) + f'(g(h(a)))g'(h(a))h'(a)bε

This automatically gives us :math:`k'(a) = f'(g(h(a))) \cdot g'(h(a)) \cdot h'(a)`, which is exactly the chain rule for three functions.


Elementary Functions
--------------------

Transcendental Functions
^^^^^^^^^^^^^^^^^^^^^^^^

The derivatives of elementary functions are encoded in their dual number extensions:

Exponential
"""""""""""

.. math::

   \exp(a + bε) = e^a + be^aε

Proof sketch: Use the Taylor series for exp(x) and the property of ε.

Logarithm
"""""""""

For :math:`a > 0`:

.. math::

   \log(a + bε) = \log(a) + \frac{b}{a}ε

Proof: Differentiate :math:`\exp(\log(x)) = x` using the chain rule.

Trigonometric Functions
^^^^^^^^^^^^^^^^^^^^^^^

Sine and Cosine
"""""""""""""""

.. math::

   \sin(a + bε) &= \sin(a) + b\cos(a)ε \\
   \cos(a + bε) &= \cos(a) - b\sin(a)ε

These follow from the Taylor series expansions and the property of ε.

Tangent and Other Functions
"""""""""""""""""""""""""""

.. math::

   \tan(a + bε) &= \tan(a) + \frac{b}{\cos^2(a)}ε \\
   \arctan(a + bε) &= \arctan(a) + \frac{b}{1 + a^2}ε
