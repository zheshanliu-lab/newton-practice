import pytest
import numpy as np
import math

import newton

## Important: structure of tests assumes a dictionary with an 'x'
## key as the output. 

def test_cosine_function():
    assert np.isclose(newton.optimize(np.cos, 2.95), math.pi)

def test_invalid_function():
    with pytest.raises(TypeError):
        newton.optimize(123, 0)

def test_bad_input():
    with pytest.raises(TypeError):   
        newton.optimize(2.95, np.cos)
    ## Ideally, our function would raise the exception with a useful message.
    with pytest.raises(TypeError, match='f must be a function'):
        newton.optimize(2.95, np.cos)

## How to check that a warning is (correctly) emitted:
## def test_warning():
##    with pytest.warns(UserWarning, match='greater'):
##        newton.optimize(...., ....)
