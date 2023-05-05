import numpy as np
from scipy.optimize import curve_fit
import unittest

#==============================================================================
# PLUMMER PROFILE
#==============================================================================

# Parameters -> Profile -------------------------------------------------------

def plummer(space_range:np.ndarray[float], max:float, radius:float, slope:float) -> np.ndarray[float]:
    """_summary_

    Args:
        space_range (np.ndarray[float]): A vector of space values for which the profile is computed.
        max (float): The maximum value of the profile.
        radius (float): The radius of the profile.
        slope (float): The slope of the profile.

    Returns:
        np.ndarray[float]: The profile.
    """
    profile = 3/(4 * np.pi * radius**3) * (1 + np.abs(space_range)**slope / radius**slope)**(-5/2)
    profile = max * profile / np.max(profile)
    return profile

# Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

class Test_plummer(unittest.TestCase):
    def test_plummer(self):
        space_range = np.linspace(-2, 2, 64, endpoint=True)
        max = 1e4
        radius = 1
        slope = 2
        profile = plummer(space_range, max, radius, slope)
        self.assertEqual(np.max(profile), max)
        # TODO: add tests

# Profile -> Parameters -------------------------------------------------------

def fit_plummer(space_range:np.ndarray[float], profile:np.ndarray[float]) -> tuple[float, float, float]:
    """_summary_

    Args:
        space_range (np.ndarray[float]): A vector of space values corresponding to the profile.
        profile (np.ndarray[float]): The profile.

    Returns:
        tuple[float, float, float]: The parameters of the profile.
    """
    param = [1e4, 2, 1]
    (max, radius, slope), _ = curve_fit(plummer, space_range, profile, p0=param, maxfev = 100000)
    return max, radius, slope

# Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

class Test_fit_plummer(unittest.TestCase):
    def test_fit_plummer(self):
        space_range = np.linspace(-25, 25, 64, endpoint=True)
        max = 1e4
        radius = 2
        slope = 1
        profile = plummer(space_range, max, radius, slope)
        max_fit, radius_fit, slope_fit = fit_plummer(space_range, profile)
        self.assertAlmostEqual(max_fit, max, places=0)
        self.assertAlmostEqual(radius_fit, radius, places=0)
        self.assertAlmostEqual(slope_fit, slope, places=0)

#==============================================================================
# MAIN
#==============================================================================

if __name__ == "__main__":
    unittest.main()