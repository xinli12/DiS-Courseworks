import numpy as np

from numba_stats import crystalball, expon, norm
from scipy import integrate

class CrystalBall:
    """
    Represents a Crystal Ball distribution with optional truncation.

    Parameters:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
        beta (float): Tail parameter.
        m (float): Power-law tail parameter.
        x_min (float): Minimum truncation value. Defaults to -inf.
        x_max (float): Maximum truncation value. Defaults to inf.
    """
    def __init__(self, mu, sigma, beta, m, x_min=-np.inf, x_max=np.inf):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.m = m
        self.x_min = x_min
        self.x_max = x_max
        # Precompute cdf values
        self.cdf_x_min = self.cdf(x_min)
        self.cdf_x_max = self.cdf(x_max)
        self.normalization = self.cdf_x_max - self.cdf_x_min

    def pdf(self, x):
        return crystalball.pdf(x, self.beta, self.m, self.mu, self.sigma)

    def cdf(self, x):
        return crystalball.cdf(x, self.beta, self.m, self.mu, self.sigma)

    def truncated_pdf(self, x):
        return self.pdf(x) / self.normalization

    def truncated_cdf(self, x):
        return (self.cdf(x) - self.cdf_x_min) / self.normalization

class ExponentialDecay:
    """
    Represents an Exponential Decay distribution with optional truncation.

    Parameters:
        decay_rate (float): Rate of decay.
        x_min (float): Minimum truncation value. Defaults to -inf.
        x_max (float): Maximum truncation value. Defaults to inf.
    """
    def __init__(self, decay_rate, x_min=-np.inf, x_max=np.inf):
        self.decay_rate = decay_rate
        self.x_min = x_min
        self.x_max = x_max
        # Precompute cdf values
        self.cdf_x_min = self.cdf(x_min)
        self.cdf_x_max = self.cdf(x_max)
        self.normalization = self.cdf_x_max - self.cdf_x_min

    def pdf(self, x):
        return expon.pdf(x, 0, 1 / self.decay_rate)

    def cdf(self, x):
        return expon.cdf(x, 0, 1 / self.decay_rate)

    def truncated_pdf(self, x):
        return self.pdf(x) / self.normalization

    def truncated_cdf(self, x):
        return (self.cdf(x) - self.cdf_x_min) / self.normalization

class Normal:
    """
    Represents a Normal distribution with optional truncation.

    Parameters:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
        x_min (float): Minimum truncation value. Defaults to -inf.
        x_max (float): Maximum truncation value. Defaults to inf.
    """
    def __init__(self, mu, sigma, x_min=-np.inf, x_max=np.inf):
        self.mu = mu
        self.sigma = sigma
        self.x_min = x_min
        self.x_max = x_max
        # Precompute cdf values
        self.cdf_x_min = self.cdf(x_min)
        self.cdf_x_max = self.cdf(x_max)
        self.normalization = self.cdf_x_max - self.cdf_x_min

    def pdf(self, x):
        return norm.pdf(x, self.mu, self.sigma)

    def cdf(self, x):
        return norm.cdf(x, self.mu, self.sigma)

    def truncated_pdf(self, x):
        return self.pdf(x) / self.normalization

    def truncated_cdf(self, x):
        return (self.cdf(x) - self.cdf_x_min) / self.normalization

class Uniform:
    """
    Represents a Uniform distribution, requires truncation.

    Parameters:
        x_min (float): Minimum truncation value. Defaults to 0.
        x_max (float): Maximum truncation value. Defaults to 1.
    """
    def __init__(self, x_min=0.0, x_max=1.0):
        self.x_min = x_min
        self.x_max = x_max

    def truncated_pdf(self, x):
        return np.full_like(x, 1 / (self.x_max - self.x_min))

    def truncated_cdf(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)

class SignalBackgroundModel:
    """
    Represents a 2D Signal + Background Model.

    Parameters:
        mu (float): Mean of the signal's Crystal Ball distribution in x.
        sigma (float): Standard deviation of the signal's Crystal Ball distribution in x.
        beta (float): Tail parameter of the signal's Crystal Ball distribution in x.
        m (float): Power-law tail parameter of the signal's Crystal Ball distribution in x.
        decay_rate (float): Decay rate of the signal's Exponential distribution in y.
        mu_bg (float): Mean of the background's Normal distribution in y.
        sigma_bg (float): Standard deviation of the background's Normal distribution in y.
        f (float): Fraction of the signal in the model.
        x_min (float): Minimum x value for truncation.
        x_max (float): Maximum x value for truncation.
        y_min (float): Minimum y value for truncation.
        y_max (float): Maximum y value for truncation.
    """
    def __init__(self, 
                 mu, sigma, beta, m, decay_rate, 
                 mu_bg, sigma_bg, 
                 f, 
                 x_min, x_max, y_min, y_max):
        self.f = f  # Signal fraction
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # Signal Components: Crystal Ball in x, Exponential Decay in y
        self.signal_x = CrystalBall(mu, sigma, beta, m, x_min, x_max)
        self.signal_y = ExponentialDecay(decay_rate, y_min, y_max)

        # Background Components: Uniform in x, Normal in y
        self.background_x = Uniform(x_min, x_max)
        self.background_y = Normal(mu_bg, sigma_bg, y_min, y_max)
    
    def pdf(self, x, y):
        # Signal PDF
        signal_pdf = self.signal_x.truncated_pdf(x) * \
                     self.signal_y.truncated_pdf(y)

        # Background PDF
        background_pdf = self.background_x.truncated_pdf(x) * \
                         self.background_y.truncated_pdf(y)

        # Combined PDF
        return self.f * signal_pdf + (1 - self.f) * background_pdf
    
    def cdf(self, x, y):
        return self.f * self.signal_x.truncated_cdf(x) * \
               self.signal_y.truncated_cdf(y) + \
               (1 - self.f) * self.background_x.truncated_cdf(x) * \
               self.background_y.truncated_cdf(y)
        
    
    def signal_xy(self, x, y):
        return self.signal_x.truncated_pdf(x) * \
               self.signal_y.truncated_pdf(y)

    def background_xy(self, x, y):
        return self.background_x.truncated_pdf(x) * \
               self.background_y.truncated_pdf(y)
    
    # Static methods for the pdfs
    # Don't need to instantiate the class to use these
    @staticmethod
    def sb_pdf(
        mu, sigma, beta, m, decay_rate, mu_bg, sigma_bg, f, 
        x, y,
        x_min, x_max, y_min, y_max
    ):
        signal_x = CrystalBall(mu, sigma, beta, m, x_min, x_max)
        signal_y = ExponentialDecay(decay_rate, y_min, y_max)
        background_x = Uniform(x_min, x_max)
        background_y = Normal(mu_bg, sigma_bg, y_min, y_max)
        return f * signal_x.truncated_pdf(x) * signal_y.truncated_pdf(y) + \
               (1 - f) * background_x.truncated_pdf(x) * background_y.truncated_pdf(y)
    
    @staticmethod
    def x_proj_pdf(
        mu, sigma, beta, m, f, 
        x, 
        x_min, x_max
    ):
        signal_x = CrystalBall(mu, sigma, beta, m, x_min, x_max)
        background_x = Uniform(x_min, x_max)
        signal_x_marginal = f * signal_x.truncated_pdf(x)
        background_x_marginal = (1 - f) * background_x.truncated_pdf(x)
        f_x_marginal = signal_x_marginal + background_x_marginal
        return f_x_marginal
    
    @staticmethod
    def y_proj_pdf(
        mu_bg, sigma_bg, f, decay_rate, 
        y, 
        y_min, y_max
    ):
        signal_y = ExponentialDecay(decay_rate, y_min, y_max)
        background_y = Normal(mu_bg, sigma_bg, y_min, y_max)
        signal_y_marginal = f * signal_y.truncated_pdf(y)
        background_y_marginal = (1 - f) * background_y.truncated_pdf(y)
        f_y_marginal = signal_y_marginal + background_y_marginal
        return f_y_marginal
    
    @staticmethod
    def s_pdf_x(
        mu, sigma, beta, m, 
        x, 
        x_min, x_max
    ):
        signal_x = CrystalBall(mu, sigma, beta, m, x_min, x_max)
        return signal_x.truncated_pdf(x)
    
    @staticmethod
    def b_pdf_x(
        x, 
        x_min, x_max
    ):
        background_x = Uniform(x_min, x_max)
        return background_x.truncated_pdf(x)
    
    @staticmethod
    def s_pdf_y(
        decay_rate, 
        y, 
        y_min, y_max
    ):
        signal_y = ExponentialDecay(decay_rate, y_min, y_max)
        return signal_y.truncated_pdf(y)

    @staticmethod
    def b_pdf_y(
        mu_bg, sigma_bg, 
        y, 
        y_min, y_max
    ):
        background_y = Normal(mu_bg, sigma_bg, y_min, y_max)
        return background_y.truncated_pdf(y)
    
    @staticmethod
    def sb_cdf(
        mu, sigma, beta, m, decay_rate, mu_bg, sigma_bg, f, 
        x, y,
        x_min, x_max, y_min, y_max
    ):
        signal_x = CrystalBall(mu, sigma, beta, m, x_min, x_max)
        signal_y = ExponentialDecay(decay_rate, y_min, y_max)
        background_x = Uniform(x_min, x_max)
        background_y = Normal(mu_bg, sigma_bg, y_min, y_max)
        return f * signal_x.truncated_cdf(x) * signal_y.truncated_cdf(y) + \
               (1 - f) * background_x.truncated_cdf(x) * background_y.truncated_cdf(y)

    @staticmethod
    def s_cdf_y(
        decay_rate, 
        y, 
        y_min, y_max
    ):
        signal_y = ExponentialDecay(decay_rate, y_min, y_max)
        return signal_y.truncated_cdf(y)
    
    def verify_normalization(self):
        result, _ = integrate.dblquad(self.pdf, self.y_min, self.y_max, lambda _: self.x_min, lambda _: self.x_max)
        return result