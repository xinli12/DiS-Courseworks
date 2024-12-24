import numpy as np
from scipy import stats, integrate


class CrystalBall:
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
        return stats.crystalball.pdf(x, self.beta, self.m, loc=self.mu, scale=self.sigma)
    
    def cdf(self, x):
        return stats.crystalball.cdf(x, self.beta, self.m, loc=self.mu, scale=self.sigma)
    
    def ppf(self, x):
        return stats.crystalball.ppf(x, self.beta, self.m, loc=self.mu, scale=self.sigma)

    def truncated_pdf(self, x):
        return self.pdf(x) / self.normalization
    
    def truncated_cdf(self, x):
        return (self.cdf(x) - self.cdf_x_min) / (self.cdf_x_max - self.cdf_x_min)

class ExponentialDecay:
    def __init__(self, decay_rate, x_min=-np.inf, x_max=np.inf):
        self.decay_rate = decay_rate
        self.x_min = x_min
        self.x_max = x_max
        # Precompute cdf values
        self.cdf_x_min = self.cdf(x_min)
        self.cdf_x_max = self.cdf(x_max)
        self.normalization = self.cdf_x_max - self.cdf_x_min

    def pdf(self, x):
        return stats.expon.pdf(x, scale=1/self.decay_rate)

    def cdf(self, x):
        return stats.expon.cdf(x, scale=1/self.decay_rate)

    def ppf(self, x):
        return stats.expon.ppf(x, scale=1/self.decay_rate)

    def truncated_pdf(self, x):
        return self.pdf(x) / self.normalization
    
    def truncated_cdf(self, x):
        return (self.cdf(x) - self.cdf_x_min) / (self.cdf_x_max - self.cdf_x_min)

class Normal:
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
        return stats.norm.pdf(x, loc=self.mu, scale=self.sigma)
    
    def cdf(self, x):
        return stats.norm.cdf(x, loc=self.mu, scale=self.sigma)
    
    def ppf(self, x):
        return stats.norm.ppf(x, loc=self.mu, scale=self.sigma)

    def truncated_pdf(self, x):
        return self.pdf(x) / self.normalization
    
    def truncated_cdf(self, x):
        return (self.cdf(x) - self.cdf_x_min) / (self.cdf_x_max - self.cdf_x_min)

class Uniform:
    def __init__(self, x_min=-np.inf, x_max=np.inf):
        self.x_min = x_min
        self.x_max = x_max

    def truncated_pdf(self, x):
        return 1 / (self.x_max - self.x_min)
    
    def truncated_cdf(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)

class SignalBackgroundModel:
    def __init__(self, signal_params, background_params, f, x_min, x_max, y_min, y_max):
        self.f = f  # Signal fraction
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # Signal Components
        self.signal_x = CrystalBall(*signal_params['crystal_ball'], x_min, x_max)
        self.signal_y = ExponentialDecay(signal_params['lambda'], y_min, y_max)

        # Background Components
        self.background_x = Uniform(x_min, x_max)
        self.background_y = Normal(*background_params['truncated_normal'], y_min, y_max)

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

    def verify_normalization(self):
        result, _ = integrate.dblquad(self.pdf, self.y_min, self.y_max, lambda _: self.x_min, lambda _: self.x_max)
        return result