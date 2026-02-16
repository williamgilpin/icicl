from scipy.integrate import solve_ivp
class Lorenz96:
    """
    Lorenz 96 model with constant forcing
    """

    def __init__(self, n=5, f=8.0):
        self.f = f
        self.n = n

    def rhs(self, t, X):

        Xdot = np.zeros_like(X)
        Xdot[2:-1] = (X[3:] - X[:-3]) * X[1:-2] - X[2:-1] + self.f
        Xdot[0] = (X[1] - X[-2]) * X[-1] - X[0] + self.f
        Xdot[1] = (X[2] - X[-1]) * X[0] - X[1] + self.f
        Xdot[-1] = (X[0] - X[-3]) * X[-2] - X[-1] + self.f

        return Xdot
    
    def __call__(self, *args, **kwargs):
        return self.rhs(*args, **kwargs)