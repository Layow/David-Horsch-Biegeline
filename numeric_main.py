import time
import sympy as sp
import mpmath as mp


def build_params():
    # Material/geometry parameters (from main1.py)
    return {
        'Dm': 110e-6,
        'Em': 170e9,
        'Dp': 190e-6,
        'Ep': 65e9,
        'Rm': 3.35e-3,
        'Rp': 3e-3,
        'd31': 2e-12,
        'E33': 150e9,
        'p': 100,
        'nu': 0.30,
    }


def build_w1_symbolic():
    # Symbolic variables
    r = sp.Symbol('r', real=True, nonnegative=True)
    Dm, Em, Dp, Ep, Rm, Rp, D = sp.symbols('Dm Em Dp Ep Rm Rp D', real=True, positive=True)
    d31, E33, p, nu = sp.symbols('d31 E33 p nu', real=True)

    # Expression copied from main1.py (region r <= Rm)
    w1 = (1/16)*( (r**2*(-3*(Dm*Em+Dp*Ep)*p*r**2*(-1+nu**2)+(6*(1+nu)*(16*d31*Dm*Dp*(Dm+Dp)*E33*Em*Ep*(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)*Rm**2*(Rm-Rp)*(Rm+Rp)+p*(-1+nu)*(4*Dm**5*Em**3*Rm**6+Dp**5*Ep**3*(Rm**2-Rp**2)**2*(1+nu)*(2*Rm**2+Rp**2*(1+nu))+2*Dm**4*Dp*Em**2*Ep*Rm**2*(2*Rm**2*Rp**2*(-2+nu)-3*Rp**4*(1+nu)+Rm**4*(9+nu))+Dm*Dp**4*Em*Ep**2*(Rm-Rp)*(Rm+Rp)*(Rm**2*Rp**2*(1+nu)**2-Rp**4*(1+nu)**2+4*Rm**4*(3+2*nu))+4*Dm**2*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(3*Ep*Rm**2*(1+nu)+2*Em*(2*Rm**2+Rp**2*(1+nu)))+4*Dm**3*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(2*Ep*Rm**2*(1+nu)+3*Em*(2*Rm**2+Rp**2*(1+nu))))))/(4*Dm**4*Em**2*Rm**4+8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+Dp**4*Ep**2*(Rm**2-Rp**2)**2*(1+nu)**2))/(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)+1/(Dm**3*Em)*(3*(1+nu)*(-(p*((Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)*Rm**4-Dp*Ep*(Dm*(3*Dm**2+6*Dm*Dp+4*Dp**2)*Em+Dp**3*Ep)*Rp**4)*(-1+nu))/(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)-(4*Dp*Ep*Rm**2*Rp**2*(16*d31*Dm**4*(Dm+Dp)*E33*Em**2*Rm**2-p*(Rm-Rp)*(Rm+Rp)*(-1+nu)*(1+nu)*(6*Dm**3*Em*Rm**2+12*Dm**2*Dp*Em*Rm**2+8*Dm*Dp**2*Em*Rm**2+Dp**3*Ep*(Rm-Rp)*(Rm+Rp)*(1+nu)))*(sp.log(Rm)-sp.log(Rp)))/(4*Dm**4*Em**2*Rm**4+8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+Dp**4*Ep**2*(Rm-Rp)**2*(Rm+Rp)**2*(1+nu)**2)))))

    symbols = {
        'r': r,
        'Dm': Dm, 'Em': Em, 'Dp': Dp, 'Ep': Ep,
        'Rm': Rm, 'Rp': Rp, 'd31': d31, 'E33': E33,
        'p': p, 'nu': nu,
    }
    return w1, symbols


def build_integrand_after_substitution():
    params = build_params()
    w1, syms = build_w1_symbolic()

    # Substitute numerical parameters early to reduce complexity
    subs_map = {
        syms['Dm']: params['Dm'], syms['Em']: params['Em'],
        syms['Dp']: params['Dp'], syms['Ep']: params['Ep'],
        syms['Rm']: params['Rm'], syms['Rp']: params['Rp'],
        syms['d31']: params['d31'], syms['E33']: params['E33'],
        syms['p']: params['p'], syms['nu']: params['nu'],
    }
    r = syms['r']
    w1p = w1.subs(subs_map)

    # inner_term = d/dr ( r * d/dr w1 ) / r
    # Compute derivatives symbolically AFTER substitution to keep formulas smaller
    w_r = sp.diff(w1p, r)
    inner = sp.diff(r * w_r, r) / r

    # Integrand U = ∫ (inner^2 * r) dr; we only lambdify the integrand
    integrand = inner**2 * r
    return integrand, r, params


def compute_bending_energy_numeric(mp_dps=80, eps=1e-9):
    t0 = time.time()
    integrand, r, params = build_integrand_after_substitution()

    # Lambdify with mpmath backend for high-precision numerical quadrature
    f = sp.lambdify(r, integrand, 'mpmath')

    mp.mp.dps = mp_dps

    Rm = params['Rm']

    # Integrate from a tiny epsilon to avoid the r=0 singularity from division by r
    try:
        U = mp.quad(f, [eps, Rm])
        method = 'mpmath.quad'
    except Exception:
        # Fallback: composite Simpson via manual refinement
        method = 'fallback_trapz'
        import numpy as np
        for n in (5000, 10000, 20000):
            xs = np.linspace(eps, Rm, n)
            # Use mpmath function over numpy array: vectorize via list comprehension
            ys = np.array([float(f(float(x))) for x in xs])
            U = np.trapz(ys, xs)
        U = mp.mpf(U)

    t1 = time.time()
    return U, (t1 - t0), method


def main():
    print("=== Numeric Bending Energy (U) ===")
    print("Building integrand and integrating numerically ...")
    U, dt, method = compute_bending_energy_numeric()
    print(f"Method: {method}")
    print(f"U (numeric) = {U}")
    print(f"Elapsed: {dt:.2f} s")


if __name__ == '__main__':
    main()

