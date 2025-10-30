import time
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import sympy as sp
from sympy.integrals.manualintegrate import manualintegrate


def build_symbols():
    r = sp.Symbol('r', real=True, positive=True)
    Dm, Em, Dp, Ep, Rm, Rp, D = sp.symbols('Dm Em Dp Ep Rm Rp D', real=True, positive=True)
    d31, E33, p, nu = sp.symbols('d31 E33 p nu', real=True)
    return r, Dm, Em, Dp, Ep, Rm, Rp, D, d31, E33, p, nu


def build_w1(r, Dm, Em, Dp, Ep, Rm, Rp, d31, E33, p, nu):
    # Expression copied from main1.py (region r <= Rm)
    w1 = (1/16)*( (r**2*(-3*(Dm*Em+Dp*Ep)*p*r**2*(-1+nu**2)+(6*(1+nu)*(16*d31*Dm*Dp*(Dm+Dp)*E33*Em*Ep*(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)*Rm**2*(Rm-Rp)*(Rm+Rp)+p*(-1+nu)*(4*Dm**5*Em**3*Rm**6+Dp**5*Ep**3*(Rm**2-Rp**2)**2*(1+nu)*(2*Rm**2+Rp**2*(1+nu))+2*Dm**4*Dp*Em**2*Ep*Rm**2*(2*Rm**2*Rp**2*(-2+nu)-3*Rp**4*(1+nu)+Rm**4*(9+nu))+Dm*Dp**4*Em*Ep**2*(Rm-Rp)*(Rm+Rp)*(Rm**2*Rp**2*(1+nu)**2-Rp**4*(1+nu)**2+4*Rm**4*(3+2*nu))+4*Dm**2*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(3*Ep*Rm**2*(1+nu)+2*Em*(2*Rm**2+Rp**2*(1+nu)))+4*Dm**3*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(2*Ep*Rm**2*(1+nu)+3*Em*(2*Rm**2+Rp**2*(1+nu))))))/(4*Dm**4*Em**2*Rm**4+8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+Dp**4*Ep**2*(Rm**2-Rp**2)**2*(1+nu)**2))/(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)+1/(Dm**3*Em)*(3*(1+nu)*(-(p*((Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)*Rm**4-Dp*Ep*(Dm*(3*Dm**2+6*Dm*Dp+4*Dp**2)*Em+Dp**3*Ep)*Rp**4)*(-1+nu))/(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)-(4*Dp*Ep*Rm**2*Rp**2*(16*d31*Dm**4*(Dm+Dp)*E33*Em**2*Rm**2-p*(Rm-Rp)*(Rm+Rp)*(-1+nu)*(1+nu)*(6*Dm**3*Em*Rm**2+12*Dm**2*Dp*Em*Rm**2+8*Dm*Dp**2*Em*Rm**2+Dp**3*Ep*(Rm-Rp)*(Rm+Rp)*(1+nu)))*(sp.log(Rm)-sp.log(Rp)))/(4*Dm**4*Em**2*Rm**4+8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+Dp**4*Ep**2*(Rm-Rp)**2*(Rm+Rp)**2*(1+nu)**2)))))
    return w1


def build_integrand_symbolic(conservative=False, return_inner=False):
    r, Dm, Em, Dp, Ep, Rm, Rp, D, d31, E33, p, nu = build_symbols()
    w1 = build_w1(r, Dm, Em, Dp, Ep, Rm, Rp, d31, E33, p, nu)

    # inner_term = d/dr( r * d/dr(w1) ) / r (keep evaluate=False to avoid expansion)
    w1_r = sp.diff(w1, r)
    inner = sp.diff(r * w1_r, r, evaluate=False) / r

    if conservative:
        integrand = inner**2 * r
    else:
        integrand = sp.together(inner**2 * r)
        integrand = sp.cancel(integrand)

    if return_inner:
        return integrand, inner, r, Rm
    return integrand, r, Rm


def default_params():
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


def integrate_single(integrand, r, Rm):
    # 1) Try direct definite integral first
    U = sp.integrate(integrand, (r, 0, Rm), conds='none')
    if not U.has(sp.Integral):
        return sp.simplify(U)

    # 2) Integrate from epsilon>0 to Rm, then take limit eps->0+
    eps = sp.Symbol('eps', positive=True)
    U_eps = sp.integrate(integrand, (r, eps, Rm), conds='none')
    if not U_eps.has(sp.Integral):
        try:
            U_lim = sp.limit(U_eps, eps, 0, dir='+')
            return sp.simplify(U_lim)
        except Exception:
            # Keep it as a formal Limit if gruntz cannot handle it
            return sp.simplify(sp.Limit(U_eps, eps, 0, dir='+'))

    # 3) Fallback: manual antiderivative and evaluate definite integral via limit
    try:
        F = manualintegrate(integrand, r)
        val_Rm = sp.simplify(F.subs(r, Rm))
        try:
            val_0 = sp.simplify(sp.limit(F, r, 0, dir='+'))
        except Exception:
            val_0 = sp.Limit(F, r, 0, dir='+')
        U = val_Rm - val_0
        return sp.simplify(U)
    except Exception:
        # Give up on evaluation; return unevaluated definite integral
        return sp.Integral(integrand, (r, 0, Rm))


def _integrate_term(term, r, Rm):
    return sp.integrate(term, (r, 0, Rm), conds='none')


def integrate_parallel(integrand, r, Rm, verbose=False):
    # Split additive terms without expanding further
    terms = list(sp.Add.make_args(integrand))
    if len(terms) == 1:
        return integrate_single(integrand, r, Rm)
    # Integrate each term in parallel with progress
    if verbose:
        print(f"[parallel] Integrating {len(terms)} term(s) across {cpu_count()} cores...", flush=True)
    worker = partial(_integrate_term, r=r, Rm=Rm)
    parts = []
    completed = 0
    with Pool(cpu_count()) as pool:
        for res in pool.imap_unordered(worker, terms):
            parts.append(res)
            completed += 1
            if verbose and (completed % max(1, len(terms)//10) == 0 or completed == len(terms)):
                print(f"[parallel] Completed {completed}/{len(terms)} term(s)", flush=True)
    U = sum(parts)
    # If any unevaluated Integrals remain, fallback to robust single-path integration
    if any(isinstance(p, sp.Integral) or (hasattr(p, 'has') and p.has(sp.Integral)) for p in parts):
        if verbose:
            print("[parallel] Unevaluated terms detected; retrying single-path integration.", flush=True)
        return integrate_single(integrand, r, Rm)
    return sp.simplify(U)


def main():
    parser = argparse.ArgumentParser(description="Purely symbolic bending energy with optional parallel term integration")
    parser.add_argument('--parallel', action='store_true', help='Integrate additive terms in parallel (may not help)')
    parser.add_argument('--no-cse', action='store_true', help='Disable common subexpression elimination before integration')
    parser.add_argument('--print-stats', action='store_true', help='Print timing and expression sizes')
    parser.add_argument('--diagnose', action='store_true', help='Print diagnostic info and a small numeric probe')
    args = parser.parse_args()

    conservative = bool(args.parallel)

    t0 = time.time()
    print("[step] Building integrand...", flush=True)
    integrand, inner, r, Rm = build_integrand_symbolic(conservative=conservative, return_inner=True)

    # CSE to shrink expression before integration (keeps it symbolic). Disable in conservative mode.
    if not args.no_cse and not conservative:
        repl, red_list = sp.cse(integrand, symbols=sp.numbered_symbols('k'))
        red = red_list[0]
    else:
        repl, red = [], integrand

    t1 = time.time()
    if args.print_stats or args.diagnose:
        try:
            n_terms = len(sp.Add.make_args(red))
        except Exception:
            n_terms = 1
        print(f"[step] Integrand ready in {t1 - t0:.2f}s; top-level terms: {n_terms}", flush=True)

    if args.parallel:
        print("[step] Starting parallel integration...", flush=True)
        U_red = integrate_parallel(red, r, Rm, verbose=(args.print_stats or args.diagnose))
    else:
        print("[step] Starting single-path integration...", flush=True)
        U_red = integrate_single(red, r, Rm)

    # Substitute CSE symbols back
    U = U_red.xreplace(dict(repl)) if repl else U_red
    U = sp.simplify(U)

    # Zero-guard: if it simplifies to 0, retry with conservative single-path integration
    if U == 0:
        print("[warn] Result simplified to 0; retrying conservative single-path...", flush=True)
        integrand_cons, r_c, Rm_c = build_integrand_symbolic(conservative=True, return_inner=False)
        U_retry = integrate_single(integrand_cons, r_c, Rm_c)
        U_retry = sp.simplify(U_retry)
        if U_retry != 0:
            U = U_retry

    t2 = time.time()

    print("=== Symbolic Bending Energy U ===")
    print(U)

    if args.print_stats or args.diagnose:
        try:
            n_terms = len(sp.Add.make_args(integrand))
        except Exception:
            n_terms = 1
        print("--- Stats ---")
        print(f"Integrand terms (pre-CSE): {n_terms}")
        print(f"Build integrand: {t1 - t0:.2f}s")
        print(f"Integrate: {t2 - t1:.2f}s")
        print(f"Total: {t2 - t0:.2f}s")

    if args.diagnose:
        print("--- Diagnose ---")
        print("inner simplified (truncated):", sp.srepr(sp.simplify(inner))[:2000])
        print("integrand simplified (truncated):", sp.srepr(sp.simplify(integrand))[:2000])
        # Small numeric probe at r = Rm/2 with default params
        try:
            params = default_params()
            subs = {sp.Symbol('Dm'): params['Dm'], sp.Symbol('Em'): params['Em'],
                    sp.Symbol('Dp'): params['Dp'], sp.Symbol('Ep'): params['Ep'],
                    sp.Symbol('Rm'): params['Rm'], sp.Symbol('Rp'): params['Rp'],
                    sp.Symbol('d31'): params['d31'], sp.Symbol('E33'): params['E33'],
                    sp.Symbol('p'): params['p'], sp.Symbol('nu'): params['nu'],
                    r: params['Rm']/2}
            inner_val = sp.N(inner.subs(subs))
            print("inner(Rm/2) numeric probe:", inner_val)
        except Exception as e:
            print("diagnostic probe failed:", e)


if __name__ == '__main__':
    main()
