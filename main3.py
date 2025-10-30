import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

print("=== Numerische Berechnung der Biegeenergie ===\n")

# Parameter definieren
params = {
    'Dm': 110e-6,    # [m]
    'Em': 170e9,     # [Pa]
    'Dp': 190e-6,    # [m]
    'Ep': 65e9,      # [Pa]
    'Rm': 3.35e-3,   # [m]
    'Rp': 3.0e-3,    # [m]
    'd31': 2e-12,    # [m/V]
    'E33': 150e9,    # [Pa]
    'p': 100,        # [Pa]
    'nu': 0.30       # [-]
}

print("Parameter:")
for key, val in params.items():
    print(f"  {key} = {val:.4e}")

# Auslagerung der w-Funktionen für bessere Lesbarkeit
def w1_func(r, Dm, Em, Dp, Ep, Rm, Rp, d31, E33, p, nu):
    """Auslenkung für Region 1: r <= Rm"""
    term1_num = r**2 * (-3*(Dm*Em+Dp*Ep)*p*r**2*(-1+nu**2) + 
                (6*(1+nu)*(16*d31*Dm*Dp*(Dm+Dp)*E33*Em*Ep*(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)*Rm**2*(Rm-Rp)*(Rm+Rp) + 
                p*(-1+nu)*(4*Dm**5*Em**3*Rm**6 + Dp**5*Ep**3*(Rm**2-Rp**2)**2*(1+nu)*(2*Rm**2+Rp**2*(1+nu)) + 
                2*Dm**4*Dp*Em**2*Ep*Rm**2*(2*Rm**2*Rp**2*(-2+nu)-3*Rp**4*(1+nu)+Rm**4*(9+nu)) + 
                Dm*Dp**4*Em*Ep**2*(Rm-Rp)*(Rm+Rp)*(Rm**2*Rp**2*(1+nu)**2-Rp**4*(1+nu)**2+4*Rm**4*(3+2*nu)) + 
                4*Dm**2*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(3*Ep*Rm**2*(1+nu)+2*Em*(2*Rm**2+Rp**2*(1+nu))) + 
                4*Dm**3*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(2*Ep*Rm**2*(1+nu)+3*Em*(2*Rm**2+Rp**2*(1+nu)))))))
    
    term1_den = (4*Dm**4*Em**2*Rm**4 + 8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu) + 
                 12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu) + 
                 8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu) + 
                 Dp**4*Ep**2*(Rm**2-Rp**2)**2*(1+nu)**2)
    
    term2_num = -(p*((Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)*Rm**4 - 
                 Dp*Ep*(Dm*(3*Dm**2+6*Dm*Dp+4*Dp**2)*Em+Dp**3*Ep)*Rp**4)*(-1+nu))
    
    term2_den = (Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)
    
    log_term_num = 4*Dp*Ep*Rm**2*Rp**2*(16*d31*Dm**4*(Dm+Dp)*E33*Em**2*Rm**2 - 
                   p*(Rm-Rp)*(Rm+Rp)*(-1+nu)*(1+nu)*(6*Dm**3*Em*Rm**2+12*Dm**2*Dp*Em*Rm**2+8*Dm*Dp**2*Em*Rm**2+Dp**3*Ep*(Rm-Rp)*(Rm+Rp)*(1+nu)))
    
    log_term = log_term_num * (np.log(Rm) - np.log(Rp)) / term1_den
    
    w = (1/16) * (term1_num/term1_den/(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2) + 
                  (3*(1+nu)*(term2_num/term2_den - log_term))/(Dm**3*Em))
    
    return w

def w2_func(r, Dm, Em, Dp, Ep, Rm, Rp, d31, E33, p, nu):
    """Auslenkung für Region 2: r > Rm"""
    term_main = (-(r-Rm)*(r+Rm)*(32*d31*Dm**5*Dp*E33*Em**2*Ep*Rm**2*Rp**2 + 
                4*Dm**4*Em**2*Rm**2*(8*d31*Dp**2*E33*Ep*Rp**2+p*(r-Rm)*Rm**2*(r+Rm)*(-1+nu)) + 
                Dp**4*Ep**2*p*(Rm-Rp)**2*(Rm+Rp)**2*(r**2-Rm**2-2*Rp**2)*(-1+nu)*(1+nu)**2 - 
                12*Dm**2*Dp**2*Em*Ep*p*Rm**2*(Rm-Rp)*(Rm+Rp)*(-r**2+Rm**2+2*Rp**2)*(-1+nu**2) - 
                8*Dm*Dp**3*Em*Ep*p*Rm**2*(Rm-Rp)*(Rm+Rp)*(-r**2+Rm**2+2*Rp**2)*(-1+nu**2) - 
                4*Dm**3*Dp*Em*Ep*p*Rm**2*(Rm-Rp)*(Rm+Rp)*(-2*r**2+2*Rm**2+3*Rp**2)*(-1+nu**2)))
    
    log_term_num = 4*Dp*Ep*Rm**2*Rp**2*(16*d31*Dm**4*(Dm+Dp)*E33*Em**2*Rm**2 - 
                   p*(Rm-Rp)*(Rm+Rp)*(-1+nu)*(1+nu)*(6*Dm**3*Em*Rm**2+12*Dm**2*Dp*Em*Rm**2+8*Dm*Dp**2*Em*Rm**2+Dp**3*Ep*(Rm-Rp)*(Rm+Rp)*(1+nu)))
    
    log_term = log_term_num * (np.log(r) - np.log(Rm))
    
    denominator = (4*Dm**4*Em**2*Rm**4 + 8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu) + 
                   12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu) + 
                   8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu) + 
                   Dp**4*Ep**2*(Rm-Rp)**2*(Rm+Rp)**2*(1+nu)**2)
    
    w = (3*(1+nu)*(term_main + log_term))/(16*Dm**3*Em*denominator)
    
    return w

def w_func(r, Dm, Em, Dp, Ep, Rm, Rp, d31, E33, p, nu):
    """Kombinierte w-Funktion mit Piecewise-Logik"""
    if np.isscalar(r):
        if r <= Rm:
            return w1_func(r, Dm, Em, Dp, Ep, Rm, Rp, d31, E33, p, nu)
        else:
            return w2_func(r, Dm, Em, Dp, Ep, Rm, Rp, d31, E33, p, nu)
    else:
        return np.where(r <= Rm, 
                       w1_func(r, Dm, Em, Dp, Ep, Rm, Rp, d31, E33, p, nu),
                       w2_func(r, Dm, Em, Dp, Ep, Rm, Rp, d31, E33, p, nu))

# Numerische Ableitungen
def numerical_derivative(f, x, h=1e-8):
    """Zentrale Differenz für erste Ableitung"""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_second_derivative(f, x, h=1e-8):
    """Zentrale Differenz für zweite Ableitung"""
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

# Wrapper für einfachere Auswertung
def w(r):
    return w_func(r, **params)

def w_r(r):
    return numerical_derivative(w, r)

def w_rr(r):
    return numerical_second_derivative(w, r)

# Integrand für Biegeenergie: ((d/dr)(r * dw/dr) / r)^2 * r
def integrand(r):
    if r < 1e-10:  # Vermeide Division durch Null
        return 0.0
    
    # Berechne (d/dr)(r * w'(r)) / r
    h = 1e-8
    r_wp_plus = (r + h) * w_r(r + h)
    r_wp_minus = (r - h) * w_r(r - h)
    d_r_wp_dr = (r_wp_plus - r_wp_minus) / (2 * h)
    
    inner_term = d_r_wp_dr / r
    
    return inner_term**2 * r

print("\n=== Berechne Biegeenergie U ===")
print("Integration läuft...")

# Numerische Integration von 0 bis Rm
U, error = quad(integrand, 1e-9, params['Rm'], limit=100)

print(f"\n✓ Integration erfolgreich!")
print(f"  Biegeenergie U = {U:.6e} J/m")
print(f"  Numerischer Fehler: {error:.2e}")

# Visualisierung
print("\n=== Erstelle Plots ===")

r_vals = np.linspace(1e-9, params['Rm'], 500)
w_vals = np.array([w(r) for r in r_vals])
w_r_vals = np.array([w_r(r) for r in r_vals])
w_rr_vals = np.array([w_rr(r) for r in r_vals])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Auslenkung w(r)
axes[0, 0].plot(r_vals * 1e3, w_vals * 1e6, 'b-', linewidth=2)
axes[0, 0].axvline(params['Rm'] * 1e3, color='r', linestyle='--', alpha=0.5, label=f'Rm = {params["Rm"]*1e3:.2f} mm')
axes[0, 0].set_xlabel('Radius r [mm]')
axes[0, 0].set_ylabel('Auslenkung w(r) [µm]')
axes[0, 0].set_title('Auslenkung der Membran')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Erste Ableitung (Neigung)
axes[0, 1].plot(r_vals * 1e3, w_r_vals * 1e3, 'g-', linewidth=2)
axes[0, 1].axvline(params['Rm'] * 1e3, color='r', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Radius r [mm]')
axes[0, 1].set_ylabel('dw/dr [10⁻³]')
axes[0, 1].set_title('Erste Ableitung (Neigung)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Zweite Ableitung (Krümmung)
axes[1, 0].plot(r_vals * 1e3, w_rr_vals, 'orange', linewidth=2)
axes[1, 0].axvline(params['Rm'] * 1e3, color='r', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Radius r [mm]')
axes[1, 0].set_ylabel('d²w/dr² [1/m]')
axes[1, 0].set_title('Zweite Ableitung (Krümmung)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Integrand
r_int = np.linspace(1e-9, params['Rm'], 300)
integrand_vals = np.array([integrand(r) for r in r_int])
axes[1, 1].plot(r_int * 1e3, integrand_vals, 'purple', linewidth=2)
axes[1, 1].set_xlabel('Radius r [mm]')
axes[1, 1].set_ylabel('Integrand')
axes[1, 1].set_title(f'Integrand der Biegeenergie\n∫ = {U:.3e}')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].fill_between(r_int * 1e3, integrand_vals, alpha=0.3)

plt.tight_layout()
plt.savefig('biegeenergie_analyse.png', dpi=300, bbox_inches='tight')
print("✓ Plots gespeichert als 'biegeenergie_analyse.png'")

plt.show()

print("\n=== Fertig! ===")