import sympy as sp
import numpy as np
import math

# Definieren Sie Symbole nach Muster x = sp.symbols('x')
# Alle benötigten Symbole definieren
r = sp.Symbol('r', real=True, positive=True)
Dm, Em, Dp, Ep, Rm, Rp = sp.symbols('Dm Em Dp Ep Rm Rp', real=True, positive=True)
d31, E33, p, nu = sp.symbols('d31 E33 p nu', real=True)


# Hier w1 und w2 definieren (Ausschnitte der langen Ausdrücke)

def transversale_auslenkung_region1(r, Dm, Em, Dp, Ep, p, nu, d31, E33, Rm, Rp): 
    return (1/16)*((r**2*(-3*(Dm*Em+Dp*Ep)*p*r**2*(-1+nu**2)+(6*(1+nu)*(16*d31*Dm*Dp*(Dm+Dp)*E33*Em*Ep*(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)*Rm**2*(Rm-Rp)*(Rm+Rp)+p*(-1+nu)*(4*Dm**5*Em**3*Rm**6+Dp**5*Ep**3*(Rm**2-Rp**2)**2*(1+nu)*(2*Rm**2+Rp**2*(1+nu))+2*Dm**4*Dp*Em**2*Ep*Rm**2*(2*Rm**2*Rp**2*(-2+nu)-3*Rp**4*(1+nu)+Rm**4*(9+nu))+Dm*Dp**4*Em*Ep**2*(Rm-Rp)*(Rm+Rp)*(Rm**2*Rp**2*(1+nu)**2-Rp**4*(1+nu)**2+4*Rm**4*(3+2*nu))+4*Dm**2*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(3*Ep*Rm**2*(1+nu)+2*Em*(2*Rm**2+Rp**2*(1+nu)))+4*Dm**3*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(2*Ep*Rm**2*(1+nu)+3*Em*(2*Rm**2+Rp**2*(1+nu))))))/(4*Dm**4*Em**2*Rm**4+8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+Dp**4*Ep**2*(Rm**2-Rp**2)**2*(1+nu)**2))/(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)+1/(Dm**3*Em)*(3*(1+nu)*(-(p*((Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)*Rm**4-Dp*Ep*(Dm*(3*Dm**2+6*Dm*Dp+4*Dp**2)*Em+Dp**3*Ep)*Rp**4)*(-1+nu))/(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)-(4*Dp*Ep*Rm**2*Rp**2*(16*d31*Dm**4*(Dm+Dp)*E33*Em**2*Rm**2-p*(Rm-Rp)*(Rm+Rp)*(-1+nu)*(1+nu)*(6*Dm**3*Em*Rm**2+12*Dm**2*Dp*Em*Rm**2+8*Dm*Dp**2*Em*Rm**2+Dp**3*Ep*(Rm-Rp)*(Rm+Rp)*(1+nu)))*(math.log(Rm)-math.log(Rp)))/(4*Dm**4*Em**2*Rm**4+8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+Dp**4*Ep**2*(Rm-Rp)**2*(Rm+Rp)**2*(1+nu)**2)))))


def transversale_auslenkung_region2(r, Dm, Em, Dp, Ep, p, nu, d31, E33, Rm, Rp): 
    return (3*(1+nu)*((-(r-Rm)*(r+Rm)*(32*d31*Dm**5*Dp*E33*Em**2*Ep*Rm**2*Rp**2+4*Dm**4*Em**2*Rm**2*(8*d31*Dp**2*E33*Ep*Rp**2+p*(r-Rm)*Rm**2*(r+Rm)*(-1+nu))+Dp**4*Ep**2*p*(Rm-Rp)**2*(Rm+Rp)**2*(r**2-Rm**2-2*Rp**2)*(-1+nu)*(1+nu)**2-12*Dm**2*Dp**2*Em*Ep*p*Rm**2*(Rm-Rp)*(Rm+Rp)*(-r**2+Rm**2+2*Rp**2)*(-1+nu**2)-8*Dm*Dp**3*Em*Ep*p*Rm**2*(Rm-Rp)*(Rm+Rp)*(-r**2+Rm**2+2*Rp**2)*(-1+nu**2)-4*Dm**3*Dp*Em*Ep*p*Rm**2*(Rm-Rp)*(Rm+Rp)*(-2*r**2+2*Rm**2+3*Rp**2)*(-1+nu**2)))+4*Dp*Ep*Rm**2*Rp**2*(16*d31*Dm**4*(Dm+Dp)*E33*Em**2*Rm**2-p*(Rm-Rp)*(Rm+Rp)*(-1+nu)*(1+nu)*(6*Dm**3*Em*Rm**2+12*Dm**2*Dp*Em*Rm**2+8*Dm*Dp**2*Em*Rm**2+Dp**3*Ep*(Rm-Rp)*(Rm+Rp)*(1+nu)))*(math.log(r)-math.log(Rm))))/(16*Dm**3*Em*(4*Dm**4*Em**2*Rm**4+8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+Dp**4*Ep**2*(Rm-Rp)**2*(Rm+Rp)**2*(1+nu)**2))

# Biegesteifigkeiten für beide Regionen definieren
D1 = (Em * Dm**3) / (12 * (1 - nu**2))  # Region 1 (Membran + Piezo)
D2 = (Ep * Dp**3) / (12 * (1 - nu**2))  # Region 2 (nur Membran)


# Biegesteifigkeiten für beide Regionen definieren
D1 = (Em * Dm**3) / (12 * (1 - nu**2))  # Region 1 (Membran + Piezo)
D2 = (Ep * Dp**3) / (12 * (1 - nu**2))  # Region 2 (nur Membran)

def bending_energy_density(w, D, r):

    # Erste und zweite Ableitung
    w_r = sp.diff(w, r)      # dw/dr
    w_rr = sp.diff(w_r, r)   # d²w/dr²
    
    print(f"Ableitungen für w:")
    print(f"  w_r = {w_r}")
    print(f"  w_rr = {w_rr}")
    
    # Biegeenergiedichte für rotationssymmetrische Platte
    # U_dens = 1/2 * D * [(d²w/dr²)² + (1/r * dw/dr)² + 2*ν*(d²w/dr²)*(1/r * dw/dr)]
    term1 = w_rr**2
    term2 = (1/r * w_r)**2
    term3 = 2 * nu * w_rr * (1/r * w_r)
    
    U_dens = 0.5 * D * (term1 + term2 + term3)
    
    print(f"\nTerme in U_dens:")
    print(f"  Term1 (w_rr²): {term1}")
    print(f"  Term2 ((1/r * w_r)²): {term2}") 
    print(f"  Term3 (2ν*w_rr*(1/r * w_r)): {term3}")
    print(f"  U_dens: {U_dens}")
    
    return U_dens

# Test für Region 1
print("=== Region 1 ===")
U_dens1 = bending_energy_density(w1, D1, r)

print("\n=== Region 2 ===")
U_dens2 = bending_energy_density(w2, D2, r)

def total_bending_energy(w, D, r_min, r_max):
    """
    Berechnet die gesamte Biegeenergie für eine Region
    """
    U_dens = bending_energy_density(w, D, r)
    
    # Integration: U = 2π ∫ U_dens * r dr
    # Der Faktor r kommt von dA = 2πr dr
    U_total = 2 * sp.pi * sp.integrate(U_dens * r, (r, r_min, r_max))
    
    print(f"\nIntegration von {r_min} bis {r_max}:")
    print(f"  Integrand: {U_dens * r}")
    print(f"  Ergebnis: {U_total}")
    
    return U_total

# Gesamte Biegeenergie berechnen
print("=== Berechnung der gesamten Biegeenergie ===")
U1 = total_bending_energy(w1, D1, 0, Rp)    # Region 1: 0 bis Rp
U2 = total_bending_energy(w2, D2, Rp, Rm)   # Region 2: Rp bis Rm

U_gesamt = U1 + U2
print(f"\n=== Gesamte Biegeenergie ===")
print(f"U_gesamt = {U_gesamt}")






















'''
print(f"Ableitungen für w:")
print(f"  w_r = {w_r}")
print(f"  w_rr = {w_rr}")
# Biegeenergie berechnen
def bending_energy(w, D, r):
    w_r = sp.diff(w, r)      # erste Ableitung dw/dr so weit sieht das alles supi aus
    w_rr = sp.diff(w_r, r)   # zweite Ableitung d^2w/dr^2 hier natürlich auch noch
     
    # Biegeenergiedichte
    U_dens = 0.5 * D * (w_rr**2 + (1/r)*w_rr*w_r + (1/r**2)*w_r**2)
    
    # Integration über Fläche
    U = 2 * sp.pi * sp.integrate(U_dens * r, r)
    return U

# Für beide Regionen berechnen
U1 = bending_energy(w1, D1, r).subs(r, Rp) - bending_energy(w1, D1, r).subs(r, 0)
U2 = bending_energy(w2, D2, r).subs(r, Rm) - bending_energy(w2, D2, r).subs(r, Rp)
U_gesamt = U1 + U2 
