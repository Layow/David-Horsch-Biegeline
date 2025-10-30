import sympy as sp
import numpy as np

# Schritt 1: Symbole definieren
print("=== Schritt 1: Symbol-Definition ===")
r = sp.Symbol('r', real=True, nonnegative=True) #null muss auch enthalten sein
Dm, Em, Dp, Ep, Rm, Rp, D = sp.symbols('Dm Em Dp Ep Rm Rp D', real=True, positive=True)
d31, E33, p, nu = sp.symbols('d31 E33 p nu', real=True)

# Die Funktionen 
print("\n=== Schritt 2: Funktionen definieren ===")
#  Ausdrücke 
w1 = (1/16)*( (r**2*(-3*(Dm*Em+Dp*Ep)*p*r**2*(-1+nu**2)+(6*(1+nu)*(16*d31*Dm*Dp*(Dm+Dp)*E33*Em*Ep*(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)*Rm**2*(Rm-Rp)*(Rm+Rp)+p*(-1+nu)*(4*Dm**5*Em**3*Rm**6+Dp**5*Ep**3*(Rm**2-Rp**2)**2*(1+nu)*(2*Rm**2+Rp**2*(1+nu))+2*Dm**4*Dp*Em**2*Ep*Rm**2*(2*Rm**2*Rp**2*(-2+nu)-3*Rp**4*(1+nu)+Rm**4*(9+nu))+Dm*Dp**4*Em*Ep**2*(Rm-Rp)*(Rm+Rp)*(Rm**2*Rp**2*(1+nu)**2-Rp**4*(1+nu)**2+4*Rm**4*(3+2*nu))+4*Dm**2*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(3*Ep*Rm**2*(1+nu)+2*Em*(2*Rm**2+Rp**2*(1+nu)))+4*Dm**3*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(2*Ep*Rm**2*(1+nu)+3*Em*(2*Rm**2+Rp**2*(1+nu))))))/(4*Dm**4*Em**2*Rm**4+8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+Dp**4*Ep**2*(Rm**2-Rp**2)**2*(1+nu)**2))/(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)+1/(Dm**3*Em)*(3*(1+nu)*(-(p*((Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)*Rm**4-Dp*Ep*(Dm*(3*Dm**2+6*Dm*Dp+4*Dp**2)*Em+Dp**3*Ep)*Rp**4)*(-1+nu))/(Dm**4*Em**2+2*Dm*Dp*(2*Dm**2+3*Dm*Dp+2*Dp**2)*Em*Ep+Dp**4*Ep**2)-(4*Dp*Ep*Rm**2*Rp**2*(16*d31*Dm**4*(Dm+Dp)*E33*Em**2*Rm**2-p*(Rm-Rp)*(Rm+Rp)*(-1+nu)*(1+nu)*(6*Dm**3*Em*Rm**2+12*Dm**2*Dp*Em*Rm**2+8*Dm*Dp**2*Em*Rm**2+Dp**3*Ep*(Rm-Rp)*(Rm+Rp)*(1+nu)))*(sp.log(Rm)-sp.log(Rp)))/(4*Dm**4*Em**2*Rm**4+8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+Dp**4*Ep**2*(Rm-Rp)**2*(Rm+Rp)**2*(1+nu)**2)))))
w2 = (3*(1+nu)*((-(r-Rm)*(r+Rm)*(32*d31*Dm**5*Dp*E33*Em**2*Ep*Rm**2*Rp**2+4*Dm**4*Em**2*Rm**2*(8*d31*Dp**2*E33*Ep*Rp**2+p*(r-Rm)*Rm**2*(r+Rm)*(-1+nu))+Dp**4*Ep**2*p*(Rm-Rp)**2*(Rm+Rp)**2*(r**2-Rm**2-2*Rp**2)*(-1+nu)*(1+nu)**2-12*Dm**2*Dp**2*Em*Ep*p*Rm**2*(Rm-Rp)*(Rm+Rp)*(-r**2+Rm**2+2*Rp**2)*(-1+nu**2)-8*Dm*Dp**3*Em*Ep*p*Rm**2*(Rm-Rp)*(Rm+Rp)*(-r**2+Rm**2+2*Rp**2)*(-1+nu**2)-4*Dm**3*Dp*Em*Ep*p*Rm**2*(Rm-Rp)*(Rm+Rp)*(-2*r**2+2*Rm**2+3*Rp**2)*(-1+nu**2)))+4*Dp*Ep*Rm**2*Rp**2*(16*d31*Dm**4*(Dm+Dp)*E33*Em**2*Rm**2-p*(Rm-Rp)*(Rm+Rp)*(-1+nu)*(1+nu)*(6*Dm**3*Em*Rm**2+12*Dm**2*Dp*Em*Rm**2+8*Dm*Dp**2*Em*Rm**2+Dp**3*Ep*(Rm-Rp)*(Rm+Rp)*(1+nu)))*(sp.log(r)-sp.log(Rm))))/(16*Dm**3*Em*(4*Dm**4*Em**2*Rm**4+8*Dm**3*Dp*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+8*Dm*Dp**3*Em*Ep*Rm**2*(Rm-Rp)*(Rm+Rp)*(1+nu)+Dp**4*Ep**2*(Rm-Rp)**2*(Rm+Rp)**2*(1+nu)**2))

w = sp.Piecewise(           #Hier mache ich die beiden einzelregionen into one
    (w1, r <= Rm),
    (w2, r > Rm)
)

params = {
    Dm:110e-6, Em:170e9, Dp:190e-6, Ep:65e9,
    Rm:3.35e-3, Rp:3e-3, d31:2e-12, E33:150e9,
    p:100, nu:0.30, D:0.125 #this D value is grossly approximated. like all the other for now
}


#Schritt 5 Biegeenergie U bestimmen
print("\n=== Schritt 5: Hilfsterme für Biegeenergie U bestimmen===")

def inner_term(w):
    return sp.diff(r*sp.diff(w, r), r) / r

def integrand(w):
    i = inner_term(w)
    return i**2 * r

print("\n=== Schritt 6: Biegeenergie berechnen===")

U = sp.pi * D * sp.integrate(integrand(w), (r, 0, Rm)) #dieser Schritt hier dauert zu lange
print("Biegeenergie Symbolisch =", U)

U_num = sp.N(U.subs(params))
print("Biegeenergie numerisch =", U_num)
