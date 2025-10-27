# Python tool will:
# - define functions for the two-region plate deflection (Biegelinie) from the user's Mathematica expressions
# - provide a simple plotting routine for r in [0, Rm]
# - include an example parameter set the user can edit
#
# Notes:
# - Natural logs: use np.log
# - All inputs are floats with SI base units (meters, pascals, coulomb/newton etc.)
# - Region 1: r in [0, Rp]; Region 2: r in [Rp, Rm]
# - The code guards r=0 by starting from a small epsilon to avoid log(0) in Region 1 pieces that include Log[Rm]-Log[Rp] only (so fine).

import numpy as np
import matplotlib.pyplot as plt

def biegelinie_region1(
    r,
    Dm, Dp, Em, Ep, nu, d31, E33, p, Rm, Rp
):
    """
    Transversale Auslenkung in Region 1 (0 <= r <= Rp)
    Direct translation from the provided Mathematica expression.
    """
    # Common composite terms
    A = (Dm**4 * Em**2 + 2*Dm*Dp*(2*Dm**2 + 3*Dm*Dp + 2*Dp**2) * Em * Ep + Dp**4 * Ep**2)
    B = (4*Dm**4*Em**2*Rm**4
         + 8*Dm**3*Dp*Em*Ep*Rm**2*(Rm - Rp)*(Rm + Rp)*(1 + nu)
         + 12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm - Rp)*(Rm + Rp)*(1 + nu)
         + 8*Dm*Dp**3*Em*Ep*Rm**2*(Rm - Rp)*(Rm + Rp)*(1 + nu)
         + Dp**4*Ep**2*(Rm**2 - Rp**2)**2 * (1 + nu)**2)
    
    # Big numerator block inside first fraction of the first big term
    # Break into readable components to minimize mistakes
    # Core piezo-pressure mix term inside the long (6(1+nu)(...)) bracket:
    T_piezo_pressure_num = (
        16*d31*Dm*Dp*(Dm + Dp)*E33*Em*Ep
        * (Dm**4*Em**2 + 2*Dm*Dp*(2*Dm**2 + 3*Dm*Dp + 2*Dp**2)*Em*Ep + Dp**4*Ep**2)
        * Rm**2 * (Rm - Rp) * (Rm + Rp)
        + p*(-1 + nu)*(
            4*Dm**5*Em**3*Rm**6
            + Dp**5*Ep**3*(Rm**2 - Rp**2)**2*(1 + nu)*(2*Rm**2 + Rp**2*(1 + nu))
            + 2*Dm**4*Dp*Em**2*Ep*Rm**2*(2*Rm**2*Rp**2*(-2 + nu) - 3*Rp**4*(1 + nu) + Rm**4*(9 + nu))
            + Dm*Dp**4*Em*Ep**2*(Rm - Rp)*(Rm + Rp)*(Rm**2*Rp**2*(1 + nu)**2 - Rp**4*(1 + nu)**2 + 4*Rm**4*(3 + 2*nu))
            + 4*Dm**2*Dp**3*Em*Ep*Rm**2*(Rm - Rp)*(Rm + Rp)*(3*Ep*Rm**2*(1 + nu) + 2*Em*(2*Rm**2 + Rp**2*(1 + nu)))
            + 4*Dm**3*Dp**2*Em*Ep*Rm**2*(Rm - Rp)*(Rm + Rp)*(2*Ep*Rm**2*(1 + nu) + 3*Em*(2*Rm**2 + Rp**2*(1 + nu)))
        )
    )

    term1_inner = (-3*(Dm*Em + Dp*Ep)*p*r**2*(-1 + nu**2)
                   + 6*(1 + nu)*T_piezo_pressure_num / B)

    term1 = (r**2 * term1_inner) / (16 * A)

    # Second big term
    # The long bracket:
    # First piece: pressure-only part
    press_piece = (
        - (p * ((A)*Rm**4 - Dp*Ep*(Dm*(3*Dm**2 + 6*Dm*Dp + 4*Dp**2)*Em + Dp**3*Ep) * Rp**4) * (-1 + nu)) / A
    )

    # Second piece: log term over B
    log_piece_num = (
        4*Dp*Ep*Rm**2*Rp**2 * (
            16*d31*Dm**4*(Dm + Dp)*E33*Em**2*Rm**2
            - p*(Rm - Rp)*(Rm + Rp)*(-1 + nu)*(1 + nu) * (
                6*Dm**3*Em*Rm**2 + 12*Dm**2*Dp*Em*Rm**2 + 8*Dm*Dp**2*Em*Rm**2
                + Dp**3*Ep*(Rm - Rp)*(Rm + Rp)*(1 + nu)
            )
        ) * (np.log(Rm) - np.log(Rp))
    )
    log_piece = - log_piece_num / B  # note the negative sign from original -( ... (Log[Rm]-Log[Rp]))/B

    bracket = 3*(1 + nu) * (press_piece + log_piece)

    term2 = bracket / (16 * Dm**3 * Em)

    w1 = term1 + term2
    return w1


def biegelinie_region2(
    r,
    Dm, Dp, Em, Ep, nu, d31, E33, p, Rm, Rp
):
    """
    Transversale Auslenkung in Region 2 (Rp <= r <= Rm)
    Direct translation from the provided Mathematica expression.
    """
    B = (4*Dm**4*Em**2*Rm**4
         + 8*Dm**3*Dp*Em*Ep*Rm**2*(Rm - Rp)*(Rm + Rp)*(1 + nu)
         + 12*Dm**2*Dp**2*Em*Ep*Rm**2*(Rm - Rp)*(Rm + Rp)*(1 + nu)
         + 8*Dm*Dp**3*Em*Ep*Rm**2*(Rm - Rp)*(Rm + Rp)*(1 + nu)
         + Dp**4*Ep**2*(Rm - Rp)**2*(Rm + Rp)**2 * (1 + nu)**2)

    # Polynomial part inside big parentheses
    poly_num = (
        32*d31*Dm**5*Dp*E33*Em**2*Ep*Rm**2*Rp**2
        + 4*Dm**4*Em**2*Rm**2*(8*d31*Dp**2*E33*Ep*Rp**2 + p*(r - Rm)*Rm**2*(r + Rm)*(-1 + nu))
        + Dp**4*Ep**2*p*(Rm - Rp)**2*(Rm + Rp)**2*(r**2 - Rm**2 - 2*Rp**2)*(-1 + nu)*(1 + nu)**2
        - 12*Dm**2*Dp**2*Em*Ep*p*Rm**2*(Rm - Rp)*(Rm + Rp)*(-r**2 + Rm**2 + 2*Rp**2)*(-1 + nu**2)
        - 8*Dm*Dp**3*Em*Ep*p*Rm**2*(Rm - Rp)*(Rm + Rp)*(-r**2 + Rm**2 + 2*Rp**2)*(-1 + nu**2)
        - 4*Dm**3*Dp*Em*Ep*p*Rm**2*(Rm - Rp)*(Rm + Rp)*(-2*r**2 + 2*Rm**2 + 3*Rp**2)*(-1 + nu**2)
    )

    log_piece_num = (
        4*Dp*Ep*Rm**2*Rp**2 * (
            16*d31*Dm**4*(Dm + Dp)*E33*Em**2*Rm**2
            - p*(Rm - Rp)*(Rm + Rp)*(-1 + nu)*(1 + nu) * (
                6*Dm**3*Em*Rm**2 + 12*Dm**2*Dp*Em*Rm**2 + 8*Dm*Dp**2*Em*Rm**2
                + Dp**3*Ep*(Rm - Rp)*(Rm + Rp)*(1 + nu)
            )
        ) * (np.log(r) - np.log(Rm))
    )

    numerator = 3*(1 + nu) * ( - (r - Rm)*(r + Rm) * poly_num + log_piece_num )
    w2 = numerator / (16 * Dm**3 * Em * B)
    return w2


def biegelinie_r(
    r_array,
    Dm, Dp, Em, Ep, nu, d31, E33, p, Rm, Rp
):
    r_array = np.asarray(r_array, dtype=float)
    w = np.zeros_like(r_array, dtype=float)
    mask1 = r_array <= Rp
    mask2 = ~mask1
    if np.any(mask1):
        w[mask1] = biegelinie_region1(r_array[mask1], Dm, Dp, Em, Ep, nu, d31, E33, p, Rm, Rp)
    if np.any(mask2):
        w[mask2] = biegelinie_region2(r_array[mask2], Dm, Dp, Em, Ep, nu, d31, E33, p, Rm, Rp)
    return w


# Example parameters (edit these to your device). Units SI.
params = dict(
    Dm = 110e-6,   # silicon thickness [m]
    Dp = 190e-6,  # piezoceramic thickness [m]
    Em = 170e9,   # Young's modulus silicon [Pa]
    Ep = 66e9,    # Young's modulus PZT-5H approx [Pa]
    nu = 0.22,    # Poisson ratio silicon [-]
    d31 = -175e-12, # PZT-5H d31 [C/N]
    E33 = 2e6,    # Applied electric field in piezo [V/m] (example)
    p = 100000,      # Pressure load [Pa] (example)
    Rm = 3.35e-3,  # membrane radius [m]
    Rp = 3e-3   # piezo radius [m]
)

# Build r grid and compute w(r)
Rm = params["Rm"]; Rp = params["Rp"]
r = np.linspace(0, Rm, 1200)
# Avoid log(0) in Region 2 term: Region 2 begins at Rp>0, so safe. Region 1 uses only log(Rm)-log(Rp), safe.
w = biegelinie_r(r, **params)

# Plot: single chart as required
plt.figure()
plt.plot(r, w, label="w(r)")
plt.xlabel("r [m]")
plt.ylabel("Deflection w [m]")
plt.title("Biegelinie der Membran mit konzentrischem Piezo")
plt.grid(True)
plt.legend()
plt.show()

# Also return a small helper that users can call again
print("Call biegelinie_r(r, Dm, Dp, Em, Ep, nu, d31, E33, p, Rm, Rp) with your parameters.\n"
      "Region 1: r <= Rp. Region 2: Rp < r <= Rm.")
