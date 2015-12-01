""" Procedurally generated handwriting using bezier curves.
    The primary method is evaluate which determines for each point
    (u,v) which texture should display.
"""

import copy
import inspect
import math
import numpy as np
import random

class Texture():
    def __init__(self, eval_func=None):
        self.evaluate = eval_func
    
    def evaluate(self, u, v):
        """Determine the texture given u,v coordinates"""
        pass

class Character():
    """A character's strokes index into the control_points array."""
    def __init__(self, control_points, strokes=None):
        self.strokes = list(strokes) if strokes else []
        self.control_points = control_points

    def __getitem__(self, index):
        return self.strokes[index]

    def __setitem__(self, index, item):
        self.strokes[index] = item
    
    def __repr__(self):
        return repr(self.strokes)

    def append(self, s):
        self.strokes.append(s)

    def link_next(self, next_cp):
        """Link from this character's final stroke to next_cp."""
        uv = list(self.control_points[self.strokes[-1]][3])
        if next_cp:
            return  [uv,
                     [uv[0] + (next_cp[0] - uv[0]) / 3,
                      min(uv[1], next_cp[1])],
                     [uv[0] + 2*(next_cp[0] - uv[0]) / 3,
                      min(uv[1], next_cp[1])],
                     list(next_cp)]
        else:
            return []

    def link_prev(self, prev_cp):
        """Link from this character's first stroke to prev_cp."""
        uv = list(self.control_points[self.strokes[-1]][0])
        if prev_cp:
            return  [list(prev_cp),
                     [prev_cp[0] + (uv[0] - prev_cp[0]) / 3,
                      min(uv[1], prev_cp[1])],
                     [prev_cp[0] + 2*(uv[0] - prev_cp[0]) / 3,
                      min(uv[1], prev_cp[1])],
                     uv]
        else:
            return []
    
    def stroke_list(self, prev_cp=None, next_cp=None):
        """Returns all the control points for each stroke."""
        slist = [self.control_points[i] for i in self.strokes]
        if prev_cp:
            slist.append(self.link_prev(prev_cp))
        if next_cp:
            slist.append(self.link_next(next_cp))
        return slist

    def perturb(self, amount):
        """Perturb the character by wiggling the control points."""
        if not 0.0 < amount < 1.0:
            raise ValueError("amount must be in [0,1]")
        pass

class Handwriting(Texture):
    """ Explicit representation of a handwriting texture.
        Returns texture1 if the point (u,v) is inside the handwriting
        and texture2 otherwise. Here, a stroke denotes a bezier curve
        and character denotes a collection of one or more strokes
        overlaid in the same domain.

        Input parameters:
        brush          Function to determine the width of the stroke.
                         May depend on the parameter of the spline.
        scale          How many characters fit on one line.
        seed           The seed for the random number generator
        strokes        The number of unique strokes.
        unique_chars   The number of different characters. May increase
                         if connect_p is nonzero.
        min_s          The minimum number of strokes per character.
        max_s          The maximum number of strokes per character.
        [hv]_stack     Determine how much the characters overlay with each
                         other across the vertical and horizontal axes.
        [hv]_border    Determines how far away from the edge
                         (u or v == 0 or 1) the control points may be.
                         Recommended to be greater than max(brush(t)).
    """

    def __init__(self,
                 texture1=Texture(lambda u,v: (0,0,0)),
                 texture2=Texture(lambda u,v: (1,1,1)),
                 brush=lambda t: 0.1,
                 scale=1,
                 seed=0,
                 unique_strokes=30,
                 unique_chars=25,
                 min_s=2,
                 max_s=3,
                 v_stack=0.3,
                 h_stack=0.3,
                 h_border=0.2,
                 v_border=0.2):
        # Validate parameters
        # Greater than 0.5 means more than 2 characters may contained
        # in a given box, complicating things.
        if not 0.0 <= v_stack <= 0.5:
            raise ValueError("v_stack must be in [0,0.5]")
        if not 0.0 <= h_stack <= 0.5:
            raise ValueError("h_stack must be in [0,0.5]")
        if not 0.0 <= h_border < 0.5:
            raise ValueError("h_border must be in [0,0.5)")
        if not 0.0 <= v_border < 0.5:
            raise ValueError("v_border must be in [0,0.5)")

        self.tex1 = texture1     # Inside bezier curve
        self.tex2 = texture2     # Outside bezier curve
        self.scale = scale
        # unique_strokes = len(control_points) may increase when connect_p > 0.
        self.octaves = 9         # Accurate to dimension / 2^(octaves+1) pixels
        self.eps = lambda t: brush(t) / self.scale
        self.h_stack = h_stack
        self.v_stack = v_stack

        # Create a random hash table to determine which and how many
        # bezier curves are used for each character.
        random.seed(seed)
        hash_sz = 257  # Choose a prime number for the hash size
        self.hash_table = list(range(hash_sz))
        random.shuffle(self.hash_table)

        # Create the control points. 4 control points = 1 bezier curve
        def cp_random():
            """Create a random pair [u,v] used as a control point"""
            return [h_border + (1 - 2*h_border) * random.random(),
                    v_border + (1 - 2*v_border) * random.random()]
        
        self.control_points = []
        for s in range(unique_strokes):
            (p0, p3) = (np.array(cp_random()), np.array(cp_random()))
            p1 = cp_random()
            p2 = cp_random()
            s = [list(p) for p in [p0,p1,p2,p3]]
            self.control_points.append(s)

        # Create each character. self.characters indexing goes
        # [character][bezier curve][control point][uv coordinate]
        # We keep track of how many times each stroke is used so
        # strokes don't get changed twice
        self.characters = []
        used_strokes = {}
        stroke_counter = 0
        for i in range(unique_chars):
            char = Character(self.control_points)
            for stroke in range(random.randint(min_s, max_s)):
                char.append(self.hash_table[stroke_counter % hash_sz]
                            % unique_strokes)
                stroke_counter += 1
                try:
                    used_strokes[char[-1]] += 1
                except KeyError:
                    used_strokes[char[-1]] = 1
            self.characters.append(char)


    def __repr__(self):
        cp_str = "[\n"
        for ind, cp in enumerate(self.control_points):
            cp_str += "{:3d}:".format(ind)
            for p in cp:
                cp_str += " ({:.4f}, {:.4f})".format(p[0], p[1])
            cp_str += "\n"
        cp_str += "]"
        return "control_points: {}\ncharacters: {}\neps: {}".format(
                cp_str, self.characters, inspect.getsource(self.eps).strip())

    def evaluate(self, u, v):
        """ The primary function. Determine whether the point is
            inside any of the bezier curves.
        """
        if self.inside_curve(u,v):
            return self.tex1.evaluate(u,v)
        else:
            return self.tex2.evaluate(u,v)

    def B(self, t, i):
        """Evaluate the ith bezier curve at t."""
        p = i

        # Get the coefficients using coeffs(collect(B),t) in Matlab
        t2 = t*t
        t3 = t2*t
        return np.array((
                # u
                p[0][0] +
                3*(p[1][0] - p[0][0])*t +
                3*(p[0][0] - 2*p[1][0] + p[2][0])*t2 +
                (3*p[1][0] - p[0][0] - 3*p[2][0] + p[3][0])*t3,
                # v
                p[0][1] +
                3*(p[1][1] - p[0][1])*t +
                3*(p[0][1] - 2*p[1][1] + p[2][1])*t2 +
                (3*p[1][1] - p[0][1] - 3*p[2][1] + p[3][1])*t3
            ))

    def inside_curve(self, u, v):
        """ Determine if the point (u,v) is inside of the bezier region.
            via binary search
        """
        def norm2(x):
            """Computes the inner product of a 2d vector x with itself"""
            return x[0]*x[0] + x[1]*x[1]

        def in_curve_exact(uv, i):
            """ Finding the real roots of the polynomial
                dB_udt * (B_u - u) + dB_vdt*(B_v - v).
                This was generated using the following Matlab code:
            syms t u v p0u p1u p2u p3u p0v p1v p2v p3v
            B_u(t) = (1-t)^3*p0u + 3*(1-t)^2*t*p1u + 3*(1-t)*t^2*p2u + t^3*p3u
            B_v(t) = (1-t)^3*p0v + 3*(1-t)^2*t*p1v + 3*(1-t)*t^2*p2v + t^3*p3v
            dB_udt = diff(B_u,t)
            %  dB_udt(t) = 3*p1*(t - 1)^2 - 3*p0*(t - 1)^2 - 3*p2*t^2 +
            %              3*p3*t^2 + 3*p1*t*(2*t - 2) - 2*p2*t*(3*t - 3)
            %  And likewise for B_v. Then we solve the polynomial
            poly = dB_udt*(B_u - u) + dB_vdt*(B_v - v)
            %  We get the roots (and thus the nastly np.array) with
            collect(poly, t)
            """
            # Set variables to be in the form that Matlab outputs
            (u,v) = uv
            ((p0u,p0v),(p1u,p1v),(p2u,p2v),(p3u,p3v)) = i
                
            roots = np.roots(np.array((
                    ((p0u - 3*p1u + 3*p2u - p3u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) + (p0v - 3*p1v + 3*p2v - p3v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)),
                    (- (6*p0u - 12*p1u + 6*p2u)*(p0u - 3*p1u + 3*p2u - p3u) - (6*p0v - 12*p1v + 6*p2v)*(p0v - 3*p1v + 3*p2v - p3v) - (3*p0u - 6*p1u + 3*p2u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) - (3*p0v - 6*p1v + 3*p2v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)),
                    ((3*p0u - 3*p1u)*(p0u - 3*p1u + 3*p2u - p3u) + (3*p0v - 3*p1v)*(p0v - 3*p1v + 3*p2v - p3v) + (3*p0u - 3*p1u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) + (3*p0u - 6*p1u + 3*p2u)*(6*p0u - 12*p1u + 6*p2u) + (3*p0v - 3*p1v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v) + (3*p0v - 6*p1v + 3*p2v)*(6*p0v - 12*p1v + 6*p2v)),
                    (- (3*p0u - 3*p1u)*(3*p0u - 6*p1u + 3*p2u) - (3*p0u - 3*p1u)*(6*p0u - 12*p1u + 6*p2u) - (3*p0v - 3*p1v)*(3*p0v - 6*p1v + 3*p2v) - (3*p0v - 3*p1v)*(6*p0v - 12*p1v + 6*p2v) - (p0u - u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) - (p0v - v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)),
                    ((3*p0u - 3*p1u)**2 + (3*p0v - 3*p1v)**2 + (p0u - u)*(6*p0u - 12*p1u + 6*p2u) + (p0v - v)*(6*p0v - 12*p1v + 6*p2v)),
                     - (p0u - u)*(3*p0u - 3*p1u) - (p0v - v)*(3*p0v - 3*p1v))))
            for r in roots:
                if r.imag == 0.0:
                    t = r.real
                    if (0.0 <= t <= 1.0 and
                           np.linalg.norm(uv - self.B(t,i)) < self.eps(t)
                        ) or (
                        t > 1.0 and
                            np.linalg.norm(uv-self.B(1.0,i)) < self.eps(1.0)
                        ) or (
                        t < 0.0 and
                            np.linalg.norm(uv-self.B(0.0,i)) < self.eps(0.0)):
                        return True
            return False

        def in_curve_bs(uv, i):
            """ Determines if the point (u,v) is inside curve i with
                binary search. May not find all points.
            """
            t = 0.5
            sf = 0.25  # How much to change t for the binary search
            for o in range(self.octaves):
                # If ||(uv - B(t+sf)||^2 < ||(uv - B(t-sf)||^2
                if norm2(uv-self.B(t+sf, i)) < norm2(uv-self.B(t-sf, i)):
                    t = t + sf
                else:
                    t = t - sf
                sf /= 2
            if np.linalg.norm(uv - self.B(t,i)) < self.eps(t):
                return True
            else:
                return False

        def get_char_index(u, v):
            """ Given a scaled u,v coordinate, take the integer parts
                to choose which character goes with that coordinate.
            """
            return self.hash_table[
                (self.hash_table[int(math.floor(u)) % len(self.hash_table)] +
                int(math.floor(v))) % len(self.hash_table)] % len(self.characters)
                
        def in_char(c, uv, prev_cp=None, next_cp=None):
            """ Given a decimal position of the character, find out
                if the point is in the character or any decorations.
            """
            char = self.characters[c]
            return any([in_curve_exact(uv, val) for val in
                       char.stroke_list(prev_cp, next_cp)])
        
        def connect_next(u, v):
            """ Find the control point to which the current character
                will connect and translate into the current character's
                coordinates.
            """
            c = get_char_index(u + 1 - self.h_stack, v)
            (nu, nv) = self.control_points[c][0]
            return (nu + 1 - self.h_stack, nv)
        
        def connect_prev(u,v):
            """Like connect_next but going to the previous character."""
            c = get_char_index(u - 1 + self.h_stack, v)
            (pu, pv) = self.control_points[c][3]
            return (pu - 1 + self.h_stack, pv)

        oneminus_h = 1.0 - self.h_stack
        oneminus_v = 1.0 - self.v_stack
        def scale_u(u):
            """g(x) = floor(x/(1-h))+ (1-h) mod(x/(1-h),1)."""
            return (int(self.scale * u / (1.0 - self.h_stack))
                        + (1.0 - self.h_stack) *
                        ((self.scale * u  / (1.0 - self.h_stack)) % 1))
        
        def scale_v(v):
            return (int(self.scale * v / (1.0 - self.v_stack))
                        + (1.0 - self.v_stack) * (
                        (self.scale * v / (1.0 - self.v_stack)) % 1))
            
        def shift_u(u):
            """f(x) = floor((x-v)/(1-v)) + (1-v)mod((x-v)/(1-v),1) + v"""
            return (math.floor((self.scale * u - self.h_stack) / oneminus_h)
                    + oneminus_h *
                    (((self.scale * u - self.h_stack)  / oneminus_h) % 1)
                    + self.h_stack)

        def shift_v(v):
            return (math.floor((self.scale * v - self.v_stack) / oneminus_v) +
                    oneminus_v *
                    (((self.scale * v - self.v_stack)  / oneminus_v) % 1)
                    + self.v_stack)

        scaled_u = scale_u(u)
        scaled_v = scale_v(v)
        shifted_u = shift_u(u)
        shifted_v = shift_v(v)

        # cc: center, cl: left, ca: above, cal: above and to the left
        cc = get_char_index(scaled_u, scaled_v)
        uvc = np.array((scaled_u % 1, scaled_v % 1))
        cl = get_char_index(shifted_u, scaled_v)
        uvl = np.array((shifted_u % 1, scaled_v % 1))
        ca = get_char_index(scaled_u, shifted_v)
        uva = np.array((scaled_u  % 1, shifted_v % 1))
        cal = get_char_index(shifted_u, shifted_v)
        uval = np.array((shifted_u % 1, shifted_v % 1))

        # Check which of the four surrounding characters may occur.
        # We don't draw characters that can't fit entirely in the surface.
        in_cal = True
        if (scaled_u + 1 - self.h_stack >= scale_u(1.0)
                or scaled_v + 1 - self.v_stack >= scale_v(1.0)):
            in_cc = False
        else:
            in_cc = in_char(cc, uvc)

        if (shifted_u < 0.0
                or shifted_u + 1 - self.h_stack >= shift_u(1.0)
                or scaled_v + 1 - self.v_stack >= scale_v(1.0)):
            in_cl = False
            in_cal = False
        else:
            in_cl = in_char(cl, uvl)

        if (shifted_v < 0.0
                or scaled_u + 1 - self.h_stack >= scale_u(1.0)
                or shifted_v + 1 - self.v_stack >= shift_v(1.0)):
            in_ca = False
            in_cal = False
        else:
            in_ca = in_char(ca, uva)

        if in_cal:
            in_cal = in_char(cal, uval)
        return (in_cc or in_cl or in_ca or in_cal)

if __name__ == '__main__':
    h = Handwriting()
    print(h)
