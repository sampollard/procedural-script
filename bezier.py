""" Procedurally generated handwriting using bezier curves.
    The primary method is evaluate which takes as input two textures
    and returns the color at one or the other for each point (u,v)
"""

import copy
import inspect
import numpy as np
import random

class Texture():
    def __init__(self, eval_func=None):
        self.evaluate = eval_func
    
    def evaluate(self, u, v):
        """Determine the color given u,v coordinates"""
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
        strokes        The number of unique strokes.
        unique_chars   The number of different characters. May increase
                         if connect_p is nonzero.
        connect_p      The probability that one character will connect
                         with the next.
        concavity_pdf  The probability density function to determine
                         each stroke's inflection: negative numbers
                         represent downward-concaving strokes, positive
                         represent upward-concaving strokes, and 0 is a
                         straight line.
        min_s          The minimum number of strokes per character.
        max_s          The maximum number of strokes per character.
    """
    def __init__(self,
                 texture1=Texture(lambda u,v: (0,0,0)),
                 texture2=Texture(lambda u,v: (1,1,1)),
                 brush=lambda t: 0.1,
                 scale=1,
                 strokes=30,
                 unique_chars=25,
                 connect_p=0.8,
                 concavity_pdf=lambda: 0, # np.random.normal,
                 min_s=2,
                 max_s=3):
                       # Idea: Direction: Specifies which direction the
                       # text gets displayed. Default "lr", but also
                       # right to left, top-down left-right, top-down
                       # right-left, spiral out, spiral in, etc.
        self.tex1 = texture1     # Inside bezier curve
        self.tex2 = texture2     # Outside bezier curve
        self.scale = scale
        # strokes = len(control_points) is the number of unique strokes.
        # May increase when connect_p > 0.
        self.octaves = 9         # Accurate to dimension / 2^(octaves+1) pixels
        self.eps = lambda t: brush(t) / self.scale

        # Create a random hash table to determine which and how many
        # bezier curves are used for each character.
        random.seed(0)
        hash_sz = 257  # Choose a prime number for the hash size
        self.hash_table = list(range(hash_sz))
        random.shuffle(self.hash_table)

        # Create the control points. 4 control points = 1 bezier curve
        def cp_random():
            """Create a random pair [u,v] used as a control point"""
            return [0.25 + 0.5*random.random(), 0.25 + 0.5*random.random()]
        
        self.control_points = []
        for s in range(strokes):
            (p0, p3) = (np.array(cp_random()), np.array(cp_random()))
            # p1r = concavity_pdf()
            # p2r = concavity_pdf()
            # p1 = np.array(cp_random())*p1r + (1 - p1r()) * (p3 - p0/3)
            p1 = p3 - (1/3) * (p0 - p3)
            p2 = p3 - (2/3) * (p0 - p3)
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
            char = []
            for stroke in range(random.randint(min_s, max_s)):
                char.append(self.hash_table[stroke_counter % hash_sz] % strokes)
                stroke_counter += 1
                try:
                    used_strokes[char[-1]] += 1
                except KeyError:
                    used_strokes[char[-1]] = 1
            self.characters.append(char)
        
        # Connect the characters at the box edges (u = 1.0 or 0.0)
        # unless the character is at the end of a row
        cp = self.control_points
        for i in range(unique_chars - 1):
            if random.random() < connect_p and i % scale < scale - 1:
                first_stroke = self.characters[i][-1]
                second_stroke = self.characters[i + 1][0]
                # If the stroke is used in more than one place then make
                # a copy and modify it
                if used_strokes[first_stroke] > 1:
                    cp.append(copy.deepcopy(cp[first_stroke]))
                    used_strokes[first_stroke] -= 1
                    first_stroke = len(cp) - 1
                    used_strokes[first_stroke] = 1
                if used_strokes[second_stroke] > 1:
                    print(second_stroke)
                    cp.append(copy.deepcopy(cp[second_stroke]))
                    used_strokes[second_stroke] -= 1
                    second_stroke = len(cp) - 1
                    used_strokes[second_stroke] = 1

                cp[first_stroke][3]  = [1.0, cp[second_stroke][0][1]]
                cp[second_stroke][0] = [0.0, cp[second_stroke][0][1]]
                self.characters[i][-1] = first_stroke
                self.characters[i + 1][0] = second_stroke


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
        p = self.control_points[i]

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
            ((p0u,p0v),(p1u,p1v),(p2u,p2v),(p3u,p3v)) = self.control_points[i]
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

        scaled_u = self.scale * u
        scaled_v = self.scale * v
        pctu = (scaled_u) % 1
        pctv = (scaled_v) % 1
        # Move left to right in row-major order through the bezier curves
        i = (self.scale*int(scaled_v) + int(scaled_u)) % len(self.characters)
        uv = np.array((pctu,pctv))
        return any([in_curve_exact(uv, val) for val in self.characters[i]])
