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

class BBox():
    def __init__(self, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
    def __repr__(self):
        return "([{:.4f},{:.4f}], [{:.4f},{:.4f}])".format(
                self.xmin, self.xmax, self.ymin, self.ymax)


class Character():
    """ Primarily a list of Bezier curves.
        Assumes transform does not change after initialization.
        BBox is in normalized coordinates (no transform applied)
    """
    @staticmethod
    def T(T, stroke):
        """Applies a 3x3 transformation matrix to each 2d point in stroke."""
        b = np.array(stroke)
        b = np.append(b, np.ones((len(stroke), 1)), axis=1)
        Tb = np.dot(T, np.transpose(b))
        return np.transpose(np.delete(Tb, 2, axis=0))

    def __init__(self, strokes=None, transform=np.identity(3)):
        self.transform = transform
        if strokes:
            self.strokes = list(strokes)
            # Compute bounding box
            xmin = 1.0
            xmax = 0.0
            ymin = 1.0
            ymax = 0.0
            for stroke in self.strokes:
                xmin = min(xmin, min([p[0] for p in stroke]))
                xmax = max(xmax, max([p[0] for p in stroke]))
                ymin = min(ymin, min([p[1] for p in stroke]))
                ymax = max(ymax, max([p[1] for p in stroke]))

            self.bbox = BBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            # TODO: Add max(brush(t)) so that the BBox doesn't cut things off.
        else:
            self.strokes = []
            self.bbox = BBox()

    def __getitem__(self, index):
        """ Turn the Bezier curve into homogeneous coordinates, apply
            transformation, then convert back to normal coordinates.
        """
        return Character.T(self.transform, self.strokes[index])

    def __setitem__(self, index, item):
        self.strokes[index] = item
    
    def __repr__(self):
        cp_str = "[\n"
        for stroke in self.strokes:
            for p in stroke:
                cp_str += " ({:.4f}, {:.4f})".format(p[0], p[1])
            cp_str += "\n"
        cp_str += "]"
        return "T =\n{}\nStrokes = {}\nBBox = {}".format(
            self.transform, cp_str, self.bbox)

    def append(self, s):
        self.strokes.append(s)

    def link_next(self, next_cp):
        """Link from this character's final stroke to next_cp."""
        uv = list(self.strokes[-1][3])
        if next_cp:
            return  [uv,
                     [uv[0] + (next_cp[0] - uv[0]) / 3,
                      min(uv[1], next_cp[1])],
                     [uv[0] + 2*(next_cp[0] - uv[0]) / 3,
                      min(uv[1], next_cp[1])],
                     list(next_cp)]
        else:
            return [] # XXX: Should this throw an exception?

    def link_prev(self, prev_cp):
        """Link from this character's first stroke to prev_cp."""
        uv = list(self.strokes[-1][0])
        if prev_cp:
            return  [list(prev_cp),
                     [prev_cp[0] + (uv[0] - prev_cp[0]) / 3,
                      min(uv[1], prev_cp[1])],
                     [prev_cp[0] + 2*(uv[0] - prev_cp[0]) / 3,
                      min(uv[1], prev_cp[1])],
                     uv]
        else:
            return [] # XXX: Should this throw an exception?
    
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


class Acceleration():
    def overlap(self, u, v):
        """ Determine which objects could potentially appear
            at position (u,v).
        """
        pass

class GridAcceleration(Acceleration):
    def __init__(self, objs, xdivs=10, ydivs=10):
        self.grid = []
        self.xlattice = np.linspace(0.0, 1.0, xdivs, endpoint=False)
        self.ylattice = np.linspace(0.0, 1.0, ydivs, endpoint=False)
        self.xsep = 1.0 - self.xlattice[-1]
        self.ysep = 1.0 - self.ylattice[-1]
        for x in self.xlattice:
            row = []
            for y in self.ylattice:
                cell = []
                for o in objs:
                    if not (o.bbox.xmax < x
                            or o.bbox.ymax < y
                            or o.bbox.xmin > x + self.xsep
                            or o.bbox.ymin > y + self.ysep):
                        cell.append(o)
                row.append(cell)
            self.grid.append(row)
    
    def overlap(self, u, v):
        """Expects (u,v) to be normalized (in [0,1]^2)"""
        xgrid = int(u / self.xsep) if u != 1.0 else self.xlattice[-1]
        ygrid = int(v / self.ysep) if v != 1.0 else self.ylattice[-1]
        return self.grid[xgrid][ygrid]

class Handwriting(Texture):
    """ Explicit representation of a handwriting texture.
        Returns texture1 if the point (u,v) is inside the handwriting
        and texture2 otherwise. Here, a stroke denotes a bezier curve
        and character denotes a collection of one or more strokes
        overlaid in the same domain.

        Input parameters:
        brush          Function to determine the width of the stroke.
                         May depend on the parameter of the spline.
        scale          One character will be in a square with sides
                         (1/scale, 1/scale) of screen width
        aspect         The aspect ratio of the surface: width / height
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
                 aspect=1.0,
                 seed=0,
                 unique_strokes=30,
                 unique_chars=25,
                 min_s=2,
                 max_s=3,
                 v_stack=0.3,
                 h_stack=0.3,
                 h_border=0.2,
                 v_border=0.2):
        # Ideas: Additional Parameters: Direction: l->r, r->l
        
        # Validate parameters
        # TODO: Change 0.5 to 1.0
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        if aspect <= 0.0:
            raise ValueError("aspect must be positive")
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
        self.octaves = 9         # Accurate to dimension / 2^(octaves+1) pixels
        self.brush = lambda t: brush(t) / self.scale
        self.h_stack = h_stack
        self.v_stack = v_stack

        # Create a random hash table
        random.seed(seed)
        hash_sz = 257            # Choose a prime number for the hash size
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

        # Create each character archetype.
        self.characters = []
        used_strokes = {}
        stroke_counter = 0
        for i in range(unique_chars):
            char = Character()
            for stroke in range(random.randint(min_s, max_s)):
                index = self.hash_table[stroke_counter % hash_sz] % unique_strokes
                char.append(self.control_points[index])
                stroke_counter += 1
                try:
                    used_strokes[index] += 1
                except KeyError:
                    used_strokes[index] = 1
            self.characters.append(char)

        # Instance the characters.
        # Determine their transformation and insert into the
        # acceleration structure
        u_trans = 0.0
        v_trans = 0.0
        transform = np.array(((1.0/self.scale, 0.0, u_trans),
                              (0.0, 1.0/self.scale, v_trans),
                              (0.0, 0.0, 1.0)))
        charlist = []
        for u in range(int(scale)):
            for v in range(int(scale * aspect)):
                char = Character(self.characters[self.hash_table[
                        (u * int(scale) + v) % hash_sz] % unique_chars].strokes,
                        transform)
                charlist.append(char)
                transform = np.copy(transform)
                # TODO: Fix translation
                transform[0][2] += self.h_stack / scale
            transform[0][2] = 0.0
            transform[1][2] += self.v_stack / scale
        
        for c in charlist:
            print(c)
        self.grid = GridAcceleration(charlist)


    def __repr__(self):
        """ Computing the representation is O(cn) where n is the
            number of characters and c is the number of control points.
        """
        def bezier_eq(b1, b2):
            """Test if two bezier curves are equal."""
            return all([p1[0] == p2[0] and p1[1] == p2[1] for
                       (p1,p2) in zip(b1, b2)])

        cp_str = "[\n"
        for ind, cp in enumerate(self.control_points):
            cp_str += "{:3d}:".format(ind)
            for p in cp:
                cp_str += " ({:.4f}, {:.4f})".format(p[0], p[1])
            cp_str += "\n"
        cp_str += "]"

        char_str = "[\n"
        for ind, char in enumerate(self.characters):
            char_str += "{:3d}:(".format(ind)
            for stroke in char.strokes:
                for ind, cp in enumerate(self.control_points):
                    found = False
                    if bezier_eq(cp, stroke):
                        found = True
                        char_str += " {}".format(ind)
                        break
                if not found:
                    char_str += " ({})".format(stroke)
            char_str += " )\n"
        char_str += "]"

        return "control_points: {}\ncharacters: {}\neps: {}".format(
                cp_str, char_str, inspect.getsource(self.brush).strip())

    def evaluate(self, u, v):
        """ The primary function. Determine whether the point is
            inside any of the bezier curves.
        """
        if self.inside_curve(u,v):
            return self.tex1.evaluate(u,v)
        else:
            return self.tex2.evaluate(u,v)

    def B(self, t, p):
        """Evaluate the ith bezier curve at t."""
        # Coefficients computed using coeffs(collect(B),t) in Matlab
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
                           np.linalg.norm(uv - self.B(t,i)) < self.brush(t)
                        ) or (
                        t > 1.0 and
                            np.linalg.norm(uv-self.B(1.0,i)) < self.brush(1.0)
                        ) or (
                        t < 0.0 and
                            np.linalg.norm(uv-self.B(0.0,i)) < self.brush(0.0)):
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
            if np.linalg.norm(uv - self.B(t,i)) < self.brush(t):
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
                
        def in_char(c, uv):
            """ Given a decimal position of the character, find out
                if the point is in the character or any decorations.
            """
            # TODO: Apply transformation
            stroke_list = c.strokes
            return any([in_curve_exact(uv, val) for val in
                       stroke_list])

        ### inside_curve body
        # Use the acceleration structure to determine which
        # character(s) to check, then check each stroke in each character.
        chars = self.grid.overlap(u, v)
        if not chars:
            return False
        else:
            return any([in_char(c, (u, v)) for c in chars])


if __name__ == '__main__':
    h = Handwriting(scale=3, aspect=0.5)
    print(h)
