from typing import List
import numbers
import sympy as sp
import numpy as np
import itertools as it

# Custom matrix class to handle required operations and cleanup bloat in other files.
class Matrix:
    # May have to upgrade to floats soon.
    matrix: List[List[int]]
    height: int
    length: int
    base: int = None

    def __init__(self, contents, direction="COLS",height: int=None, length: int=None, base=None):
        # Parameter checking.
        if height == None and length == None: raise Exception("Matrix must initialise with height or length.")
        if direction.upper() not in ("COLS", "ROWS"): raise Exception("Matrix has invalid direction.")
        if isinstance(contents, str): contents = contents.split()
        if any(not isinstance(item, (numbers.Number, str)) or (isinstance(item, str) and not item.isnumeric()) for item in contents):
            raise Exception("Matrix contents must be numeric.")
        
        # Dimension calculation and checking.
        m_len = len(contents)
        if m_len == 1 and (height != 1 or len != 1) and isinstance(contents[0], str):
            contents = list(contents[0])
            m_len = len(contents)
        if not height:
            if m_len % length != 0:
                raise Exception("Matrix contents have invalid dimensions.")
            else:
                height = m_len // length
        if not length:
            if m_len % height != 0:
                raise Exception("Matrix contents have invalid dimensions.")
            else:
                length = m_len // height
        if height * length != m_len:
            raise Exception("Matrix contents have invalid dimensions.")
        
        contents = [int(item) for item in contents]
        if base:
            self.base = base
            contents = [item % base for item in contents]
        
        self.height = height
        self.length = length
    
        self.matrix = [[None] * length for _ in range(height)]
        for index, value in enumerate(contents):
            if direction == "COLS":
                row = index % height
                col = index // height
            else:
                row = index // length
                col = index % length
            self.matrix[row][col] = value
    
    def nullspace(self):
        return [Matrix([item for row in vector.T.tolist() for item in row], length=1, base=self.base)
                for vector in sp.Matrix(self.matrix).nullspace()]
    
    def row_reduce(self):
        row_reduced, pivot = sp.Matrix(self.matrix).rref()
        return Matrix([item for row in row_reduced.T.tolist() for item in row], length=self.length, base=self.base)
    
    # Brute force generate all nullspace.
    def all_nullspace_bf(self):
        all_vectors = list(it.product(range(self.base), repeat=self.length))
        nullspace = []
        for vector in all_vectors:
            v_matrix = Matrix(list(vector), length=1, base=self.base)
            result = matrix * v_matrix
            if all(var == 0 for var in [item for row in result.matrix for item in row]):
                nullspace.append(v_matrix)

        return nullspace

    # Currently does not actually generate the whole nullspace. TODO: fix!!
    def all_nullspace(self):
        nullspace_vectors = self.nullspace()

        if not nullspace_vectors:
            return []

        combinations = set()
        num_vectors = len(nullspace_vectors)
        coefficients = range(self.base)

        for coeffs in it.product(coefficients, repeat=num_vectors):
            combination = Matrix([0] * (nullspace_vectors[0].height * nullspace_vectors[0].length),
                                height=nullspace_vectors[0].height,
                                length=nullspace_vectors[0].length,
                                base=self.base)
            for coeff, vector in zip(coeffs, nullspace_vectors):
                combination += vector * coeff

            combinations.add(tuple(map(tuple, combination.matrix)))

        return [Matrix(list(sum(combination, ())), height=nullspace_vectors[0].height, length=nullspace_vectors[0].length, base=self.base)
                for combination in combinations]

    def transpose(self):
        return Matrix([row[i] for row in self.matrix for i in range(self.length)], height = self.length, base=self.base)

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise Exception("Cannot add Matrix to non-Matrix.")
        if (self.length, self.height, self.base) == (other.length, other.height, other.base):
            flat_self = [item for row in self.matrix for item in row]
            flat_other = [item for row in other.matrix for item in row]
            flat_total = [flat_self[i] + flat_other[i] for i in range(len(flat_self))]
            return Matrix(flat_total, direction="ROWS", height=self.height, base=self.base)
        else:
            raise Exception("Incompatible dimensions for Matrix addition.")
            

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        else:
            return self.__dict__ == other.__dict__

    def __str__(self):
        if self.height == 0 or self.length == 0: return "[]"
        
        col_widths = [max(len(str(row[col])) for row in self.matrix) for col in range(self.length)]

        contents = ""
        for row in self.matrix: 
            contents += "["
            for index in range(self.length):
                contents += f" {row[index]:^{col_widths[index]}}"
            contents += " ]\n"
        return contents.rstrip()

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            contents = [item * other for row in self.matrix for item in row]
            return Matrix(contents, direction="ROWS", height = self.height, length = self.length, base = self.base)
        
        elif isinstance(other, Matrix):
            if self.length != other.height:
                raise Exception("Invalid dimensions for Matrix multiplication.")
            
            if self.base != other.base:
                raise Exception("Incompatible bases for Matrix multiplication")

            contents = [[0 for _ in range(other.length)] for _ in range(self.height)]

            for i in range(self.height):
                for j in range(other.length):
                    for k in range(self.length):
                        contents[i][j] += self.matrix[i][k] * other.matrix[k][j]
            
            return Matrix([item for row in contents for item in row], direction="ROWS", height=self.height, length=other.length, base = self.base)
            
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            contents = [item * other for row in self.matrix for item in row]
            return Matrix(contents, direction="ROWS", height = self.height, length = self.length, base = self.base)
        
        else:
            # Matrix multiplication handled by __mul__()
            return NotImplemented

matrix = Matrix("42024420", height=2, base=5)
answer = Matrix("0121", length=1, base=5)
print(matrix * answer)
all_nulls = matrix.all_nullspace_bf()
print(answer in all_nulls)
for null_vector in all_nulls:
    print(null_vector.transpose())