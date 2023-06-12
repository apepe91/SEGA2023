class Matri:
    @staticmethod
    def multiplyMatrix(A, B):
        m = len(A)
        n = len(A[0])
        p = len(B[0])
        result = Matri.zeros2D(m, p)

        Bcolj = [0] * n
        for j in range(p):
            for k in range(n):
                Bcolj[k] = B[k][j]

            for i in range(m):
                s = 0
                for k in range(n):
                    s += A[i][k] * Bcolj[k]

                result[i][j] = s
        return result

    @staticmethod
    def zeros1D(rows):
        return [0] * rows

    @staticmethod
    def zeros2D(rows, columns):
        return [[0] * columns for _ in range(rows)]

    @staticmethod
    def zeros3D(rows, columns, depth):
        return [[[0] * depth for _ in range(columns)] for _ in range(rows)]

    @staticmethod
    def ones1D(rows):
        return [1] * rows

    @staticmethod
    def ones2D(rows, columns):
        return [[1] * columns for _ in range(rows)]

    @staticmethod
    def ones3D(rows, columns, depth):
        return [[[1] * depth for _ in range(columns)] for _ in range(rows)]
