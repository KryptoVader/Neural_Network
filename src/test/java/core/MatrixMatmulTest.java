package core;

import exceptions.InvalidValue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MatrixMatmulTest {

    @Test
    @DisplayName("Matrix multiplication matches independently computed mathematical ground truth")
    public void testMatmulCorrectness() throws InvalidValue {
        
        Matrix A = new Matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        });
        
        Matrix B = new Matrix(new double[][]{
            {7.0, 8.0},
            {9.0, 1.0},
            {2.0, 3.0}
        });

        Matrix C = Matrix.matmul(A, B);

        assertArrayEquals(new int[]{2, 2}, C.shape());
        assertEquals(31.0, C.get(0, 0), 1e-12);
        assertEquals(19.0, C.get(0, 1), 1e-12);
        assertEquals(85.0, C.get(1, 0), 1e-12);
        assertEquals(55.0, C.get(1, 1), 1e-12);
    }

    @Test
    @DisplayName("Matmul 1x1 matrices")
    public void testMatmul1x1() throws InvalidValue {
        Matrix A = new Matrix(new double[][]{{3.5}});
        Matrix B = new Matrix(new double[][]{{2.0}});
        Matrix C = Matrix.matmul(A, B);

        assertArrayEquals(new int[]{1, 1}, C.shape());
        assertEquals(7.0, C.get(0, 0), 1e-12);
    }

    @Test
    @DisplayName("Matmul outer product (3x1 * 1x3 = 3x3)")
    public void testMatmulOuterProduct() throws InvalidValue {
        Matrix u = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}});
        Matrix v = new Matrix(new double[][]{{4.0, 5.0, 6.0}});

        Matrix C = Matrix.matmul(u, v);
        assertArrayEquals(new int[]{3, 3}, C.shape());

        double[][] expected = {
            {4.0, 5.0, 6.0},
            {8.0, 10.0, 12.0},
            {12.0, 15.0, 18.0}
        };
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expected[i][j], C.get(i, j), 1e-12);
            }
        }
    }

    @Test
    @DisplayName("Matmul inner product (1x3 * 3x1 = 1x1)")
    public void testMatmulInnerProduct() throws InvalidValue {
        Matrix u = new Matrix(new double[][]{{1.0, 2.0, 3.0}});
        Matrix v = new Matrix(new double[][]{{4.0}, {5.0}, {6.0}});

        Matrix C = Matrix.matmul(u, v);
        assertArrayEquals(new int[]{1, 1}, C.shape());
        assertEquals(32.0, C.get(0, 0), 1e-12);
    }

    @Test
    @DisplayName("Matmul with Identity matrix preserves matrix")
    public void testMatmulIdentity() throws InvalidValue {
        Matrix A = new Matrix(new double[][]{
            {2.0, -1.0, 5.0},
            {0.0, 4.0, -3.0}
        });
        Matrix I3 = Matrix.identity(3);
        Matrix res = Matrix.matmul(A, I3);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(A.get(i, j), res.get(i, j), 1e-12);
            }
        }
    }

    @Test
    @DisplayName("Matmul throws InvalidValue for dimension mismatch")
    public void testMatmulDimensionMismatch() {
        assertThrows(InvalidValue.class, () -> {
            Matrix A = new Matrix(2, 3);
            Matrix B = new Matrix(4, 2);
            Matrix.matmul(A, B);
        });
    }
}
