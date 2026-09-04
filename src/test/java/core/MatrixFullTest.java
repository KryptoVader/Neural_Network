package core;

import exceptions.InvalidValue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MatrixFullTest {

    @Test
    @DisplayName("Matrix addition and subtraction")
    public void testAddSub() throws InvalidValue {
        Matrix A = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        Matrix B = new Matrix(new double[][]{{5.0, 6.0}, {7.0, 8.0}});

        Matrix sum = Matrix.add(A, B);
        assertEquals(6.0, sum.get(0, 0), 1e-12);
        assertEquals(8.0, sum.get(0, 1), 1e-12);
        assertEquals(10.0, sum.get(1, 0), 1e-12);
        assertEquals(12.0, sum.get(1, 1), 1e-12);

        Matrix diff = Matrix.sub(B, A);
        assertEquals(4.0, diff.get(0, 0), 1e-12);
        assertEquals(4.0, diff.get(0, 1), 1e-12);
        assertEquals(4.0, diff.get(1, 0), 1e-12);
        assertEquals(4.0, diff.get(1, 1), 1e-12);
    }

    @Test
    @DisplayName("Matrix transpose swaps dimensions and elements")
    public void testTranspose() throws InvalidValue {
        Matrix A = new Matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        });
        Matrix At = Matrix.Transpose(A);

        assertArrayEquals(new int[]{3, 2}, At.shape());
        assertEquals(1.0, At.get(0, 0), 1e-12);
        assertEquals(4.0, At.get(0, 1), 1e-12);
        assertEquals(2.0, At.get(1, 0), 1e-12);
        assertEquals(5.0, At.get(1, 1), 1e-12);
        assertEquals(3.0, At.get(2, 0), 1e-12);
        assertEquals(6.0, At.get(2, 1), 1e-12);
    }

    @Test
    @DisplayName("Hadamard product multiplies element-wise")
    public void testHadamard() throws InvalidValue {
        Matrix A = new Matrix(new double[][]{{2.0, 3.0}, {4.0, 5.0}});
        Matrix B = new Matrix(new double[][]{{10.0, 20.0}, {30.0, 40.0}});

        Matrix H = Matrix.Hadamard(A, B);
        assertEquals(20.0, H.get(0, 0), 1e-12);
        assertEquals(60.0, H.get(0, 1), 1e-12);
        assertEquals(120.0, H.get(1, 0), 1e-12);
        assertEquals(200.0, H.get(1, 1), 1e-12);
    }

    @Test
    @DisplayName("Row slicing with range and indices")
    public void testRowSlicing() throws InvalidValue {
        Matrix A = new Matrix(new double[][]{
            {10.0, 11.0},
            {20.0, 21.0},
            {30.0, 31.0},
            {40.0, 41.0}
        });

        Matrix sliceRange = A.getRows(1, 3);
        assertArrayEquals(new int[]{2, 2}, sliceRange.shape());
        assertEquals(20.0, sliceRange.get(0, 0), 1e-12);
        assertEquals(31.0, sliceRange.get(1, 1), 1e-12);

        Matrix sliceIndices = A.getRows(new int[]{3, 0});
        assertArrayEquals(new int[]{2, 2}, sliceIndices.shape());
        assertEquals(40.0, sliceIndices.get(0, 0), 1e-12);
        assertEquals(10.0, sliceIndices.get(1, 0), 1e-12);
    }

    @Test
    @DisplayName("Dot product of 1D vectors")
    public void testDot() throws InvalidValue {
        double[] v1 = {1.0, 2.0, 3.0};
        double[] v2 = {4.0, -5.0, 6.0};

        double d = Matrix.dot(v1, v2);
        
        assertEquals(12.0, d, 1e-12);
    }

    @Test
    @DisplayName("Shape validation throws exceptions on invalid inputs")
    public void testShapeValidation() {
        assertThrows(InvalidValue.class, () -> new Matrix(0));
        assertThrows(InvalidValue.class, () -> new Matrix(-1));
        assertThrows(InvalidValue.class, () -> {
            Matrix A = new Matrix(2, 3);
            Matrix B = new Matrix(3, 2);
            Matrix.add(A, B);
        });
        assertThrows(InvalidValue.class, () -> {
            double[] v1 = {1.0, 2.0};
            double[] v2 = {1.0, 2.0, 3.0};
            Matrix.dot(v1, v2);
        });
    }
}
