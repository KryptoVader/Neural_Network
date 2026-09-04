package core;

import exceptions.InvalidValue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MatrixTest {

    @Test
    @DisplayName("Augment preserves multiple rows and columns while appending bias column of 1s")
    public void testAugmentNormalMatrix() throws InvalidValue {
        Matrix m = new Matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        });

        Matrix aug = Matrix.augment(m);

        assertArrayEquals(new int[]{2, 4}, aug.shape(), "Shape should be [2, 4]");

        assertEquals(1.0, aug.get(0, 0), 1e-12);
        assertEquals(2.0, aug.get(0, 1), 1e-12);
        assertEquals(3.0, aug.get(0, 2), 1e-12);
        assertEquals(4.0, aug.get(1, 0), 1e-12);
        assertEquals(5.0, aug.get(1, 1), 1e-12);
        assertEquals(6.0, aug.get(1, 2), 1e-12);

        assertEquals(1.0, aug.get(0, 3), 1e-12);
        assertEquals(1.0, aug.get(1, 3), 1e-12);
    }

    @Test
    @DisplayName("Augment 1x1 matrix")
    public void testAugment1x1() throws InvalidValue {
        Matrix m = new Matrix(new double[][]{{42.0}});
        Matrix aug = Matrix.augment(m);

        assertArrayEquals(new int[]{1, 2}, aug.shape());
        assertEquals(42.0, aug.get(0, 0), 1e-12);
        assertEquals(1.0, aug.get(0, 1), 1e-12);
    }

    @Test
    @DisplayName("Augment single-row matrix")
    public void testAugmentSingleRow() throws InvalidValue {
        Matrix m = new Matrix(new double[][]{{3.14, 2.71, -1.0}});
        Matrix aug = Matrix.augment(m);

        assertArrayEquals(new int[]{1, 4}, aug.shape());
        assertEquals(3.14, aug.get(0, 0), 1e-12);
        assertEquals(2.71, aug.get(0, 1), 1e-12);
        assertEquals(-1.0, aug.get(0, 2), 1e-12);
        assertEquals(1.0, aug.get(0, 3), 1e-12);
    }

    @Test
    @DisplayName("Augment single-column matrix")
    public void testAugmentSingleColumn() throws InvalidValue {
        Matrix m = new Matrix(new double[][]{{10.0}, {20.0}, {30.0}});
        Matrix aug = Matrix.augment(m);

        assertArrayEquals(new int[]{3, 2}, aug.shape());
        assertEquals(10.0, aug.get(0, 0), 1e-12);
        assertEquals(20.0, aug.get(1, 0), 1e-12);
        assertEquals(30.0, aug.get(2, 0), 1e-12);
        assertEquals(1.0, aug.get(0, 1), 1e-12);
        assertEquals(1.0, aug.get(1, 1), 1e-12);
        assertEquals(1.0, aug.get(2, 1), 1e-12);
    }
}
