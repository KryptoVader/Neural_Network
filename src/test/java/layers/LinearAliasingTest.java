package layers;

import core.Matrix;
import exceptions.InvalidValue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LinearAliasingTest {

    @Test
    @DisplayName("Linear backward must not be corrupted if input matrix X is mutated externally after forward")
    public void testLinearInputAliasing() throws InvalidValue {
        Linear linear = new Linear(2, 2);
        
        linear.setWeight(0, 0, 1.0); linear.setWeight(0, 1, 0.0);
        linear.setWeight(1, 0, 0.0); linear.setWeight(1, 1, 1.0);
        linear.setBias(0, 0.0);
        linear.setBias(1, 0.0);

        Matrix X = new Matrix(new double[][]{{1.0, 2.0}});
        Matrix dZ = new Matrix(new double[][]{{1.0, 1.0}});

        linear.forward(X);

        X.set(0, 0, 999.0);
        X.set(0, 1, 888.0);

        linear.backward(dZ);

        Matrix dW = linear.getDW();

        assertEquals(1.0, dW.get(0, 0), 1e-10, "dW[0, 0] should reflect original X[0, 0] = 1.0, not mutated 999.0");
        assertEquals(2.0, dW.get(0, 1), 1e-10, "dW[0, 1] should reflect original X[0, 1] = 2.0, not mutated 888.0");
    }

    @Test
    @DisplayName("Matrix.getRow returns defensive copy so mutating row array does not corrupt matrix")
    public void testMatrixRowAliasing() throws InvalidValue {
        Matrix m = new Matrix(new double[][]{{10.0, 20.0}, {30.0, 40.0}});
        double[] row0 = m.getRow(0);
        row0[0] = -999.0;

        assertEquals(10.0, m.get(0, 0), 1e-10, "Matrix element must remain 10.0 after mutating returned row array");
    }
}
