package activations;

import core.Matrix;
import exceptions.InvalidValue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ActivationTest {

    @Test
    @DisplayName("ReLU forward and backward on positive, negative, and zero values")
    public void testReLU() throws InvalidValue {
        ReLU relu = new ReLU();
        Matrix Z = new Matrix(new double[][]{{-2.5, 0.0, 3.5}});
        Matrix A = relu.forward(Z);

        assertEquals(0.0, A.get(0, 0), 1e-12);
        assertEquals(0.0, A.get(0, 1), 1e-12);
        assertEquals(3.5, A.get(0, 2), 1e-12);

        Matrix dLA = new Matrix(new double[][]{{1.0, 1.0, 1.0}});
        Matrix dZ = relu.backward(dLA, Z);

        assertEquals(0.0, dZ.get(0, 0), 1e-12, "Subgradient for z < 0 must be 0");
        assertEquals(0.0, dZ.get(0, 1), 1e-12, "Subgradient for z == 0 evaluated to 0");
        assertEquals(1.0, dZ.get(0, 2), 1e-12, "Gradient for z > 0 must be 1.0");
    }

    @Test
    @DisplayName("Sigmoid forward and backward")
    public void testSigmoid() throws InvalidValue {
        Sigmoid sigmoid = new Sigmoid();
        Matrix Z = new Matrix(new double[][]{{0.0, 1000.0, -1000.0}});
        Matrix A = sigmoid.forward(Z);

        assertEquals(0.5, A.get(0, 0), 1e-12);
        assertEquals(1.0, A.get(0, 1), 1e-6);
        assertEquals(0.0, A.get(0, 2), 1e-6);

        Matrix dLA = new Matrix(new double[][]{{1.0, 1.0, 1.0}});
        Matrix dZ = sigmoid.backward(dLA, Z);

        assertEquals(0.25, dZ.get(0, 0), 1e-12);
        
        assertEquals(0.0, dZ.get(0, 1), 1e-6);
        assertEquals(0.0, dZ.get(0, 2), 1e-6);
    }

    @Test
    @DisplayName("Softmax normalization and invariance to constant logit shift")
    public void testSoftmaxInvariance() throws InvalidValue {
        Softmax sm = new Softmax();
        Matrix Z1 = new Matrix(new double[][]{{1.0, 2.0, 3.0}});
        Matrix Z2 = new Matrix(new double[][]{{1001.0, 1002.0, 1003.0}});

        Matrix P1 = sm.forward(Z1);
        Matrix P2 = sm.forward(Z2);

        for (int c = 0; c < 3; c++) {
            assertEquals(P1.get(0, c), P2.get(0, c), 1e-10, "Softmax must be invariant to constant shift");
        }

        double sum = P1.get(0, 0) + P1.get(0, 1) + P1.get(0, 2);
        assertEquals(1.0, sum, 1e-12);
    }
}
