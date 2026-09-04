package layers;

import activations.ReLU;
import core.Matrix;
import exceptions.InvalidValue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LinearTest {

    @Test
    @DisplayName("Linear forward pass computes X W^T + b")
    public void testLinearForward() throws InvalidValue {
        Linear linear = new Linear(2, 2);
        linear.setWeight(0, 0, 1.0); linear.setWeight(0, 1, 2.0);
        linear.setWeight(1, 0, 3.0); linear.setWeight(1, 1, 4.0);
        linear.setBias(0, 0.5);
        linear.setBias(1, -0.5);

        Matrix X = new Matrix(new double[][]{{1.0, 1.0}});
        
        Matrix Z = linear.forward(X);

        assertArrayEquals(new int[]{1, 2}, Z.shape());
        assertEquals(3.5, Z.get(0, 0), 1e-12);
        assertEquals(6.5, Z.get(0, 1), 1e-12);
    }

    @Test
    @DisplayName("Linear backward pass computes dW, dB, and returns dX")
    public void testLinearBackward() throws InvalidValue {
        Linear linear = new Linear(2, 2);
        linear.setWeight(0, 0, 1.0); linear.setWeight(0, 1, 2.0);
        linear.setWeight(1, 0, 3.0); linear.setWeight(1, 1, 4.0);
        linear.setBias(0, 0.0);
        linear.setBias(1, 0.0);

        Matrix X = new Matrix(new double[][]{{2.0, 3.0}});
        linear.forward(X);

        Matrix dZ = new Matrix(new double[][]{{0.1, 0.2}});
        Matrix dX = linear.backward(dZ);

        assertArrayEquals(new int[]{1, 2}, dX.shape());
        assertEquals(0.7, dX.get(0, 0), 1e-12);
        assertEquals(1.0, dX.get(0, 1), 1e-12);

        Matrix dW = linear.getDW();
        assertEquals(0.2, dW.get(0, 0), 1e-12);
        assertEquals(0.3, dW.get(0, 1), 1e-12);
        assertEquals(0.4, dW.get(1, 0), 1e-12);
        assertEquals(0.6, dW.get(1, 1), 1e-12);

        Matrix dB = linear.getDB();
        assertEquals(0.1, dB.get(0, 0), 1e-12);
        assertEquals(0.2, dB.get(0, 1), 1e-12);
    }

    @Test
    @DisplayName("Layer composes Linear and Activation forward and backward")
    public void testLayerComposition() throws InvalidValue {
        Layer layer = new Layer(2, 2, new ReLU());
        layer.getLinear().setWeight(0, 0, 1.0); layer.getLinear().setWeight(0, 1, 0.0);
        layer.getLinear().setWeight(1, 0, -1.0); layer.getLinear().setWeight(1, 1, 0.0);
        layer.getLinear().setBias(0, 0.0);
        layer.getLinear().setBias(1, 0.0);

        Matrix X = new Matrix(new double[][]{{5.0, 0.0}});
        Matrix A = layer.forward(X);

        assertEquals(5.0, A.get(0, 0), 1e-12);
        assertEquals(0.0, A.get(0, 1), 1e-12);

        Matrix dLA = new Matrix(new double[][]{{1.0, 1.0}});
        Matrix dX = layer.backward(dLA);

        assertEquals(1.0, dX.get(0, 0), 1e-12);
        assertEquals(0.0, dX.get(0, 1), 1e-12);
    }
}
