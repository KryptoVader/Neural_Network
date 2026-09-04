package losses;

import core.Matrix;
import exceptions.InvalidValue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LossTest {

    @Test
    @DisplayName("MSE forward and derivative matches mathematical formulation")
    public void testMSE() throws InvalidValue {
        MSE mse = new MSE();
        Matrix Y = new Matrix(new double[][]{{1.0, 2.0}});
        Matrix Pred = new Matrix(new double[][]{{1.5, 1.0}});

        double loss = mse.forward(Y, Pred);
        assertEquals(0.3125, loss, 1e-12);

        Matrix grad = mse.derivative(Y, Pred);
        assertEquals(0.25, grad.get(0, 0), 1e-12);
        assertEquals(-0.5, grad.get(0, 1), 1e-12);
    }

    @Test
    @DisplayName("CrossEntropy handles boundary probabilities without throwing or producing NaN/Inf")
    public void testCrossEntropyBoundaries() throws InvalidValue {
        CrossEntropy ce = new CrossEntropy();
        Matrix Y = new Matrix(new double[][]{{1.0, 0.0}});
        Matrix PredZero = new Matrix(new double[][]{{0.0, 1.0}});

        double lossZero = ce.forward(Y, PredZero);
        assertTrue(Double.isFinite(lossZero), "Loss must be finite for p=0");

        Matrix gradZero = ce.derivative(Y, PredZero);
        assertTrue(Double.isFinite(gradZero.get(0, 0)), "Gradient must be finite for p=0");
        assertTrue(Double.isFinite(gradZero.get(0, 1)), "Gradient must be finite for p=0");
    }
}
