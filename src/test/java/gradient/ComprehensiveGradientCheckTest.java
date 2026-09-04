package gradient;

import activations.ReLU;
import activations.Sigmoid;
import activations.Softmax;
import core.Matrix;
import exceptions.InvalidValue;
import layers.Layer;
import losses.CrossEntropy;
import losses.MSE;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class ComprehensiveGradientCheckTest {

    private static Matrix randomMatrix(int r, int c, Random rng) throws InvalidValue {
        Matrix m = new Matrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                m.set(i, j, rng.nextDouble() * 2.0 - 1.0);
            }
        }
        return m;
    }

    @Test
    @DisplayName("Linear + MSE gradient check across multiple random seeds")
    public void testLinearMSEGradientCheck() throws InvalidValue {
        int[] seeds = {42, 101, 2024};
        for (int seed : seeds) {
            Random rng = new Random(seed);
            Matrix X = randomMatrix(2, 3, rng);
            Matrix Y = randomMatrix(2, 2, rng);

            Layer layer = new Layer(3, 2);
            GradientCheckResult wResult = GradientChecker.checkLayerWeights(layer, new MSE(), X, Y, 1e-6);
            GradientCheckResult bResult = GradientChecker.checkLayerBiases(layer, new MSE(), X, Y, 1e-6);

            assertTrue(wResult.maxRelError < 1e-5, "Weight rel error too large: " + wResult);
            assertTrue(bResult.maxRelError < 1e-5, "Bias rel error too large: " + bResult);
        }
    }

    @Test
    @DisplayName("Linear + ReLU + MSE gradient check across multiple random seeds")
    public void testLinearReLUMSEGradientCheck() throws InvalidValue {
        int[] seeds = {7, 888, 9999};
        for (int seed : seeds) {
            Random rng = new Random(seed);
            Matrix X = randomMatrix(3, 4, rng);
            Matrix Y = randomMatrix(3, 2, rng);

            Layer layer = new Layer(4, 2, new ReLU());
            GradientCheckResult wResult = GradientChecker.checkLayerWeights(layer, new MSE(), X, Y, 1e-6);
            GradientCheckResult bResult = GradientChecker.checkLayerBiases(layer, new MSE(), X, Y, 1e-6);

            assertTrue(wResult.maxRelError < 1e-5, "Weight rel error too large: " + wResult);
            assertTrue(bResult.maxRelError < 1e-5, "Bias rel error too large: " + bResult);
        }
    }

    @Test
    @DisplayName("Linear + Sigmoid + MSE gradient check across multiple random seeds")
    public void testLinearSigmoidMSEGradientCheck() throws InvalidValue {
        int[] seeds = {11, 222, 3333};
        for (int seed : seeds) {
            Random rng = new Random(seed);
            Matrix X = randomMatrix(2, 3, rng);
            Matrix Y = randomMatrix(2, 2, rng);

            Layer layer = new Layer(3, 2, new Sigmoid());
            GradientCheckResult wResult = GradientChecker.checkLayerWeights(layer, new MSE(), X, Y, 1e-6);
            GradientCheckResult bResult = GradientChecker.checkLayerBiases(layer, new MSE(), X, Y, 1e-6);

            assertTrue(wResult.maxRelError < 1e-5, "Weight rel error too large: " + wResult);
            assertTrue(bResult.maxRelError < 1e-5, "Bias rel error too large: " + bResult);
        }
    }

    @Test
    @DisplayName("Linear + Softmax + CrossEntropy gradient check across multiple random seeds")
    public void testLinearSoftmaxCrossEntropyGradientCheck() throws InvalidValue {
        int[] seeds = {19, 456, 7890};
        for (int seed : seeds) {
            Random rng = new Random(seed);
            Matrix X = randomMatrix(3, 4, rng);
            
            Matrix Y = new Matrix(3, 3);
            Y.set(0, 0, 1.0);
            Y.set(1, 2, 1.0);
            Y.set(2, 1, 1.0);

            Layer layer = new Layer(4, 3, new Softmax());
            GradientCheckResult wResult = GradientChecker.checkLayerWeights(layer, new CrossEntropy(), X, Y, 1e-6);
            GradientCheckResult bResult = GradientChecker.checkLayerBiases(layer, new CrossEntropy(), X, Y, 1e-6);

            assertTrue(wResult.maxRelError < 1e-5, "Weight rel error too large: " + wResult);
            assertTrue(bResult.maxRelError < 1e-5, "Bias rel error too large: " + bResult);
        }
    }
}
