package losses;

import activations.Softmax;
import core.Matrix;
import core.NeuralNetwork;
import exceptions.InvalidValue;
import layers.Layer;
import optimizers.SGD;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SoftmaxCrossEntropyTest {

    @Test
    @DisplayName("Fused Softmax + CrossEntropy produces exact (p - y)/N gradient for ordinary predictions")
    public void testOrdinaryPredictions() throws InvalidValue {
        
        Matrix Z = new Matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {0.5, -0.5, 1.5}
        });
        
        Matrix Y = new Matrix(new double[][]{
            {0.0, 0.0, 1.0},
            {1.0, 0.0, 0.0}
        });

        Softmax softmax = new Softmax();
        Matrix P = softmax.forward(Z);

        int N = Y.shape()[0];
        int O = Y.shape()[1];

        Matrix expectedDZ = new Matrix(N, O);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < O; j++) {
                expectedDZ.set(i, j, (P.get(i, j) - Y.get(i, j)) / N);
            }
        }

        Layer layer = new Layer(3, 3, new Softmax());
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                layer.getLinear().setWeight(i, j, i == j ? 1.0 : 0.0);
            }
            layer.getLinear().setBias(i, 0.0);
        }

        NeuralNetwork nn = new NeuralNetwork(new Layer[]{layer}, new CrossEntropy(), new SGD(0.1));
        double loss = nn.train(Z, Y);

        assertTrue(Double.isFinite(loss), "Loss must be finite");
        assertTrue(loss > 0, "Loss must be positive");

        Matrix dB = layer.getLinear().getDB();
        for (int j = 0; j < O; j++) {
            double expectedSum = 0;
            for (int i = 0; i < N; i++) {
                expectedSum += expectedDZ.get(i, j);
            }
            assertEquals(expectedSum, dB.get(0, j), 1e-10, "Bias gradient must match sum of dZ");
        }
    }

    @Test
    @DisplayName("Confident incorrect prediction does not suffer from gradient vanishing")
    public void testConfidentIncorrectPredictionGradient() throws InvalidValue {
        
        Matrix X = new Matrix(new double[][]{{1.0, 0.0}});
        Matrix Y = new Matrix(new double[][]{{1.0, 0.0}});

        Layer layer = new Layer(2, 2, new Softmax());
        
        layer.getLinear().setWeight(0, 0, -50.0); layer.getLinear().setWeight(0, 1, 0.0);
        layer.getLinear().setWeight(1, 0, 50.0);  layer.getLinear().setWeight(1, 1, 0.0);
        layer.getLinear().setBias(0, 0.0);
        layer.getLinear().setBias(1, 0.0);

        NeuralNetwork nn = new NeuralNetwork(new Layer[]{layer}, new CrossEntropy(), new SGD(0.1));
        double loss = nn.train(X, Y);

        assertTrue(Double.isFinite(loss), "Loss must be finite under extreme logits");
        assertTrue(loss > 25.0, "Loss should be high for confident wrong prediction");

        Matrix dB = layer.getLinear().getDB();
        assertEquals(-1.0, dB.get(0, 0), 1e-5, "Gradient on true class logit must be approx -1.0, NOT 0.0!");
        assertEquals(1.0, dB.get(0, 1), 1e-5, "Gradient on false class logit must be approx +1.0");
    }

    @Test
    @DisplayName("Confident correct prediction has near-zero gradient")
    public void testConfidentCorrectPredictionGradient() throws InvalidValue {
        Matrix X = new Matrix(new double[][]{{1.0, 0.0}});
        Matrix Y = new Matrix(new double[][]{{1.0, 0.0}});

        Layer layer = new Layer(2, 2, new Softmax());
        layer.getLinear().setWeight(0, 0, 50.0);  layer.getLinear().setWeight(0, 1, 0.0);
        layer.getLinear().setWeight(1, 0, -50.0); layer.getLinear().setWeight(1, 1, 0.0);
        layer.getLinear().setBias(0, 0.0);
        layer.getLinear().setBias(1, 0.0);

        NeuralNetwork nn = new NeuralNetwork(new Layer[]{layer}, new CrossEntropy(), new SGD(0.1));
        double loss = nn.train(X, Y);

        assertTrue(Double.isFinite(loss));
        assertTrue(loss < 1e-10, "Loss should be near zero for confident correct prediction");

        Matrix dB = layer.getLinear().getDB();
        assertEquals(0.0, dB.get(0, 0), 1e-5);
        assertEquals(0.0, dB.get(0, 1), 1e-5);
    }

    @Test
    @DisplayName("Batch averaging scales gradient strictly by 1/N")
    public void testBatchAveraging() throws InvalidValue {
        int[] batchSizes = {1, 4, 16};
        double[][] expectedGrads = new double[batchSizes.length][2];

        for (int b = 0; b < batchSizes.length; b++) {
            int bs = batchSizes[b];
            Matrix X = new Matrix(bs, 2);
            Matrix Y = new Matrix(bs, 2);
            for (int i = 0; i < bs; i++) {
                X.set(i, 0, 0.5); X.set(i, 1, -0.5);
                Y.set(i, 0, 1.0); Y.set(i, 1, 0.0);
            }

            Layer layer = new Layer(2, 2, new Softmax());
            layer.getLinear().setWeight(0, 0, 0.2); layer.getLinear().setWeight(0, 1, -0.1);
            layer.getLinear().setWeight(1, 0, -0.2); layer.getLinear().setWeight(1, 1, 0.1);
            layer.getLinear().setBias(0, 0.0);
            layer.getLinear().setBias(1, 0.0);

            NeuralNetwork nn = new NeuralNetwork(new Layer[]{layer}, new CrossEntropy(), new SGD(0.1));
            nn.train(X, Y);

            expectedGrads[b][0] = layer.getLinear().getDB().get(0, 0);
            expectedGrads[b][1] = layer.getLinear().getDB().get(0, 1);

            assertEquals(expectedGrads[0][0], expectedGrads[b][0], 1e-12, "Mean batch gradient must be invariant to batch replication");
            assertEquals(expectedGrads[0][1], expectedGrads[b][1], 1e-12, "Mean batch gradient must be invariant to batch replication");
        }
    }
}
