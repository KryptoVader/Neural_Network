package core;

import activations.ReLU;
import activations.Softmax;
import exceptions.InvalidValue;
import layers.Layer;
import losses.CrossEntropy;
import optimizers.SGD;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class ReproducibilityTest {

    @Test
    @DisplayName("Two neural networks initialized with the same seed produce identical weights, losses, and predictions")
    public void testDeterministicTraining() throws InvalidValue {
        long seed = 42L;

        Layer l1_a = new Layer(4, 3, new ReLU(), new Random(seed));
        Layer l2_a = new Layer(3, 2, new Softmax(), new Random(seed + 1));
        NeuralNetwork nn1 = new NeuralNetwork(new Layer[]{l1_a, l2_a}, new CrossEntropy(), new SGD(0.05));
        nn1.setSeed(seed + 2);

        Layer l1_b = new Layer(4, 3, new ReLU(), new Random(seed));
        Layer l2_b = new Layer(3, 2, new Softmax(), new Random(seed + 1));
        NeuralNetwork nn2 = new NeuralNetwork(new Layer[]{l1_b, l2_b}, new CrossEntropy(), new SGD(0.05));
        nn2.setSeed(seed + 2);

        Random dataRng = new Random(999);
        Matrix X = new Matrix(10, 4);
        Matrix Y = new Matrix(10, 2);
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 4; j++) X.set(i, j, dataRng.nextDouble());
            int label = dataRng.nextInt(2);
            Y.set(i, label, 1.0);
        }

        for (int epoch = 0; epoch < 3; epoch++) {
            double loss1 = nn1.train(X, Y, 2);
            double loss2 = nn2.train(X, Y, 2);
            assertEquals(loss1, loss2, 1e-12, "Loss must match identically across deterministic runs");
        }

        Matrix pred1 = nn1.forward(X);
        Matrix pred2 = nn2.forward(X);
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(pred1.get(i, j), pred2.get(i, j), 1e-12, "Predictions must match identically");
            }
        }
    }
}
