package gradient;

import core.Matrix;
import exceptions.InvalidValue;
import layers.Layer;
import losses.MSE;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class EpsilonSweepTest {

    @Test
    @DisplayName("Epsilon sweep across 1e-2 to 1e-8 demonstrates finite difference behavior")
    public void testEpsilonSweep() throws InvalidValue {
        double[] epsilons = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};
        Random rng = new Random(42);

        Matrix X = new Matrix(new double[][]{{0.5, -0.2, 0.8}});
        Matrix Y = new Matrix(new double[][]{{1.0, 0.0}});

        Layer layer = new Layer(3, 2);

        System.out.println("\n--- EPSILON SWEEP RESULTS ---");
        System.out.printf("%-10s | %-15s | %-15s | %-15s\n", "Epsilon", "Max Abs Err", "Mean Abs Err", "Max Rel Err");
        System.out.println("---------------------------------------------------------------");

        for (double eps : epsilons) {
            GradientCheckResult res = GradientChecker.checkLayerWeights(layer, new MSE(), X, Y, eps);
            System.out.printf("%-10.1e | %-15.6e | %-15.6e | %-15.6e\n",
                eps, res.maxAbsError, res.meanAbsError, res.maxRelError);

            if (eps >= 1e-6 && eps <= 1e-3) {
                assertTrue(res.maxRelError < 1e-4, "Relative error at eps=" + eps + " must be < 1e-4");
            }
        }
    }
}
