package gradient;

import core.Matrix;
import exceptions.InvalidValue;
import layers.Layer;
import losses.MSE;
import optimizers.SGD;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class BatchGradientValidationTest {

    @Test
    @DisplayName("Batch gradients follow MEAN convention and remain invariant under repeated sample replication")
    public void testBatchGradientInvariance() throws InvalidValue {
        int[] batchSizes = {1, 2, 4, 8, 16, 32, 64};
        double[] sampleX = {0.5, -0.3, 0.8};
        double[] sampleY = {1.0, 0.0};

        double baseLoss = 0.0;
        Matrix baseDW = null;
        Matrix baseDB = null;
        Matrix baseUpdatedW = null;

        for (int i = 0; i < batchSizes.length; i++) {
            int bs = batchSizes[i];
            Matrix X = new Matrix(bs, 3);
            Matrix Y = new Matrix(bs, 2);

            for (int r = 0; r < bs; r++) {
                for (int c = 0; c < 3; c++) X.set(r, c, sampleX[c]);
                for (int c = 0; c < 2; c++) Y.set(r, c, sampleY[c]);
            }

            Layer layer = new Layer(3, 2);
            layer.getLinear().setWeight(0, 0, 0.2); layer.getLinear().setWeight(0, 1, -0.1); layer.getLinear().setWeight(0, 2, 0.4);
            layer.getLinear().setWeight(1, 0, -0.2); layer.getLinear().setWeight(1, 1, 0.3); layer.getLinear().setWeight(1, 2, -0.5);
            layer.getLinear().setBias(0, 0.1);
            layer.getLinear().setBias(1, -0.1);

            MSE mse = new MSE();
            Matrix pred = layer.forward(X);
            double loss = mse.forward(Y, pred);
            Matrix dLoss = mse.derivative(Y, pred);
            layer.backward(dLoss);

            Matrix dW = layer.getLinear().getDW();
            Matrix dB = layer.getLinear().getDB();

            SGD sgd = new SGD(0.05);
            Matrix updatedW = sgd.update(layer.getLinear().getW(), dW);

            if (bs == 1) {
                baseLoss = loss;
                baseDW = dW;
                baseDB = dB;
                baseUpdatedW = updatedW;
            } else {
                assertEquals(baseLoss, loss, 1e-12, "Loss must be mean-normalized across batch size " + bs);
                for (int r = 0; r < 2; r++) {
                    for (int c = 0; c < 3; c++) {
                        assertEquals(baseDW.get(r, c), dW.get(r, c), 1e-12,
                            String.format("dW[%d,%d] must be invariant to batch replication at bs=%d", r, c, bs));
                        assertEquals(baseUpdatedW.get(r, c), updatedW.get(r, c), 1e-12,
                            String.format("Updated W[%d,%d] must be invariant to batch replication at bs=%d", r, c, bs));
                    }
                    assertEquals(baseDB.get(0, r), dB.get(0, r), 1e-12,
                        String.format("dB[%d] must be invariant to batch replication at bs=%d", r, bs));
                }
            }
        }
    }
}
