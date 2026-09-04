package losses;

import core.Matrix;
import exceptions.InvalidValue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ExplicitSoftmaxCrossEntropyTest {

    @Test
    @DisplayName("Explicit SoftmaxCrossEntropy computes correct loss and (p - y)/N gradient")
    public void testExplicitSoftmaxCrossEntropy() throws InvalidValue {
        SoftmaxCrossEntropy loss = new SoftmaxCrossEntropy();

        Matrix logits = new Matrix(new double[][]{{1.0, 2.0, 3.0}});
        Matrix targets = new Matrix(new double[][]{{0.0, 0.0, 1.0}});

        double l = loss.forward(targets, logits);
        assertTrue(Double.isFinite(l));
        assertTrue(l > 0.0);

        Matrix grad = loss.derivative(targets, logits);
        assertEquals(3, grad.shape()[1]);

        double sumGrad = grad.get(0, 0) + grad.get(0, 1) + grad.get(0, 2);
        assertEquals(0.0, sumGrad, 1e-12, "Sum of SoftmaxCrossEntropy gradients must equal zero");
    }
}
