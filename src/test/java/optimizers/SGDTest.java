package optimizers;

import core.Matrix;
import exceptions.InvalidValue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SGDTest {

    @Test
    @DisplayName("SGD updates parameter according to theta = theta - lr * grad")
    public void testSGDUpdate() throws InvalidValue {
        SGD sgd = new SGD(0.1);
        Matrix param = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        Matrix grad = new Matrix(new double[][]{{0.5, -0.5}, {1.0, 2.0}});

        Matrix updated = sgd.update(param, grad);

        assertEquals(0.95, updated.get(0, 0), 1e-12);
        assertEquals(2.05, updated.get(0, 1), 1e-12);
        assertEquals(2.90, updated.get(1, 0), 1e-12);
        assertEquals(3.80, updated.get(1, 1), 1e-12);
    }
}
