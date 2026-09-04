package losses;

import core.Matrix;
import exceptions.InvalidValue;

public class SoftmaxCrossEntropy implements Loss {
    private static final double EPS = 1e-15;

    @Override
    public double forward(Matrix y, Matrix logits) {
        int N = y.shape()[0];
        int O = y.shape()[1];
        double totalLoss = 0.0;

        for (int i = 0; i < N; i++) {
            double maxLogit = logits.get(i, 0);
            for (int j = 1; j < O; j++) {
                maxLogit = Math.max(maxLogit, logits.get(i, j));
            }

            double sumExp = 0.0;
            for (int j = 0; j < O; j++) {
                sumExp += Math.exp(logits.get(i, j) - maxLogit);
            }
            double logSumExp = Math.log(Math.max(sumExp, EPS)) + maxLogit;

            for (int j = 0; j < O; j++) {
                totalLoss += y.get(i, j) * (logSumExp - logits.get(i, j));
            }
        }
        return totalLoss / N;
    }

    @Override
    public Matrix derivative(Matrix y, Matrix logits) throws InvalidValue {
        int N = logits.shape()[0];
        int O = logits.shape()[1];
        Matrix result = new Matrix(N, O);

        for (int i = 0; i < N; i++) {
            double maxLogit = logits.get(i, 0);
            for (int j = 1; j < O; j++) {
                maxLogit = Math.max(maxLogit, logits.get(i, j));
            }

            double sumExp = 0.0;
            for (int j = 0; j < O; j++) {
                double exp = Math.exp(logits.get(i, j) - maxLogit);
                result.set(i, j, exp);
                sumExp += exp;
            }

            for (int j = 0; j < O; j++) {
                double p = result.get(i, j) / sumExp;
                result.set(i, j, (p - y.get(i, j)) / N);
            }
        }
        return result;
    }
}
