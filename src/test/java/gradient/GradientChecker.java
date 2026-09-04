package gradient;

import core.Matrix;
import core.NeuralNetwork;
import exceptions.InvalidValue;
import layers.Layer;
import layers.Linear;
import losses.Loss;

public class GradientChecker {

    public static GradientCheckResult checkLayerWeights(Layer layer, Loss loss, Matrix X, Matrix Y, double eps) throws InvalidValue {
        Linear linear = layer.getLinear();
        int outFeatures = linear.getW().shape()[0];
        int inFeatures = linear.getW().shape()[1];

        Matrix pred = layer.forward(X);
        Matrix dLoss = loss.derivative(Y, pred);
        layer.backward(dLoss);
        Matrix dW_analytical = linear.getDW();

        double maxAbs = 0.0, sumAbs = 0.0, maxRel = 0.0;
        String worst = "";
        int count = 0;

        for (int r = 0; r < outFeatures; r++) {
            for (int c = 0; c < inFeatures; c++) {
                double orig = linear.getW().get(r, c);

                linear.setWeight(r, c, orig + eps);
                double lPlus = loss.forward(Y, layer.forward(X));

                linear.setWeight(r, c, orig - eps);
                double lMinus = loss.forward(Y, layer.forward(X));

                linear.setWeight(r, c, orig);

                double numGrad = (lPlus - lMinus) / (2.0 * eps);
                double anaGrad = dW_analytical.get(r, c);

                double absErr = Math.abs(numGrad - anaGrad);
                double relErr = absErr / (Math.max(1e-12, Math.abs(anaGrad) + Math.abs(numGrad)));

                if (absErr > maxAbs) {
                    maxAbs = absErr;
                    worst = String.format("W[%d,%d] (ana=%.4e, num=%.4e)", r, c, anaGrad, numGrad);
                }
                if (relErr > maxRel) {
                    maxRel = relErr;
                }
                sumAbs += absErr;
                count++;
            }
        }
        return new GradientCheckResult(maxAbs, sumAbs / count, maxRel, worst);
    }

    public static GradientCheckResult checkLayerBiases(Layer layer, Loss loss, Matrix X, Matrix Y, double eps) throws InvalidValue {
        Linear linear = layer.getLinear();
        int outFeatures = linear.getB().shape()[1];

        Matrix pred = layer.forward(X);
        Matrix dLoss = loss.derivative(Y, pred);
        layer.backward(dLoss);
        Matrix dB_analytical = linear.getDB();

        double maxAbs = 0.0, sumAbs = 0.0, maxRel = 0.0;
        String worst = "";
        int count = 0;

        for (int c = 0; c < outFeatures; c++) {
            double orig = linear.getB().get(0, c);

            linear.setBias(c, orig + eps);
            double lPlus = loss.forward(Y, layer.forward(X));

            linear.setBias(c, orig - eps);
            double lMinus = loss.forward(Y, layer.forward(X));

            linear.setBias(c, orig);

            double numGrad = (lPlus - lMinus) / (2.0 * eps);
            double anaGrad = dB_analytical.get(0, c);

            double absErr = Math.abs(numGrad - anaGrad);
            double relErr = absErr / (Math.max(1e-12, Math.abs(anaGrad) + Math.abs(numGrad)));

            if (absErr > maxAbs) {
                maxAbs = absErr;
                worst = String.format("b[%d] (ana=%.4e, num=%.4e)", c, anaGrad, numGrad);
            }
            if (relErr > maxRel) {
                maxRel = relErr;
            }
            sumAbs += absErr;
            count++;
        }
        return new GradientCheckResult(maxAbs, sumAbs / count, maxRel, worst);
    }
}
