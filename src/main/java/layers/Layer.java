package layers;

import core.Matrix;
import activations.Activation;
import exceptions.InvalidValue;

import java.util.Random;

public class Layer {
    private Linear l;
    private Activation a;
    private Matrix z;

    public Layer(int input_features, int output_features, Activation a) throws InvalidValue {
        this.l = new Linear(input_features, output_features);
        this.a = a;
    }

    public Layer(int input_features, int output_features, Activation a, long seed) throws InvalidValue {
        this.l = new Linear(input_features, output_features, seed);
        this.a = a;
    }

    public Layer(int input_features, int output_features, Activation a, Random rng) throws InvalidValue {
        this.l = new Linear(input_features, output_features, rng);
        this.a = a;
    }

    public Layer(int input_features, int output_features) throws InvalidValue {
        this.l = new Linear(input_features, output_features);
        this.a = null;
    }

    public Matrix forward(Matrix X) throws InvalidValue {
        Matrix z = l.forward(X);
        this.z = z;
        if (this.a != null) {
            return new Matrix(a.forward(z));
        }
        return new Matrix(z);
    }

    public Matrix backward(Matrix dLA) throws InvalidValue {
        Matrix dz = dLA;
        if (this.a != null) {
            dz = this.a.backward(dLA, this.z);
        }
        return this.l.backward(dz);
    }

    public Linear getLinear() {
        return this.l;
    }

    public Activation getActivation() {
        return this.a;
    }
}
