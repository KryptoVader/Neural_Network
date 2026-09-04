package gradient;

public class GradientCheckResult {
    public final double maxAbsError;
    public final double meanAbsError;
    public final double maxRelError;
    public final String worstParam;

    public GradientCheckResult(double maxAbsError, double meanAbsError, double maxRelError, String worstParam) {
        this.maxAbsError = maxAbsError;
        this.meanAbsError = meanAbsError;
        this.maxRelError = maxRelError;
        this.worstParam = worstParam;
    }

    @Override
    public String toString() {
        return String.format("MaxAbs=%.4e, MeanAbs=%.4e, MaxRel=%.4e, Worst=%s",
            maxAbsError, meanAbsError, maxRelError, worstParam);
    }
}
