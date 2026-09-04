package examples;

import core.Matrix;
import exceptions.InvalidValue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class MNISTTest {

    @Test
    @DisplayName("Synthetic IDX image and label files parse with exact dimensions, normalization, and one-hot encoding")
    public void testSyntheticIdxParsing() throws Exception {
        Path tempDir = Files.createTempDirectory("mnist_test");
        try {
            File imgFile = tempDir.resolve("train-images-idx3-ubyte").toFile();
            File lblFile = tempDir.resolve("train-labels-idx1-ubyte").toFile();
            File testImgFile = tempDir.resolve("t10k-images-idx3-ubyte").toFile();
            File testLblFile = tempDir.resolve("t10k-labels-idx1-ubyte").toFile();

            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(imgFile)))) {
                out.writeInt(2051); 
                out.writeInt(2);    
                out.writeInt(28);   
                out.writeInt(28);   
                
                for (int i = 0; i < 784; i++) {
                    out.writeByte(255);
                }
                
                for (int i = 0; i < 784; i++) {
                    out.writeByte(128);
                }
            }

            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(lblFile)))) {
                out.writeInt(2049); 
                out.writeInt(2);    
                out.writeByte(3);
                out.writeByte(7);
            }

            Files.copy(imgFile.toPath(), testImgFile.toPath());
            Files.copy(lblFile.toPath(), testLblFile.toPath());

            MNIST mnist = new MNIST(tempDir.toString());

            Matrix xTrain = mnist.getXTrain();
            Matrix yTrain = mnist.getYTrain();

            assertArrayEquals(new int[]{2, 784}, xTrain.shape(), "Image shape should be 2x784");
            assertArrayEquals(new int[]{2, 10}, yTrain.shape(), "Label shape should be 2x10");

            assertEquals(1.0, xTrain.get(0, 0), 1e-6, "Pixel 255 normalized should be 1.0");
            assertEquals(128.0 / 255.0, xTrain.get(1, 0), 1e-6, "Pixel 128 normalized should be 128/255");

            assertEquals(1.0, yTrain.get(0, 3), 1e-6, "Label 3 should have 1.0 at index 3");
            assertEquals(0.0, yTrain.get(0, 0), 1e-6, "Other indices should be 0.0");
            assertEquals(1.0, yTrain.get(1, 7), 1e-6, "Label 7 should have 1.0 at index 7");

            double sumRow0 = 0, sumRow1 = 0;
            for (int c = 0; c < 10; c++) {
                sumRow0 += yTrain.get(0, c);
                sumRow1 += yTrain.get(1, c);
            }
            assertEquals(1.0, sumRow0, 1e-6, "One-hot vector must sum to 1.0");
            assertEquals(1.0, sumRow1, 1e-6, "One-hot vector must sum to 1.0");

        } finally {
            
            for (File f : tempDir.toFile().listFiles()) {
                f.delete();
            }
            tempDir.toFile().delete();
        }
    }
}
