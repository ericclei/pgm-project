package cmu.ml.pgm.project;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.List;

import no.uib.cipr.matrix.Matrix;

/**
 * Created by dexter on 15. 3. 3..
 */
public class test {
	public static void main(String[] args) {
		MatrixFactorizationMovieLens mfTrain
		= new MatrixFactorizationMovieLens("Data/ml-100k/u.user", "Data/ml-100k/u.item",
				"Data/ml-100k/u.data.train", "Data/ml-100k/u.info.train");
		//		mfTrain.printMatrix();
		int latentDim = 5;
		double stepSize = .3;
		double regCoef = 0;
		int maxIter = 50;
		double eps = 1e-7;
		MatrixFactorizationMovieLens mfTest
		= new MatrixFactorizationMovieLens("Data/ml-100k/u.user", "Data/ml-100k/u.item",
				"Data/ml-100k/u.data.test", "Data/ml-100k/u.info.test");
		Matrix testR = mfTest.getRelationMatrix();
		int nTest;
		PrintWriter writer;

		boolean doBaselineMethod = false;
		boolean doFeaturesMethod = true;
		if (doBaselineMethod) {
			System.out.println("Starting Baseline MF");
			MatrixFactorizationResult factorizationResult = BaselineMatrixFactorization.factorizeMatrix(
					mfTrain, latentDim, regCoef, maxIter * mfTrain.getTrainingData().size(), tol);
			Matrix rBaseline = factorizationResult.getR();
			System.out.println("Done with Baseline MF");
			nTest = 0;
			double error = 0;
			for (int i = 0; i < mfTest.getNumUsers(); i++) {
				for (int j = 0; j < mfTest.getNumItems(); j++) {
					if (testR.get(i, j) != 0) {
						nTest++;
						error += Math.pow(testR.get(i, j) - rBaseline.get(i, j), 2);
					}
				}
			}
			error = Math.sqrt(error / nTest);
			System.out.printf("RMSE baseline = %f\n", error);

			try {
				writer = new PrintWriter("output/rNoFeatures.txt", "UTF-8");
				writer.println(rBaseline);
				writer.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (UnsupportedEncodingException e) {
				e.printStackTrace();
			}
		}

		if (doFeaturesMethod) {
			for (int d = 1; d <= 10; d++) {
				System.out.println("starting with features");
				MatrixFactorizationResult featuresResult = MatrixFactorization.factorizeMatrixWithFeatures(
						mfTrain, d, stepSize, maxIter, eps);
				Matrix rFeatures = featuresResult.getR();
				System.out.println("done with features");
				double featuresError = 0;
				nTest = 0;
				for (int i = 0; i < mfTest.getNumUsers(); i++)
					for (int j = 0; j < mfTest.getNumItems(); j++) {
						if (testR.get(i, j) != 0) {
							nTest++;
							featuresError += Math.pow(testR.get(i, j) - rFeatures.get(i, j), 2);
						}
					}
				featuresError = Math.sqrt(featuresError / nTest);
				System.out.printf("RMSE with features = %f\n", featuresError);

				System.out.println("ten entries of R with features:");
				for (int i = 0; i < 10; i++)
					System.out.print(rFeatures.get(0, i) + " ");
				System.out.println();

				try {
					writer = new PrintWriter("output/rFeatures.txt", "UTF-8");
					writer.println(rFeatures);
					writer.close();
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (UnsupportedEncodingException e) {
					e.printStackTrace();
				}
			}
		}

	}

}
