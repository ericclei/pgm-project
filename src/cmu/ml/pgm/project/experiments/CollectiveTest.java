package cmu.ml.pgm.project.experiments;

import cmu.ml.pgm.project.*;
import no.uib.cipr.matrix.Matrix;

public class CollectiveTest {

	public static void main(String[] args) {
		CollectiveMatrixFactorizationDataset mfTrain = new MatrixFactorizationMovieLens(
				"Data/ml-100k/u.user", "Data/ml-100k/u.item",
				"Data/ml-100k/u.data.train", "Data/ml-100k/u.info.train", 0);
		// mfTrain.printMatrix();
		// int latentDim = 3;
		int[] ls = { 2, 4, 6, 8 };
		double step = 1e-5;
		int maxIterOuter = 50;
		int maxIterInner = 20;
		CollectiveMatrixFactorizationDataset mfTest = new MatrixFactorizationMovieLens(
				"Data/ml-100k/u.user", "Data/ml-100k/u.item",
				"Data/ml-100k/u.data.test", "Data/ml-100k/u.info.test", 0);
		Matrix testR = mfTest.getRelations(0, 1);

		for (int l : ls) {
			System.out.println("l=" + l);
			System.out.println("starting with features");
			CollectiveMatrixFactorizationResult featuresResult = CollectiveMatrixFactorization
					.factorizeMatricesWithFeatures(mfTrain, l, maxIterOuter,
							maxIterInner, step, step);
			System.out.println("done with features");

			Matrix rFeatures = featuresResult.getRelations(0, 1);
			double featuresError = 0;
			int nTest = 0;
			for (int i = 0; i < mfTest.getNumItems(0); i++)
				for (int j = 0; j < mfTest.getNumItems(1); j++)
					if (testR.get(i, j) != 0) {
						nTest++;
						featuresError += Math.pow(
								testR.get(i, j) - rFeatures.get(i, j), 2);
					}
			featuresError = Math.sqrt(featuresError / nTest);
			System.out.printf("RMSE with features = %f\n", featuresError);
		}

		// for (int k = 0; k < featuresResult.getNumIntermediate(); k++) {
		// Matrix rFeatures = featuresResult.getIntermediateRelations(k, 0, 1);
		// double featuresError = 0;
		// int nTest = 0;
		// for (int i = 0; i < mfTest.getNumItems(0); i++)
		// for (int j = 0; j < mfTest.getNumItems(1); j++)
		// if (testR.get(i, j) != 0) {
		// nTest++;
		// featuresError += Math.pow(
		// testR.get(i, j) - rFeatures.get(i, j), 2);
		// }
		// featuresError = Math.sqrt(featuresError / nTest);
		// System.out.printf("RMSE with features = %f\n", featuresError);
		// }

	}
}