package cmu.ml.pgm.project;

import no.uib.cipr.matrix.Matrix;

public class CollectiveTest {

	public static void main(String[] args) {
		CollectiveMatrixFactorizationDataset mfTrain = new MatrixFactorizationMovieLens(
				"Data/ml-1m/users.dat", "Data/ml-1m/movies.dat",
				"Data/ml-1m/ratings.train.dat", "Data/ml-1m/info.train", 1);
		// mfTrain.printMatrix();
		int latentDim = 3;
		double step = 1e-6;
		int maxIterOuter = 10;
		int maxIterInner = 20;
		CollectiveMatrixFactorizationDataset mfTest = new MatrixFactorizationMovieLens(
				"Data/ml-1m/users.dat", "Data/ml-1m/movies.dat",
				"Data/ml-1m/ratings.test.dat", "Data/ml-1m/info.test", 1);
		Matrix testR = mfTest.getRelations(0, 0);

		System.out.println("starting with features");
		CollectiveMatrixFactorizationResult featuresResult = CollectiveMatrixFactorization
				.factorizeMatricesWithFeatures(mfTrain, latentDim,
						maxIterOuter, maxIterInner, step);
		System.out.println("done with features");
		for (int k = 0; k < featuresResult.getNumIntermediate(); k++) {
			Matrix rFeatures = featuresResult.getIntermediateRelations(k, 0, 1);
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

	}
}