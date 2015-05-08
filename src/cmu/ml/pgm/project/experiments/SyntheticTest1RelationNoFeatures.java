package cmu.ml.pgm.project.experiments;

import cmu.ml.pgm.project.*;
import no.uib.cipr.matrix.Matrix;

public class SyntheticTest1RelationNoFeatures {

	public static void main(String[] args) {
		CollectiveMatrixFactorizationDataset mfTrain = new SyntheticDataset1RelationNoFeatures(
				"Data/synthetic/R.1.2.train.dat", true);
		int latentDim = 10;
		double step = 1e-4;
		int maxIterOuter = 20;
		int maxIterInner = 10;
		CollectiveMatrixFactorizationDataset mfTest = new SyntheticDataset1RelationNoFeatures(
				"Data/synthetic/R.1.2.test.dat", false);

		System.out.println("starting");
		CollectiveMatrixFactorizationResult featuresResult = CollectiveMatrixFactorization
				.factorizeMatricesWithFeatures(mfTrain, latentDim,
						maxIterOuter, maxIterInner, step, step);
		System.out.println("done");
		for (int k = 0; k < featuresResult.getNumIntermediate(); k++) {
			int s = 0, t = 1;
			Matrix rFeatures = featuresResult.getIntermediateRelations(k, s, t);
			Matrix testR = mfTest.getRelations(s, t);
			double featuresError = 0;
			int nTest = 0;
			for (int i = 0; i < mfTest.getNumItems(s); i++)
				for (int j = 0; j < mfTest.getNumItems(t); j++)
					if (testR.get(i, j) != 0) {
						nTest++;
						featuresError += Math.pow(
								testR.get(i, j) - rFeatures.get(i, j), 2);
					}
			featuresError = Math.sqrt(featuresError / nTest);
			System.out.printf("RMSE of R" + (s + 1) + (t + 1)
					+ " without features = %f\n", featuresError);

		}
		System.out.println();
	}

}
