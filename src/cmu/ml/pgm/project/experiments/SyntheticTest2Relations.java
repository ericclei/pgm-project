package cmu.ml.pgm.project.experiments;

import cmu.ml.pgm.project.*;
import no.uib.cipr.matrix.Matrix;

public class SyntheticTest2Relations {

	public static void main(String[] args) {
		CollectiveMatrixFactorizationDataset mfTrain = new SyntheticDataset2Relations(
				"Data/synthetic/F.normal.1.dat", "Data/synthetic/F.normal.2.dat",
				"Data/synthetic/F.normal.3.dat", "Data/synthetic/R.1.2.train.dat",
				"Data/synthetic/R.1.3.train.dat", true);
		int latentDim = 10;
		double step = 1e-4;
		int maxIterOuter = 20;
		int maxIterInner = 10;
		CollectiveMatrixFactorizationDataset mfTest = new SyntheticDataset2Relations(
				"Data/synthetic/F.normal.1.dat", "Data/synthetic/F.normal.2.dat",
				"Data/synthetic/F.normal.3.dat", "Data/synthetic/R.1.2.test.dat",
				"Data/synthetic/R.1.3.test.dat", false);

		boolean isFeatures = false;
		boolean isBayesian = true;

		if(isFeatures) {
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
						+ " with features = %f\n", featuresError);

			}
			System.out.println();

			for (int k = 0; k < featuresResult.getNumIntermediate(); k++) {
				int s = 0, t = 2;
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
						+ " with features = %f\n", featuresError);

			}
		}
		if(isBayesian) {
			CollectiveMatrixFactorizationResult featuresResult = BayesianMatrixFactorization
					.factorizeMatrixWithFeatures(mfTrain, latentDim, 100, 1);
			System.out.println("done");

			int s = 0;
			for(int t = s + 1; t < 3; t++) {
				Matrix averageRelation = new no.uib.cipr.matrix.DenseMatrix
						(featuresResult.getIntermediateRelations(0, s, t).numRows(),
								featuresResult.getIntermediateRelations(0, s, t).numColumns());
				for (int k = 0; k < featuresResult.getNumIntermediate(); k++) {
					averageRelation.add(featuresResult.getIntermediateRelations(k, s, t));
//						Matrix rFeatures = featuresResult.getIntermediateRelations(k, s, t);
					Matrix rFeatures = averageRelation.copy().scale(1.0 / (k + 1));
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
							+ " with features = %f\n", featuresError);

				}
				System.out.println();
			}
		}
	}

}
