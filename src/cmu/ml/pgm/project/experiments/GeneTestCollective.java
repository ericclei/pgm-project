package cmu.ml.pgm.project.experiments;

import cmu.ml.pgm.project.*;
import no.uib.cipr.matrix.Matrix;

public class GeneTestCollective {
	public static void main(String[] args) {
		boolean useFeatures = false;
		CollectiveMatrixFactorizationDataset train = new DataGeneDisease(
				"Data/IMC/", true, useFeatures);
		CollectiveMatrixFactorizationDataset test = new DataGeneDisease(
				"Data/IMC/", false, useFeatures);
		System.out.println("useFeatures=" + useFeatures);
		int latentDim = 5;
		double stepTransform = 1e-6;
		double stepLatentFromFeatures = 1e-6;
		double stepLatentFromRelations = 1e-5;
		int maxIterOuter = 8;
		int maxIterInner = 5;

		System.out.println("starting");
		CollectiveMatrixFactorizationResult featuresResult = CollectiveMatrixFactorization
				.factorizeMatricesWithFeatures(train, latentDim, maxIterOuter,
						maxIterInner, stepTransform, stepLatentFromFeatures,
						stepLatentFromRelations, false, true);
		System.out.println("done");

		int s = 0, t = 1;
		Matrix rFeatures = featuresResult.getRelations(s, t);
		Matrix testR = test.getRelations(s, t);
		double featuresError = 0;
		int nTest = 0;
		for (int i = 0; i < test.getNumItems(s); i++)
			for (int j = 0; j < test.getNumItems(t); j++)
				if (testR.get(i, j) != 0) {
					nTest++;
					featuresError += Math.pow(
							testR.get(i, j) - rFeatures.get(i, j), 2);
				}
		featuresError = Math.sqrt(featuresError / nTest);
		System.out.printf("RMSE of R" + s + t + " with features = %f\n",
				featuresError);

	}
}
