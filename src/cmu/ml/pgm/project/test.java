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
		mfTrain.printMatrix();
		int latentDim = 1;
		double stepSize = 1e-7;
		int maxIter = 10;
		double eps = 1e-5;
		MatrixFactorizationMovieLens mfTest
		= new MatrixFactorizationMovieLens("Data/ml-100k/u.user", "Data/ml-100k/u.item",
				"Data/ml-100k/u.data.test", "Data/ml-100k/u.info.test");
		Matrix testR = mfTest.getRelationMatrix();
		int nTest;
		PrintWriter writer;

//		List<Matrix> featuresResult = MatrixFactorization.featureEnrichedMatrixFactorization(
//				mfTrain, latentDim, stepSize, maxIter, eps);
//		Matrix rFeatures = featuresResult.get(0);
//		System.out.println("done with features");
//
//
//		double featuresError = 0;
//		nTest = 0;
//		for (int i = 0; i < mfTest.getNumUsers(); i++)
//			for (int j = 0; j < mfTest.getNumItems(); j++) {
//				if (testR.get(i, j) != 0)
//					nTest++;
//				featuresError += Math.pow(testR.get(i, j) - rFeatures.get(i, j), 2);
//			}
//		featuresError = Math.sqrt(featuresError / nTest);
//		System.out.printf("RMSE with features = %f\n", featuresError);
//
//		System.out.println("ten entries of R with features:");
//		for (int i = 0; i < 10; i++)
//			System.out.print(rFeatures.get(0, i) + " ");
//		System.out.println();
//
//		try {
//			writer = new PrintWriter("output/rFeatures.txt", "UTF-8");
//			writer.println(rFeatures);
//			writer.close();
//		} catch (FileNotFoundException e) {
//			e.printStackTrace();
//		} catch (UnsupportedEncodingException e) {
//			e.printStackTrace();
//		}

		List<Matrix> noFeaturesResult = MatrixFactorization.matrixFactorization(
				mfTrain, latentDim, stepSize, maxIter, eps);
		Matrix rNoFeatures = noFeaturesResult.get(0);
		System.out.println("done with no features");
		double noFeaturesError = 0;
		nTest = 0;
		for (int i = 0; i < mfTest.getNumUsers(); i++)
			for (int j = 0; j < mfTest.getNumItems(); j++) {
				if (testR.get(i, j) != 0)
					nTest++;
				noFeaturesError += Math.pow(testR.get(i, j) - rNoFeatures.get(i, j), 2);
			}
		noFeaturesError = Math.sqrt(noFeaturesError / nTest);
		System.out.printf("RMSE without features = %f\n", noFeaturesError);
		System.out.println("ten entries of R without features:");
		for (int i = 0; i < 10; i++)
			System.out.print(rNoFeatures.get(0, i) + " ");
		System.out.println();
		try {
			writer = new PrintWriter("output/rNoFeatures.txt", "UTF-8");
			writer.println(rNoFeatures);
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}
}
