package cmu.ml.pgm.project;

import static cmu.ml.pgm.project.MatrixMethods.*;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;

public final class CollectiveMatrixFactorization {

	private CollectiveMatrixFactorization() {
	}

	public static CollectiveMatrixFactorizationResult factorizeMatricesWithFeatures(
			CollectiveMatrixFactorizationDataset data, int latentDim,
			int maxIterOuter, int maxIterInner, double stepFeatureTransforms,
			double stepLatentFeatures) {
		return factorizeMatricesWithFeatures(data, latentDim, maxIterOuter,
				maxIterInner, stepFeatureTransforms, stepLatentFeatures, true);
	}

	public static CollectiveMatrixFactorizationResult factorizeMatricesWithFeatures(
			CollectiveMatrixFactorizationDataset data, int latentDim,
			int maxIterOuter, int maxIterInner, double stepFeatureTransforms,
			double stepLatentFeatures, boolean saveIntermediate) {

		int nEntities = data.getNumEntities();
		CollectiveMatrixFactorizationResult result = new CollectiveMatrixFactorizationResult(
				nEntities);

		// initialize estimates
		for (int s = 0; s < nEntities; s++) {
			int n = data.getNumItems(s);
			int dNorm = data.getNumNormalFeatures(s);
			result.setLatentFeatures(s, new DenseMatrix(n, latentDim));
			randomlyInitialize(result.getLatentFeatures(s));
			result.setBernoulliFeatureMap(s,
					new DenseMatrix(latentDim, data.getNumBernoulliFeatures(s)));
			randomlyInitialize(result.getBernoulliFeatureMap(s));
			result.setNormalFeatureMap(s,
					new DenseMatrix(latentDim, data.getNumNormalFeatures(s)));
			randomlyInitialize(result.getNormalFeatureMap(s));

			Matrix fNorm = data.getNormalFeatures(s);
			Matrix u = result.getLatentFeatures(s);
			Matrix aNorm = result.getNormalFeatureMap(s);
			result.setFeatureVariance(s,
					squaredFrobeniusNorm(minus(fNorm, times(u, aNorm)))
							/ (n * dNorm));
		}
		for (int s = 0; s < nEntities; s++) {
			for (int t = s; t < nEntities; t++) {
				Matrix r = data.getRelations(s, t);
				if (r == null)
					continue;
				Matrix rHat = result.getRelations(s, t);
				result.setRelationVariance(
						s,
						t,
						sparseSquaredFrobeniusNormOfDiff(r, rHat)
								/ data.getNumObserved(s, t));
			}
		}

		// coordinate descent
		for (int t = 0; t < maxIterOuter; t++) {
			System.out.println("t = " + t);
			// update feature maps for each entity
			for (int s = 0; s < nEntities; s++) {
				System.out.println("updating feature maps for " + s);
				Matrix u = result.getLatentFeatures(s);
				Matrix fBern = data.getBernoulliFeatures(s);
				Matrix fNorm = data.getNormalFeatures(s);
				int dBern = data.getNumBernoulliFeatures(s);
				int dNorm = data.getNumNormalFeatures(s);
				int n = data.getNumItems(s);
				double sigma2 = result.getFeatureVariance(s);

				// update Bernoulli maps
				for (int tt = 0; tt < maxIterInner; tt++) {
					System.out.println("\ttt = " + tt);
					// update each column
					for (int k = 0; k < dBern; k++) {
						Vector grad = new DenseVector(latentDim);
						Matrix a = result.getBernoulliFeatureMap(s);
						Vector a_k = getColumn(a, k);
						for (int i = 0; i < n; i++) {
							Vector u_i = getRow(u, i);
							Vector newVal = times(u_i, fBern.get(i, k));
							newVal.add(
									-1,
									times(u_i,
											1 - 1 / (1 + Math.exp(u_i.dot(a_k)))));
							grad.add(newVal);
						}
						Vector scaledGrad = times(grad, stepFeatureTransforms);
						addToColumn(a, k, scaledGrad);
						result.setBernoulliFeatureMap(s, a);
					}
				}

				// update normal maps
				for (int tt = 0; tt < maxIterInner; tt++) {
					Matrix an = result.getNormalFeatureMap(s);
					Matrix grad = times(transpose(u),
							minus(fNorm, times(u, an)));
					grad.scale(1 / sigma2);
					Matrix scaledGrad = times(grad, stepFeatureTransforms);
					an.add(scaledGrad);
					result.setNormalFeatureMap(s, an);
				}

				// update variance
				Matrix an = result.getNormalFeatureMap(s);
				Matrix mu = times(u, an);
				sigma2 = squaredFrobeniusNorm(minus(fNorm, mu)) / (n * dNorm);
				result.setFeatureVariance(s, sigma2);
			}
			// System.out.println("finished feature maps");

			// update latent features for each entity
			for (int s = 0; s < nEntities; s++) {
				System.out.println("updating latent features for " + s);
				Matrix fBern = data.getBernoulliFeatures(s);
				Matrix fNorm = data.getNormalFeatures(s);
				Matrix aBern = result.getBernoulliFeatureMap(s);
				Matrix aNorm = result.getNormalFeatureMap(s);
				int dBern = data.getNumBernoulliFeatures(s);
				int n = data.getNumItems(s);
				double sigma2_s = result.getFeatureVariance(s);

				for (int tt = 0; tt < maxIterInner; tt++) {
					System.out.println("\ttt = " + tt);
					Matrix u = result.getLatentFeatures(s);

					// gradient from Bernoulli feature term
					if (data.getNumBernoulliFeatures(s) != 0) {
						for (int i = 0; i < n; i++) {
							// update row i of U
							Vector grad = new DenseVector(latentDim);
							for (int k = 0; k < dBern; k++) {
								Vector u_i = getRow(u, i);
								Vector a_k = getColumn(aBern, k);
								Vector newVal = times(a_k, fBern.get(i, k));
								newVal.add(
										-1,
										times(a_k, 1 - 1 / (1 + Math.exp(u_i
												.dot(a_k)))));
								grad.add(newVal);
							}
							Vector scaledGrad = times(grad, stepLatentFeatures);
							addToRow(result.getLatentFeatures(s), i, scaledGrad);
						}
						// System.out.println("\t1");
					}

					System.out.println("getting gradient from normal features");
					// gradient from normal feature term
					if (data.getNumNormalFeatures(s) != 0) {
						Matrix gradNorm = times(minus(fNorm, times(u, aNorm)),
								transpose(aNorm)).scale(1 / sigma2_s);
						Matrix scaledGradNorm = times(gradNorm,
								stepLatentFeatures);
						result.getLatentFeatures(s).add(scaledGradNorm);
						// System.out.println("\t2");
					}

					System.out.println("getting gradient from relations");
					// gradient from relation term with each entity
					for (int w = 0; w < nEntities; w++) {
						// if (w == s)
						// continue;
						Matrix r = data.getRelations(s, w);
						if (r == null)
							continue;
						Matrix v = result.getLatentFeatures(w);
						Matrix rHat = times(u, transpose(v));
						int n_w = data.getNumItems(w);
						double sigma2_sw = result.getRelationVariance(s, w);
						for (int i = 0; i < n; i++) {
							// update row i of U
							Vector grad = new DenseVector(latentDim);
							for (int j = 0; j < n_w; j++) {
								double r_ij = r.get(i, j);
								double rHat_ij = rHat.get(i, j);
								double selfRelationCorrection = (w == s && i == j) ? 2
										: 1;
								if (r_ij != 0)
									grad.add(times(getRow(v, j),
											selfRelationCorrection
													* (r_ij - rHat_ij)));
							}
							grad.scale(1 / sigma2_sw);
							Vector scaledGrad = times(grad, stepLatentFeatures);
							addToRow(result.getLatentFeatures(s), i, scaledGrad);
						}
					}
					// System.out.println("\t3");
				}
			}
			// System.out.println("finished latent features");

			// update variances
			for (int s = 0; s < nEntities; s++) {
				Matrix aNorm = result.getNormalFeatureMap(s);
				Matrix fNorm = data.getNormalFeatures(s);
				Matrix u = result.getLatentFeatures(s);
				int n = data.getNumItems(s);
				int dNorm = data.getNumNormalFeatures(s);
				double sigma2 = squaredFrobeniusNorm(minus(fNorm,
						times(u, aNorm)))
						/ (n * dNorm);
				result.setFeatureVariance(s, sigma2);

				for (int w = s + 1; w < nEntities; w++) {
					Matrix r = data.getRelations(s, w);
					if (r == null)
						continue;
					Matrix rHat = result.getRelations(s, w);
					result.setRelationVariance(
							s,
							w,
							sparseSquaredFrobeniusNormOfDiff(r, rHat)
									/ data.getNumObserved(s, w));
				}
			}

			if (saveIntermediate)
				result.addIntermediateRelations();
		}

		return result;
	}
}
