package cmu.ml.pgm.project;

import java.util.ArrayList;
import java.util.List;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

import static cmu.ml.pgm.project.MatrixMethods.*;

/**
 * Methods for feature-enriched matrix factorization. Static class.
 * 
 * @author eric
 *
 */
public final class MatrixFactorization {

	private MatrixFactorization() {
	}

	/**
	 * 
	 * @param data
	 * @param latentDim
	 * @param maxIter
	 * @param tol
	 * @return estimates (R, U, V)
	 */
	public static MatrixFactorizationResult factorizeMatrix(
			MatrixFactorizationMovieLens data, int latentDim, double stepSize,
			int maxIter, double tol) {
		throw new UnsupportedOperationException();
	}

	/**
	 * 
	 * @param data
	 * @param latentDim
	 * @param maxIter
	 * @param tol
	 * @param returnIntermediate
	 *            whether to attach Rhat after each full round of coordinate
	 *            descent
	 * @return estimates (R, U, V, A, B, sigma_R^2, sigma_F^2, sigma_G^2)
	 */
	public static MatrixFactorizationResult factorizeMatrixWithFeatures(
			MatrixFactorizationMovieLens data, int latentDim, double stepSize,
			int maxIter, double tol) {
		LinkedSparseMatrix R = data.getRelationMatrix();
		int nObservedRelations = l0Norm(R);
		// Matrix Fn = new DenseMatrix(data.getuFeatureMatrix());
		// Matrix Gn = new DenseMatrix(data.getiFeatureMatrix());
		Matrix Fn = new DenseMatrix(getColumn(data.getuFeatureMatrix(), 0));
		Matrix Gn = new DenseMatrix(getColumn(data.getiFeatureMatrix(), 0));
		Matrix Fb = new DenseMatrix(removeColumn(data.getuFeatureMatrix(), 0));
		Matrix Gb = new DenseMatrix(removeColumn(data.getiFeatureMatrix(), 0));
		int dFn = Fn.numColumns();
		int dGn = Gn.numColumns();
		int dFb = Fb.numColumns();
		int dGb = Gb.numColumns();
		int m = Fn.numRows();
		int n = Gn.numRows();
		Matrix U = new DenseMatrix(m, latentDim);
		Matrix V = new DenseMatrix(n, latentDim);
		Matrix An = new DenseMatrix(latentDim, dFn);
		Matrix Bn = new DenseMatrix(latentDim, dGn);
		Matrix Ab = new DenseMatrix(latentDim, dFb);
		Matrix Bb = new DenseMatrix(latentDim, dGb);
		randomlyInitialize(U);
		randomlyInitialize(V);
		randomlyInitialize(An);
		randomlyInitialize(Ab);
		randomlyInitialize(Bn);
		randomlyInitialize(Bb);
		double sigma2R = 0, sigma2F = 0, sigma2G = 0;
		List<Matrix> iR = new ArrayList<>();

		// coordinate descent
		for (int t = 0; t < maxIter; t++) {
			// System.out.printf("t = %d\n", t);
			double overallMaxUpdate = 0;
			Matrix Rhat = times(U, transpose(V));
			sigma2R = sparseSquaredFrobeniusNormOfDiff(R, Rhat)
					/ nObservedRelations;
			sigma2F = squaredFrobeniusNorm(minus(Fn, times(U, An))) / m / dFn;
			sigma2G = squaredFrobeniusNorm(minus(Gn, times(V, Bn))) / n / dGn;

			// updates An, parameter for Fn
			for (int tA = 0; tA < maxIter; tA++) {
				// System.out.printf("\ttA = %d\n", tA);
				Matrix grad = times(transpose(U), minus(Fn, times(U, An)));
				grad.scale(-1 / sigma2F);
				Matrix scaledGrad = times(grad, stepSize);
				An.add(scaledGrad);
				double maxUpdate = scaledGrad.norm(Matrix.Norm.Maxvalue);
				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}
			sigma2F = squaredFrobeniusNorm(minus(Fn, times(U, An))) / m / dFn;

			// updates Bn, parameter for Gn
			for (int tB = 0; tB < maxIter; tB++) {
				// System.out.printf("\ttB = %d\n", tB);
				Matrix grad = times(transpose(V), minus(Gn, times(V, Bn)));
				grad.scale(-1 / sigma2G);
				Matrix scaledGrad = times(grad, stepSize);
				Bn.add(scaledGrad);
				double maxUpdate = scaledGrad.norm(Matrix.Norm.Maxvalue);
				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}
			sigma2G = squaredFrobeniusNorm(minus(Gn, times(V, Bn))) / n / dGn;

			// updates Ab, parameter for Fb
			for (int tA = 0; tA < maxIter; tA++) {
				for (int k = 0; k < dFb; k++) {
					// updates column k of Ab
					Vector grad = new DenseVector(latentDim);
					for (int i = 0; i < m; i++) {
						Vector Ui = getRow(U, i);
						Vector Ak = getColumn(Ab, k);
						Vector newVal = Ui.scale(Fb.get(i, k));
						newVal.add(-1,
								Ui.scale(1 - 1 / (1 + Math.exp(Ui.dot(Ak)))));
						grad.add(newVal);
					}
					Vector scaledGrad = times(grad, stepSize);
					addToColumn(Ab, k, scaledGrad);
					double maxUpdate = scaledGrad.norm(Vector.Norm.Infinity);
					overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
					// if (maxUpdate < tol)
					// break;
				}
				// Matrix grad = new DenseMatrix(latentDim, dFb);
				// Matrix P = times(U, Ab);
				// bound(P, 0.05, .95);
				// for (int k = 0; k < latentDim; k++) {
				// for (int l = 0; l < dFb; l++) {
				// for (int i = 0; i < m; i++) {
				// double val = grad.get(k, l);
				// double newVal = Fb.get(i, l) * U.get(i, k)
				// / P.get(i, l) - (1 - Fb.get(i, l))
				// * U.get(i, k) / (1 - P.get(i, l));
				// grad.set(k, l, val + newVal);
				// }
				// }
				// }
				// Matrix scaledGrad = times(grad, stepSize);
				// double maxUpdate = scaledGrad.norm(Matrix.Norm.Maxvalue);
				// overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				// if (maxUpdate < tol)
				// break;
			}

			// updates Bb, parameter for Gb
			for (int tB = 0; tB < maxIter; tB++) {
				for (int k = 0; k < dGb; k++) {
					// updates column k of Gb
					Vector grad = new DenseVector(latentDim);
					for (int j = 0; j < n; j++) {
						Vector Vj = getRow(V, j);
						Vector Bk = getColumn(Bb, k);
						Vector newVal = Vj.scale(Gb.get(j, k));
						newVal.add(-1,
								Vj.scale(1 - 1 / (1 + Math.exp(Vj.dot(Bk)))));
						grad.add(newVal);
					}
					Vector scaledGrad = times(grad, stepSize);
					addToColumn(Bb, k, scaledGrad);
					double maxUpdate = scaledGrad.norm(Vector.Norm.Infinity);
					overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
					// if (maxUpdate < tol)
					// break;
				}
				// Matrix grad = new DenseMatrix(latentDim, dGb);
				// Matrix P = times(V, Bb);
				// bound(P, 0.05, .95);
				// for (int k = 0; k < latentDim; k++) {
				// for (int l = 0; l < dGb; l++) {
				// for (int j = 0; j < n; j++) {
				// double val = grad.get(k, l);
				// double newVal = Gb.get(j, l) * V.get(j, k)
				// / P.get(j, l) - (1 - Gb.get(j, l))
				// * V.get(j, k) / (1 - P.get(j, l));
				// grad.set(k, l, val + newVal);
				// }
				// }
				// }
				// Matrix scaledGrad = times(grad, stepSize);
				// Bb.add(scaledGrad);
				// double maxUpdate = scaledGrad.norm(Matrix.Norm.Maxvalue);
				// overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				// if (maxUpdate < tol)
				// break;
			}

			// updates U
			for (int tU = 0; tU < maxIter; tU++) {
				// System.out.printf("\ttU = %d\n", tU);
				double maxUpdate = 0;
				Matrix muFn = times(U, An);
				// Matrix P = times(U, Ab);
				// bound(P, 0.05, .95);
				for (int i = 0; i < m; i++) {
					// updates U_i
					Vector grad = new DenseVector(latentDim);
					for (int j = 0; j < n; j++) {
						double rij = R.get(i, j);
						double rijHat = Rhat.get(i, j);
						if (rij != 0)
							grad.add(times(getRow(V, j), (rij - rijHat)));
					}
					grad.scale(1 / sigma2R);
					Vector scaledGrad = times(grad, stepSize);
					addToRow(U, i, scaledGrad);
					maxUpdate = maxAbs(scaledGrad);
				}

				Matrix grad2 = times(minus(Fn, muFn), transpose(An)).scale(
						1 / sigma2F);
				Matrix scaledGrad2 = times(grad2, stepSize);
				maxUpdate = Math.max(maxUpdate,
						scaledGrad2.norm(Matrix.Norm.Maxvalue));
				U.add(scaledGrad2);

				for (int i = 0; i < m; i++) {
					// updates row i of U
					Vector grad = new DenseVector(latentDim);
					for (int k = 0; k < dFb; k++) {
						Vector Ui = getRow(U, i);
						Vector Ak = getColumn(Ab, k);
						Vector newVal = Ak.scale(Fb.get(i, k));
						newVal.add(-1,
								Ak.scale(1 - 1 / (1 + Math.exp(Ui.dot(Ak)))));
						grad.add(newVal);
					}
					Vector scaledGrad = times(grad, stepSize);
					addToRow(U, i, scaledGrad);
					maxUpdate = scaledGrad.norm(Vector.Norm.Infinity);
					overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
					// if (maxUpdate < tol)
					// break;
				}
				// Matrix grad3 = new DenseMatrix(m, latentDim);
				// for (int i = 0; i < m; i++) {
				// for (int k = 0; k < latentDim; k++) {
				// for (int l = 0; l < dFb; l++) {
				// double val = grad3.get(i, k);
				// double newVal = Fb.get(i, l) * Ab.get(k, l)
				// / P.get(i, l) - (1 - Fb.get(i, l))
				// * Ab.get(k, l) / (1 - P.get(i, l));
				// grad3.set(i, k, val + newVal);
				// }
				// }
				// }
				// maxUpdate = Math.max(maxUpdate,
				// grad3.norm(Matrix.Norm.Maxvalue));
				// U.add(grad3);

				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}
			Rhat = times(U, transpose(V));
			sigma2R = sparseSquaredFrobeniusNormOfDiff(R, Rhat)
					/ nObservedRelations;
			sigma2F = squaredFrobeniusNorm(minus(Fn, times(U, An))) / m / dFn;

			// updates V
			for (int tV = 0; tV < maxIter; tV++) {
				// System.out.printf("\ttV = %d\n", tV);
				double maxUpdate = 0;
				Matrix muGn = times(V, Bn);
				// Matrix P = times(V, Bb);
				// bound(P, 0.05, .95);
				for (int j = 0; j < n; j++) {
					// updates V_j
					Vector grad = new DenseVector(latentDim);
					for (int i = 0; i < m; i++) {
						double rij = R.get(i, j);
						double rijHat = Rhat.get(i, j);
						if (rij != 0)
							grad.add(times(getRow(U, i), (rij - rijHat)));
					}
					grad.scale(1 / sigma2R);
					Vector scaledGrad = times(grad, stepSize);
					addToRow(V, j, scaledGrad);
					maxUpdate = maxAbs(scaledGrad);
				}

				Matrix grad2 = times(minus(Gn, muGn), transpose(Bn)).scale(
						1 / sigma2G);
				Matrix scaledGrad2 = times(grad2, stepSize);
				maxUpdate = Math.max(maxUpdate,
						scaledGrad2.norm(Matrix.Norm.Maxvalue));
				V.add(scaledGrad2);

				for (int j = 0; j < n; j++) {
					// updates row j of V
					Vector grad = new DenseVector(latentDim);
					for (int k = 0; k < dGb; k++) {
						Vector Vj = getRow(V, j);
						Vector Bk = getColumn(Bb, k);
						Vector newVal = Bk.scale(Gb.get(j, k));
						newVal.add(-1,
								Bk.scale(1 - 1 / (1 + Math.exp(Vj.dot(Bk)))));
						grad.add(newVal);
					}
					Vector scaledGrad = times(grad, stepSize);
					addToRow(V, j, scaledGrad);
					maxUpdate = scaledGrad.norm(Vector.Norm.Infinity);
					overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
					// if (maxUpdate < tol)
					// break;
				}
				// Matrix grad3 = new DenseMatrix(n, latentDim);
				// for (int j = 0; j < n; j++) {
				// for (int k = 0; k < latentDim; k++) {
				// for (int l = 0; l < dGb; l++) {
				// double val = grad3.get(j, k);
				// double newVal = Gb.get(j, l) * Bb.get(k, l)
				// / P.get(j, l) - (1 - Gb.get(j, l))
				// * Bb.get(k, l) / (1 - P.get(j, l));
				// grad3.set(j, k, val + newVal);
				// }
				// }
				// }
				// maxUpdate = Math.max(maxUpdate,
				// grad3.norm(Matrix.Norm.Maxvalue));
				// V.add(grad3);

				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}
			Rhat = times(U, transpose(V));
			sigma2R = sparseSquaredFrobeniusNormOfDiff(R, Rhat)
					/ nObservedRelations;
			sigma2G = squaredFrobeniusNorm(minus(Gn, times(V, Bn))) / n / dGn;
			iR.add(Rhat);

			if (overallMaxUpdate < tol)
				break;

			if (t == maxIter - 1)
				System.out
						.printf("Coordinate descent did not converge. Last max update = %f.\n",
								overallMaxUpdate);
		}

		Matrix Rhat = times(U, transpose(V));
		return new MatrixFactorizationResult(Rhat, U, V, An, Bn, An, Bb,
				sigma2R, sigma2F, sigma2G, iR);
	}

}